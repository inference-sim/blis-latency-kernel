package latencykernel

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Inputs are the documents a kernel is built from. They arrive already parsed: loading
// belongs to blis-schemas, and New takes documents rather than paths so a caller holding
// structs — a config search producing deployment variants never written to disk, a test, a
// generator — does not have to serialize them first. Open is the path-based half of the
// pair.
//
// New DOES validate them, at the field layer, which is the half of blis-schemas' two-layer
// validation that is version-independent and always the author's to fix. See New.
type Inputs struct {
	Scenario *scenario.Scenario
	// Deployment is the tunable configuration applied to the scenario: the pools that lay
	// the model out. It is a separate document because a scenario fixes the immutable
	// problem and an optimizer sweeps deployments against it.
	Deployment *deployment.Deployment
	// PoolIndex selects which pool of a disaggregated deployment this kernel prices. Each
	// pool runs its own engine with its own settings, so one kernel per pool.
	PoolIndex    int
	Model        *model.Graph
	Chip         *hardware.Chip
	Fabric       *hardware.Fabric
	Devices      []*hardware.StorageDevice
	Coefficients []*coefficient.Set
	// Rules carry the engine-version behaviour the resolver needs.
	Rules resolve.EngineRules
}

// New resolves the inputs into a kernel.
//
// Everything that can be decided once is decided here: which coefficients apply, which
// requested settings the layout overrides, and the model graph flattened to a per-kind
// plan. The result holds no mutable state, so its methods are pure and safe to call
// concurrently.
//
// Construction is deliberately strict. A missing coefficient, an unresolvable condition or
// a non-positive rate is an error rather than a default, because each would otherwise
// produce a step time that looks plausible and is wrong by whatever the term contributes.
// validateDocuments extends that strictness to the documents themselves.
func New(in Inputs) (*Kernel, error) {
	if in.Scenario == nil || in.Deployment == nil || in.Model == nil || in.Chip == nil {
		return nil, fmt.Errorf(
			"a kernel needs a scenario, a deployment, a model graph and a chip")
	}
	if in.PoolIndex < 0 || in.PoolIndex >= len(in.Deployment.Pools) {
		return nil, fmt.Errorf("pool index %d is outside the deployment's %d pool(s)",
			in.PoolIndex, len(in.Deployment.Pools))
	}
	if in.Rules == nil {
		return nil, fmt.Errorf("a kernel needs engine rules for version %q",
			in.Scenario.EngineVersion)
	}
	if err := validateDocuments(in); err != nil {
		return nil, err
	}
	// A deep copy: the engine block's optional settings are pointers, and a kernel that
	// shared them with the caller's documents would change its answers when the caller
	// edited a deployment it had already priced -- the config-search shape Inputs exists
	// for. The kernel's purity is a promise about its own state, so it owns that state.
	pool := clonePool(in.Deployment.Pools[in.PoolIndex])

	fab := resolve.Fabric{IntraNodeBwGBps: in.Chip.IntraNodeBwGBps}
	if in.Fabric != nil {
		fab.InterNodeBwGBps = in.Fabric.InterNodeBwGBps
	}
	// A rack is one NVLink domain where the chip says so: the catalog records a rack tier
	// only on parts whose fabric spans chassis.
	fab.RackIsOneDomain = in.Chip.GPUsPerRack > 0 && in.Chip.IntraRackBwGBps > 0

	layout, err := resolve.ResolveLayout(in.Scenario, pool, fab, in.Rules)
	if err != nil {
		return nil, fmt.Errorf("resolving the layout: %w", err)
	}

	coeffs, err := resolve.Load(in.Coefficients, resolve.Scope{
		Hardware: in.Chip.Name, Model: in.Model.Name,
		TP: layout.TP, EP: layout.ExpertWidth, NodesSpanned: layout.NodesSpanned,
	})
	if err != nil {
		return nil, fmt.Errorf("resolving coefficients: %w", err)
	}

	k := &Kernel{layout: layout, fabric: fab, pool: pool, chip: *in.Chip}
	k.tp = float64(max(layout.TP, 1))

	// The served weight format, which the scenario may override. A bf16 checkpoint
	// served fp8 reads half the weight bytes, runs against a different compute peak and
	// sits on a different efficiency envelope, so this has to be resolved before
	// anything is priced.
	served, err := servedDType(in.Model.Global.WeightDType, pool.Engine.Quantization)
	if err != nil {
		return nil, err
	}
	k.servedDType = served

	_, _, _, attnKind := kvGeometry(in.Model)
	if err := checkCacheDTypes(pool.Engine.CacheDType, pool.Engine.MambaCacheDType,
		attnKind); err != nil {
		return nil, err
	}
	cacheBytes := cacheDTypeBytes(pool.Engine.CacheDType, served)
	stateBytes := cacheDTypeBytes(pool.Engine.MambaCacheDType, model.DTypeFP32)

	// PIPELINE PARALLELISM IS NOT PRICED. No collective here spans the pipeline axis, a
	// stage's layers are not split out of the stack, and the rank-layout strides assume
	// pp = 1 (groupStride), so a pp > 1 deployment would be priced as one stage holding the
	// whole model. Refused rather than priced as something it is not.
	if pool.Parallel.PP > 1 {
		return nil, fmt.Errorf("pipeline parallelism (pp %d) is not priced by this kernel: "+
			"it has no pipeline-stage split or inter-stage transfer, and pricing the whole "+
			"model on one stage would misstate every term", pool.Parallel.PP)
	}

	plan, err := price.BuildPlan(in.Model, emitter{layout}, served.Bytes(), stateBytes)
	if err != nil {
		return nil, fmt.Errorf("planning the model graph: %w", err)
	}
	k.plan = plan

	// The recurrent cache mode as it runs. The config's default is "none"
	// (vllm/config/cache.py:190), and for a hybrid model vLLM v0.31.0 turns "none" into
	// "align" whenever prefix caching is on (vllm/model_executor/models/config.py:640-642,
	// run for every hybrid model from vllm/config/vllm.py:2797-2798) -- a STATED "none"
	// included, since the hook cannot tell it from the default. Prefix caching is on unless
	// disabled (vllm/config/cache.py:142).
	hybrid := false
	for _, l := range plan.Layers {
		hybrid = hybrid || l.RecurrentKind != ""
	}
	prefixCaching := true
	if pc := pool.Engine.EnablePrefixCaching; pc != nil && !*pc {
		prefixCaching = false
	}
	stated := price.RecurrentCacheMode(pool.Engine.MambaCacheMode)
	k.recurrentCacheMode = stated
	if stated == "" {
		k.recurrentCacheMode = price.RecurrentCacheNone
	}
	if hybrid && prefixCaching && k.recurrentCacheMode == price.RecurrentCacheNone {
		k.recurrentCacheMode = price.RecurrentCacheAlign
	}
	if hybrid {
		switch {
		case stated == "":
			k.assume("mamba_cache_mode", string(k.recurrentCacheMode),
				"vLLM v0.31.0's choice for a hybrid model: align with prefix caching on, "+
					"none with it off (vllm/model_executor/models/config.py:640-642)")
		case stated != k.recurrentCacheMode:
			k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
				Field: "mamba_cache_mode", Requested: string(stated),
				Resolved: string(k.recurrentCacheMode),
				Reason: "with prefix caching on, vLLM v0.31.0 runs a hybrid model's " +
					"recurrent cache in align mode whatever none was stated " +
					"(vllm/model_executor/models/config.py:640-642)",
			})
		}
	}

	k.blockSize = pool.Engine.BlockSize
	dsa := isDSA(in.Model)
	k.blockSizeFinal = !hybrid && (k.blockSize > 0 || dsa)
	switch {
	case dsa && k.blockSize > 0 && k.blockSize%dsaBlockSize != 0:
		return nil, fmt.Errorf("block_size %d on a DSA sparse-MLA model: its indexer runs "+
			"on %d-token kernel blocks, so a stated size must be a multiple of %d, and the "+
			"engine refuses this layout at startup (see dsaBlockSize)",
			k.blockSize, dsaBlockSize, dsaBlockSize)
	case dsa && k.blockSize <= 0:
		k.blockSize = dsaBlockSize
		k.assume("block_size", strconv.Itoa(dsaBlockSize),
			"what vLLM v0.31.0 picks for a DSA sparse-MLA model, whose indexer accepts "+
				"only 64-token blocks (vllm/v1/attention/backends/mla/indexer.py:203-204; "+
				"vllm/platforms/interface.py:666-689)")
	case k.blockSize <= 0:
		k.blockSize = 16
		k.assume("block_size", "16",
			"CacheConfig.DEFAULT_BLOCK_SIZE (vllm/config/cache.py:71 at v0.31.0); the "+
				"platform substitutes a size every attention backend supports where the "+
				"default is not one, and aligns a hybrid model's block to its recurrent "+
				"page even over a stated size (vllm/platforms/interface.py:683-702), "+
				"neither of which the kernel can see")
	}
	if err := k.resolveContextParallel(in); err != nil {
		return nil, err
	}

	if err := k.lift(coeffs, in.Model, cacheBytes); err != nil {
		return nil, err
	}
	k.tiers = map[string]hardware.StorageDevice{}
	for _, d := range in.Devices {
		if d != nil {
			k.tiers[d.Name] = *d
		}
	}
	// The engine behaviour priced here is vLLM v0.31.0's, whatever release the scenario
	// declares; the rules pack follows the declared release. Where they differ, a behaviour
	// that changed between the two is priced at v0.31.0, and that is disclosed rather than
	// left for a reader to infer from the version string.
	if v := in.Scenario.EngineVersion; v != EngineBehaviourVersion {
		k.assume("engine_version", EngineBehaviourVersion,
			"the scenario declares vLLM "+v+" and its rules pack is that release's, but "+
				"the engine behaviour this kernel encodes -- backend selection, defaults, "+
				"refusals -- is v"+EngineBehaviourVersion+"'s (README, Sources of truth)")
	}
	k.buildProvenance(coeffs)
	k.fixed = k.computeFixedBytes(in.Model)
	return k, nil
}

// resolveContextParallel settles the decode-context knobs and refuses the context-parallel
// layouts vLLM v0.31.0 cannot run for THIS model. blis-schemas refuses the layouts that are
// inadmissible for every model (parallel.py:563-578 and the world size); what remains
// depends on the model's attention, which only the graph knows.
func (k *Kernel) resolveContextParallel(in Inputs) error {
	pl := k.pool.Parallel

	// PCP RUNS ONLY WHERE EVERY LAYER'S BACKEND SUPPORTS IT, which at v0.31.0 means a
	// stack of latent attention and nothing else. The engine asserts it per layer at startup
	// (vllm/v1/worker/cp_utils.py:35-38, "PCP requires attention backend support"), and a
	// backend reports support through its implementation class:
	//
	//   - attention implementations default to no (AttentionImplBase.supports_pcp = False,
	//     vllm/v1/attention/backend.py:843) and only MLAAttentionImpl says yes (:1048);
	//   - recurrent backends -- Mamba1, Mamba2, GDN, linear attention, and Kimi-K3's KDA,
	//     which subclasses GDN -- declare no implementation class at all, so the lookup
	//     raises and support reads as no (:230-235).
	//
	// So a stack with any full-attention, sliding-window or recurrent layer does not start
	// under pcp > 1, whichever backend is chosen, and pricing it would describe nothing the
	// engine runs. Model Runner V2, which runs PCP, says the same in one line: "MRV2 PCP
	// currently supports MLA models only" (vllm/v1/worker/gpu/pcp_manager.py:132-133).
	//
	// Admitting every latent stack is the other edge, and it is not exact: a latent layer
	// served by a backend that declares no implementation class refuses PCP too, and
	// DeepSeek-V4's sparse-MLA backends are such (vllm/models/deepseek_v4/sparse_mla.py).
	// The graph cannot name the backend, so that refusal is the engine's to make.
	if pl.PCP > 1 {
		for _, l := range k.plan.Layers {
			var kind string
			switch {
			case l.AttnQHeads > 0 && !latentAttention(l.AttnKind):
				kind = string(l.AttnKind) + " attention"
			case l.RecurrentKind != "":
				kind = "a " + string(l.RecurrentKind) + " recurrent mixer"
			default:
				continue
			}
			return fmt.Errorf(
				"prefill-context parallelism (pcp %d) needs every layer's backend to support "+
					"it, and layer kind %q has %s, whose backend does not: the engine "+
					"refuses this layout at startup", pl.PCP, l.ID, kind)
		}
	}

	// PCP WITH DCP RUNS ONLY ON DSA SPARSE-MLA LAYERS. An MLA attention layer refuses the
	// combination unless its class opts in -- "Under PCP+DCP only the decode rows carry an
	// LSE; the base forward merges a full-batch LSE, so subclasses opt in"
	// (MLAAttention.supports_pcp_dcp = False, vllm/model_executor/layers/attention/
	// mla_attention.py:440-442, raised at :684-687) -- and the one class that opts in is
	// DeepseekV32Attention (vllm/models/deepseek_v32/attention.py:121-124), the DSA layer
	// the catalog states as sparse_mla. A plain mla layer therefore does not start.
	//
	// The converse is not certain: a sparse_mla layer served by another class may refuse
	// too, and the kernel cannot see the class. It is admitted here because the DSA
	// family -- DeepSeek-V3.2, GLM-5 -- is what the kind describes, and it is stated
	// rather than assumed silent.
	if pl.PCP > 1 && pl.DCP > 1 {
		for _, l := range k.plan.Layers {
			if l.AttnQHeads > 0 && l.AttnKind != model.AttentionSparseMLA {
				return fmt.Errorf(
					"prefill- and decode-context parallelism together (pcp %d, dcp %d) run "+
						"only on DSA sparse-MLA attention, and layer kind %q is %s: its "+
						"attention class does not support PCP with DCP, so the engine "+
						"refuses this layout at startup", pl.PCP, pl.DCP, l.ID, l.AttnKind)
			}
		}
	}

	// A REPLICATED QUERY PROJECTION IS NOT PRICED, so it is refused where it would take
	// effect rather than priced half-way. dcp_q_replicate skips the decode query
	// all-gather, and pays for it by building the MLA query projection over tp/dcp ranks
	// instead of tp (DCPGroupColumnParallelLinear, vllm/model_executor/layers/linear.py:
	// 632-659), so each rank holds and computes dcp times that projection. Crediting the
	// skipped gather without the replicated GEMM would make the knob look free; charging
	// the GEMM needs to know which projection is the query's, and the graph identifies
	// GEMMs by shape, not purpose. It takes effect only with dcp > 1, pcp <= 1
	// (vllm/model_executor/models/deepseek_v2.py:1072-1076) and on a latent layer, so
	// everywhere else the request is inert and is accepted.
	if q := k.pool.Engine.DCPQReplicate; q != nil && *q && pl.DCP > 1 && pl.PCP <= 1 {
		for _, l := range k.plan.Layers {
			if l.AttnQHeads > 0 && latentAttention(l.AttnKind) {
				return fmt.Errorf(
					"dcp_q_replicate is requested with dcp %d on a latent-attention model; "+
						"the replicated query projection it costs is not priced by this "+
						"kernel, so the knob is refused rather than credited for the "+
						"collective it saves", pl.DCP)
			}
		}
	}

	// The release's own list of backend names, where the rules value is a blis-schemas pack.
	// resolve.EngineRules cannot ask for it -- rules.Pack carries the list as a field, with no
	// method an interface could name -- so any other rules value leaves the release's check
	// undone, and that is disclosed rather than silent: the name is still checked against
	// what this kernel prices, but not against what the engine release accepts.
	var accepted map[string]bool
	if pack, ok := in.Rules.(*rules.Pack); ok && pack != nil {
		accepted = pack.DCPCommBackends
	}
	if b := k.pool.Engine.DCPCommBackend; b != "" && accepted == nil {
		k.assume("dcp_comm_backend", b,
			"stated, and checked against the backends this kernel prices, but not against "+
				"the names the engine release accepts: the rules value carries no list")
	}
	dc, overrides, err := resolve.ResolveDecodeContext(
		k.pool, in.Deployment, k.blockSize, k.blockSizeFinal, accepted)
	if err != nil {
		return fmt.Errorf("resolving decode-context parallelism: %w", err)
	}
	k.decodeContext = dc
	k.layout.Overrides = append(k.layout.Overrides, overrides...)
	if pl.DCP > 1 && k.pool.Engine.DCPCommBackend == "" {
		k.assume("dcp_comm_backend", dc.CommBackend,
			"the engine's stock default (ParallelConfig.set_dcp_defaults, "+
				"vllm/config/parallel.py:581-594 at v0.31.0), with the query projection not "+
				"replicated; a model's configuration hook runs first and may choose "+
				"otherwise -- GlmMoeDsaForCausalLM selects a2a with dcp_q_replicate "+
				"(vllm/model_executor/models/config.py:43-50) -- and the kernel cannot see "+
				"which model class serves this graph, so state both to price it exactly. On "+
				"that model, where FlashMLA-sparse serves the layout, the difference "+
				"decides whether it starts: that backend runs DCP only with ag_rs "+
				"(flashmla_sparse.py:416-422)")
	}
	return nil
}

// dsMLAStateBytes is the packed page cell, in bytes per token per layer, of the cache
// layouts that state their own size.
var dsMLAStateBytes = map[string]float64{"fp8_ds_mla": 656, "nvfp4_ds_mla": 352}

// latentAttention reports whether an attention kind keeps a latent (MLA) cache, which is
// what the engine's MLA attention implementations serve.
func latentAttention(kind model.AttentionKind) bool {
	return kind == model.AttentionMLA || kind == model.AttentionSparseMLA
}

// clonePool copies a pool and everything its engine block points to, so the copy shares no
// memory with the original. The pointees are flat structs and tri-state flags, so a field
// copy of each is a deep copy.
func clonePool(p deployment.Pool) deployment.Pool {
	e := &p.Engine
	for _, b := range []**bool{&e.DisableCustomAllReduce, &e.AsyncScheduling,
		&e.DisableCascadeAttn, &e.EnablePrefixCaching, &e.DCPQReplicate} {
		if *b != nil {
			v := **b
			*b = &v
		}
	}
	if e.DBO != nil {
		v := *e.DBO
		e.DBO = &v
	}
	if e.EPLB != nil {
		v := *e.EPLB
		e.EPLB = &v
	}
	if e.Speculative != nil {
		v := *e.Speculative
		e.Speculative = &v
	}
	return p
}

// emitter adapts a layout to the plan builder's interface.
type emitter struct{ l resolve.Layout }

func (e emitter) Emits(c model.EmitCondition) bool      { return e.l.Emits(c) }
func (e emitter) Recognizes(c model.EmitCondition) bool { return resolve.Recognizes(c) }

// lift copies the coefficients the hot path needs into fields, so a step reads a struct
// rather than hashing strings. Each lookup that has no default is required: a rate the
// kernel cannot resolve is an error, not a zero.
func (k *Kernel) lift(c *resolve.Coefficients, g *model.Graph, cacheBytes float64) error {
	derate, err := c.Value("hbm_derate")
	if err != nil {
		return err
	}
	k.hbmBytesPerSecond = k.chip.MemoryBandwidthTBs * 1e12 * derate
	if err := mustPositive("HBM bandwidth", k.hbmBytesPerSecond); err != nil {
		return err
	}

	// Compute peak and the efficiency ramp are selected by the weight dtype: the three
	// formats in the source data reach materially different asymptotes, so a single ramp
	// would misprice two of them.
	suffix, peak := dtypeFit(k.servedDType, k.chip)
	if err := mustPositive("compute peak", peak); err != nil {
		return err
	}
	k.computeFLOPsPerSecond = peak
	if k.epsMax, err = c.Value("gemm_eps_max_" + suffix); err != nil {
		return fmt.Errorf("no efficiency asymptote for served dtype %s: %w",
			k.servedDType, err)
	}
	if k.mHalf, err = c.Value("gemm_m_half_" + suffix); err != nil {
		return fmt.Errorf("no efficiency half-max for served dtype %s: %w",
			k.servedDType, err)
	}
	// The shape-aware ramp is opt-in by the presence of its coefficients, and both are
	// required together: a k factor without an n factor is a different law, not a partial
	// one, and fitting one of the two leaves the other's work absorbed into epsMax.
	nHalf, nErr := c.Value("gemm_n_half_" + suffix)
	kHalf, kErr := c.Value("gemm_k_half_" + suffix)
	switch {
	case nErr == nil && kErr == nil:
		k.nHalf, k.kHalf, k.gemmShapeAware = nHalf, kHalf, true
	case nErr == nil || kErr == nil:
		return fmt.Errorf(
			"served dtype %s has one of gemm_n_half/gemm_k_half but not both; the "+
				"shape-aware ramp needs both or neither", k.servedDType)
	}

	k.nvlinkBytesPerSecond = k.chip.IntraNodeBwGBps * 1e9
	k.nicBytesPerSecond = k.fabric.InterNodeBwGBps * 1e9
	if k.nicBytesPerSecond <= 0 {
		// No fabric: a single-node deployment, where no collective leaves the node. The
		// rate is unused, and setting it to the on-node rate keeps a stray cross-node term
		// finite rather than infinite.
		k.nicBytesPerSecond = k.nvlinkBytesPerSecond
	}

	// Collective floors are per operation, dtype and rank count. All three axes
	// matter and the measurements say so: on H200 an all-reduce floor is 15.98 us
	// against an all-to-all's 7.39, and an all-reduce floor grows 6.0 / 9.4 / 16.0
	// with 2 / 4 / 8 ranks where an all-to-all's barely moves. A single floor for
	// every collective is wrong in both directions at once, so each op resolves its
	// own and they are looked up once here rather than per step.
	if err := k.liftCollectiveFloors(c); err != nil {
		return err
	}

	// The CPU-to-GPU link rate, per part. NVIDIA's descriptor states it, and it is
	// not a constant across platforms: a Grace-Blackwell part connects over
	// NVLink-C2C where an HGX part uses PCIe, so an offload tier that is a liability
	// on one host can be affordable on the other.
	// Recurrent families, one entry per kind the registry measures. A kind with no entries
	// is left out of the map rather than defaulted to zero cost, so the difference between
	// "measured as cheap" and "not measured" stays visible.
	//
	// Every coefficient read from here to the host terms is optional: the kernel prices
	// without it. Each one the deployment needs and the scenario's sets do not carry is
	// recorded as an assumption (optional), so Provenance and Evidence show what the
	// prediction lacks. A per-kind entry is needed only where the model has a layer of that
	// kind.
	usedAttn, usedRec := map[model.AttentionKind]bool{}, map[model.RecurrentKind]bool{}
	moe := false
	for _, l := range k.plan.Layers {
		if l.AttnQHeads > 0 {
			usedAttn[l.AttnKind] = true
		}
		if l.RecurrentKind != "" {
			usedRec[l.RecurrentKind] = true
		}
		moe = moe || l.ExpertWeightBytesPerExpert > 0
	}
	k.recurrent = map[model.RecurrentKind]floorRate{}
	for kind, suffix := range map[model.RecurrentKind]string{
		model.RecurrentMamba2: "mamba2",
		model.RecurrentKDA:    "kda",
		model.RecurrentGDN:    "gdn",
	} {
		without := "no measured decode form for the " + suffix + " mixer, so its decode " +
			"term is not charged"
		floor := k.optional(c, "recurrent_decode_floor_"+suffix, usedRec[kind], without)
		rate := k.optional(c, "recurrent_decode_rate_"+suffix, usedRec[kind], without)
		if floor <= 0 || rate <= 0 {
			continue
		}
		k.recurrent[kind] = floorRate{
			floor: time.Duration(floor * float64(time.Microsecond)),
			rate:  rate * 1e6, // tokens per microsecond to tokens per second
		}
	}

	// The attention primitive's measured floor and rate. Absent for a part with no
	// generation-attention sweep, in which case the KV read is charged at HBM bandwidth
	// with no floor, and the absence is recorded.
	k.attentionFloor = time.Duration(k.optional(c, "attention_decode_floor",
		len(usedAttn) > 0, "the per-layer attention decode floor is not charged") *
		float64(time.Microsecond))
	k.attentionRate = k.optional(c, "attention_decode_rate", len(usedAttn) > 0,
		"no measured attention decode rate, so the KV read is charged at HBM "+
			"bandwidth and no attention floor is charged") * 1e6

	// Per-attention-kind decode terms, where the registry carries them. A kind absent from the
	// registry falls back to the unsuffixed pair above, so a deployment with no per-kind fit
	// prices exactly as it did before this map existed.
	//
	// The kinds are different KERNELS, not one kernel at different settings: full attention
	// reads the whole context, a sliding window reads a bounded slice of it. NVIDIA's own
	// sweeps collect them separately for that reason, and fitted separately a windowed kernel
	// sustains 1.1x to 2.8x less effective bandwidth than a full-attention one on the same
	// silicon (0.26 against 0.58 of peak on H200). Pricing both with the full-attention rate
	// makes every windowed layer too cheap, and gpt-oss-120b is half windowed layers.
	//
	// Naming follows the recurrent terms, which already carry a per-kind suffix. That is why
	// no schema change is needed: coefficient.Scope has no kind axis and does not need one.
	// MLA joins the map for the same reason SWA did: its kernel reads a different number
	// of bytes per token and sustains a different fraction of peak, so pricing it with the
	// full-attention rate is wrong by construction rather than by a little. The WIDTH comes
	// from the graph -- blis-catalog declares an MLA node with `n_kv: 1` and d_h the latent
	// width (576 = kv_lora_rank + qk_rope_head_dim for the DeepSeek-V3 family and Kimi, 512
	// for DeepSeek-V4), so kvGeometry's NumKVHeads*HeadDim is that width --
	// and KVBytesPerToken charges it ONCE per token, since a latent cache has no value
	// tensor. The RATE is the other half: an MLA decode sustains 0.61-0.80 of datasheet
	// bandwidth against full attention's 0.52-0.88.
	//
	// The FLOOR is not what distinguishes them, which an earlier version of this comment
	// had backwards: it claimed an MLA floor "several times larger (51.5-89.5us against
	// 9.5-19.5us) because the per-call setup reads a latent cache". The registry says
	// otherwise -- attention_decode_floor_mla is 9.5-14.5us against the part-wide
	// 9.5-19.0us -- because that entry IS the part's own measured attention floor reused
	// for the kind ("this is this part's own measured attention-kernel decode floor,
	// reused for the MLA kind", cost-model-attention.yaml). The larger module-derived
	// floors, 4.7x-6.2x those, were measured and REJECTED: they cover the whole MLA block
	// including down-projections this kernel prices separately as GEMM nodes, and charging
	// them took kimi-k2.5's TPOT error from 6.38% to 14.57%.
	//
	// Nine models in the pinned catalog declare an mla or sparse_mla layer, and two of them
	// (deepseek-v4-pro, kimi-k3) appear in both the FPM dataset and the InferenceX corpus,
	// so this is on the scored path rather than hypothetical.
	//
	// sparse_mla is now mapped too, and its pair is asymmetric: the registry ships a
	// RATE and deliberately no floor.
	//
	// The reason is recorded in blis-registry scripts/fit_attention_sparse_mla.py.
	// Searching a sparse floor freely lands at 12.6-14.4us on every part -- a band that
	// does not track the part-wide floors it would replace -- and pinning it to each
	// part's own measured attention floor costs under 8% of fit. Adding the parameter
	// would re-open the failure that `attention_decode_floor_mla` shipped with: a floor
	// fitted from a MODULE table, charging projections the catalog already prices as
	// separate GEMM nodes, which cost 2.45 points of end-to-end TPOT before
	// correct_mla_floor.py undid it.
	//
	// So a kind may supply a rate alone and inherit the part-wide floor. A floor without its
	// rate is not used -- the kind is priced with the part-wide pair -- and the missing rate
	// is recorded, so the floor's listing in Provenance is not mistaken for its use.
	k.attentionByKind = map[model.AttentionKind]floorRate{}
	for kind, suffix := range map[model.AttentionKind]string{
		model.AttentionSWA:       "swa",
		model.AttentionMLA:       "mla",
		model.AttentionSparseMLA: "sparse_mla",
	} {
		floor := c.ValueOr("attention_decode_floor_"+suffix, 0)
		rate := k.optional(c, "attention_decode_rate_"+suffix, usedAttn[kind],
			"no per-kind decode rate for "+suffix+" attention, so it is priced with the "+
				"part-wide attention pair and any per-kind floor is not used")
		if rate <= 0 {
			continue
		}
		if floor <= 0 {
			// Inherit the part-wide measured floor, which is what the sparse fit was
			// conditioned on. Lifted after k.attentionFloor is set, so this is the same
			// number the fitter pinned to.
			k.attentionByKind[kind] = floorRate{
				floor: k.attentionFloor, rate: rate * 1e6,
			}
			continue
		}
		k.attentionByKind[kind] = floorRate{
			floor: time.Duration(floor * float64(time.Microsecond)),
			rate:  rate * 1e6, // bytes per microsecond to bytes per second
		}
	}
	k.attentionPrefillFloor = time.Duration(k.optional(c, "attention_prefill_floor",
		len(usedAttn) > 0, "the per-layer attention prefill floor is not charged") *
		float64(time.Microsecond))
	k.attentionPrefillScale = k.optional(c, "attention_prefill_work_scale",
		len(usedAttn) > 0, "no measured attention prefill scale, so prefill attention "+
			"is charged on the unmodified GEMM efficiency ramp with no floor")

	k.hostBytesPerSecond = k.optional(c, "host_link_bandwidth", true,
		"no host-link rate, so an offload transfer is not capped at it") * 1e6
	k.moeImbalance = 1
	if v := k.optional(c, "moe_routing_imbalance_median", moe,
		"no measured routing imbalance, so experts are priced as evenly loaded"); v > 0 {
		k.moeImbalance = v
	}

	hostWithout := func(what string) string {
		return "no measured " + what + ", so that host cost is charged at zero"
	}
	k.admissionPerToken = time.Duration(k.optional(c, "host_admission_per_token", true,
		hostWithout("admission cost per token")) * float64(time.Microsecond))
	k.admissionPerRequest = time.Duration(k.optional(c, "host_admission_per_request", true,
		hostWithout("admission cost per request")) * float64(time.Microsecond))
	k.outputTokenCost = time.Duration(k.optional(c, "host_output_token", true,
		hostWithout("per-output-token cost")) * float64(time.Microsecond))
	k.completionCost = time.Duration(k.optional(c, "host_completion", true,
		hostWithout("completion cost")) * float64(time.Microsecond))
	k.launchPerLayer = time.Duration(k.optional(c, "host_launch_eager_per_layer", true,
		hostWithout("eager launch cost per layer")) * float64(time.Microsecond))
	// Per-kernel dispatch, which dominates a single-request decode step. Charged at zero
	// when the registry carries none, like the other host terms here -- which removes the
	// term two published runs say is most of a small step's cost, so a registry that
	// carries the host set is what a scored deployment needs, and its absence is recorded.
	k.launchPerKernel = time.Duration(k.optional(c, "host_launch_per_kernel", true,
		hostWithout("per-kernel launch cost")) * float64(time.Microsecond))
	k.replayPerStep = time.Duration(k.optional(c, "host_replay_graph_per_step", true,
		hostWithout("graph replay cost")) * float64(time.Microsecond))
	// How much of a step a captured graph covers, which sets the launch count rather
	// than only whether capture happens. The mode decides it PER BATCH: vLLM v0.31.0
	// documents each (vllm/config/compilation.py:604-640) and dispatches each step by
	// whether it is a uniform decode batch (vllm/v1/cudagraph_dispatcher.py:233-310):
	//
	//	PIECEWISE           piecewise for every step
	//	FULL                one full graph for every step
	//	FULL_AND_PIECEWISE  full for a uniform decode batch, piecewise otherwise
	//	FULL_DECODE_ONLY    full for a uniform decode batch, NO graph otherwise
	//	NONE                no graph
	//
	// FULL_AND_PIECEWISE is v0.31.0's default ("(v1 default)", compilation.py:614), so
	// that is what an unstated mode resolves to, and the kernel says so in Provenance:
	// the engine downgrades it where an attention backend cannot capture a full graph,
	// which the kernel cannot see.
	mode := strings.ToUpper(k.pool.Engine.CUDAGraphMode)
	if mode == "" {
		mode = "FULL_AND_PIECEWISE"
		k.assume("cudagraph_mode", mode,
			"vLLM v0.31.0's default (vllm/config/compilation.py:614); the engine "+
				"downgrades it where an attention backend does not support full graphs, "+
				"which the kernel cannot see")
	}
	switch mode {
	case "PIECEWISE":
		k.graphDecode, k.graphOther = graphModePiecewise, graphModePiecewise
	case "FULL":
		k.graphDecode, k.graphOther = graphModeFull, graphModeFull
	case "FULL_AND_PIECEWISE":
		k.graphDecode, k.graphOther = graphModeFull, graphModePiecewise
	case "FULL_DECODE_ONLY":
		k.graphDecode, k.graphOther = graphModeFull, graphModeEager
	case "NONE":
		k.graphDecode, k.graphOther = graphModeEager, graphModeEager
	default:
		return fmt.Errorf(
			"cudagraph_mode %q is not one this kernel prices; add it with its launch "+
				"count rather than letting it fall through",
			k.pool.Engine.CUDAGraphMode)
	}
	// The capture's memory is keyed by vLLM's own mode name, lowercased: the modes hold
	// different graph sets, so FULL_AND_PIECEWISE and FULL do not share a figure even
	// where they share a decode launch count.
	k.graphModeName = strings.ToLower(mode)
	k.uniformDecodeWidth = 1
	if s := k.pool.Engine.Speculative; s != nil && s.NumSpecTokens > 0 {
		// A uniform decode batch verifies 1 + num_spec_tokens tokens per request
		// (uniform_decode_query_len, vllm/v1/worker/gpu_model_runner.py:883).
		k.uniformDecodeWidth = 1 + s.NumSpecTokens
	}

	// The largest batch a captured graph covers. vLLM v0.31.0 captures token counts up to
	// max_cudagraph_capture_size and runs anything larger with no graph at all, whatever the
	// mode (vllm/v1/worker/gpu/cudagraph_utils.py:499-529; V1's
	// vllm/v1/cudagraph_dispatcher.py:270-279). Unstated, that ceiling is
	// min(max_num_seqs x decode_query_len x 2, 512), or 1024 on a data-center Blackwell part
	// (VllmConfig._set_cudagraph_sizes, vllm/config/vllm.py:2442-2460). With speculation the
	// uniform decode sizes are appended only up to that same ceiling
	// (`n * query_len <= max_cudagraph_capture_size`, :2531-2538), so a decode step has no
	// higher one. The ceiling is then capped at max_num_batched_tokens (:2541-2542), which
	// no step exceeds, so that cap changes no verdict here. blis-schemas has no field for an
	// explicit capture size, so this is always the default.
	//
	// The catalog records no compute capability, so a data-center Blackwell part is
	// recognised by native NVFP4 support, which among the catalog's parts is exactly the
	// SM100 family (b200, b300, gb200-nvl72, gb300).
	seqs := k.pool.Engine.MaxNumSeqs
	if seqs <= 0 {
		seqs = defaultMaxNumSeqs(k.chip)
		k.assume("max_num_seqs", strconv.Itoa(seqs),
			"vLLM v0.31.0's OpenAI-API-server default for this part's memory "+
				"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887); it sets "+
				"the cudagraph capture ceiling and the MLA context-gather chunk")
	}
	if k.graphDecode != graphModeEager || k.graphOther != graphModeEager {
		platform := 512
		if k.chip.NVFP4Peak > 0 {
			platform = 1024
		}
		k.captureTokens = min(seqs*k.uniformDecodeWidth*2, platform)
		k.assume("max_cudagraph_capture_size", strconv.Itoa(k.captureTokens),
			"vLLM v0.31.0's default, min(max_num_seqs x decode_query_len x 2, "+
				strconv.Itoa(platform)+") (VllmConfig._set_cudagraph_sizes, "+
				"vllm/config/vllm.py:2442-2460); blis-schemas has no field to state it, and "+
				"a step above it runs with no graph")
	}

	// The rows one chunk of an MLA prefill's context gather covers:
	// min(max(8 x max_model_len, 4 x max_num_seqs x block_size), 65536)
	// (determine_chunked_prefill_workspace_size, vllm/model_executor/layers/attention/
	// mla_attention.py:2297-2320 at v0.31.0). An unstated max_model_len is the model's own
	// window, which the graph does not carry, so it is taken at the 65,536 cap: exact for any
	// window of 8,192 tokens or more, which every latent model in the catalog has.
	chunk := 65536
	if mml := k.pool.Engine.MaxModelLen; mml > 0 {
		chunk = min(max(8*mml, 4*seqs*k.blockSize), 65536)
	} else if k.layout.DCP > 1 {
		for _, l := range k.plan.Layers {
			if latentAttention(l.AttnKind) {
				k.assume("mla_context_chunk", strconv.Itoa(chunk),
					"max_model_len is unstated, so the MLA context-gather chunk is taken at "+
						"its 65,536-row cap, exact for a model window of 8,192 tokens or more")
				break
			}
		}
	}
	// Then rounded up to a whole number of blocks -- of lcm(block, dcp x interleave) blocks
	// under DCP, so every rank's share stays interleave-aligned
	// (align_mla_chunked_context_workspace_size, mla_attention.py:1993-2005, applied at
	// :2319).
	align := max(k.blockSize, 1)
	if k.layout.DCP > 1 {
		align = lcm(align, k.layout.DCP*max(k.decodeContext.Interleave, 1))
	}
	chunk = (max(chunk, align) + align - 1) / align * align
	k.mlaContextChunk = max(chunk, 1)

	// Expert geometry, derived once.
	experts := 0
	for _, l := range k.plan.Layers {
		if l.TopK > 0 {
			for _, lk := range g.LayerKinds {
				if lk.ID != l.ID {
					continue
				}
				for _, n := range lk.Nodes {
					if n.Op == model.OpGroupedGEMM && n.Experts > experts {
						experts = n.Experts
					}
				}
			}
		}
	}
	// How an expert's weights are divided, which depends on whether expert parallelism
	// is on. vLLM gives a rank whole experts when it is (ep_size the group width, tp_size 1)
	// and a tensor slice of every expert when it is not (ep_size 1)
	// (FusedMoEParallelConfig.make, vllm/model_executor/layers/fused_moe/config.py:1186-1214
	// at v0.31.0). So exactly one of the two axes divides an expert, never both.
	//
	// The slice is over every rank the MoE spans, not the tensor-parallel group alone: with
	// expert parallelism off vLLM flattens tp across dp and pcp, "so we shard across all
	// devices" (flatten_tp_across_dp_and_pcp, config.py:1090-1098). An earlier form divided
	// by tp, which priced a dp > 1 deployment's experts as dp independent replicas, each rank
	// holding dp times its true share.
	//
	// COVERAGE LIMIT: the flattening runs over pcp as well, and the shard here does not. With
	// pcp > 1 vLLM gives a rank a 1/(dp x pcp x tp) slice of each expert, and this charges
	// 1/(dp x tp). The token gather over those ranks is priced where it runs -- with dp > 1
	// the dispatch spans dp x pcp x tp ranks (resolve.Layout.MoEGroupWidth); with dp = 1 there
	// is no dispatch (all2all_utils.py:202-214) -- but the reduction a tensor-parallel MoE
	// runs over the wider slice is not, so dividing by pcp alone would make a PCP step's
	// experts cheaper than the engine runs them.
	k.localExpertShare = 1
	k.expertTensorShards = 1
	if k.layout.ExpertWidth <= 1 {
		// No expert parallelism: every rank holds every expert, tensor-sharded across
		// the dp x tp ranks.
		k.expertTensorShards = k.tp * float64(max(k.layout.DP, 1))
	}
	if experts > 0 {
		redundant := 0
		if k.pool.Engine.EPLB != nil && k.pool.Engine.EPLB.Enabled {
			redundant = k.pool.Engine.EPLB.NumRedundantExperts
		}
		// With expert parallelism off the width is 1, so every rank holds every expert
		// — which is right: the division is by tensor shard, applied at pricing time.
		base, heaviest, _, _ := price.ExpertsPerRank(experts+redundant, k.layout.ExpertWidth)
		if base == 0 {
			return fmt.Errorf(
				"expert-parallel width %d exceeds the model's %d physical experts, so some "+
					"ranks would hold none", k.layout.ExpertWidth, experts+redundant)
		}
		// The HEAVIEST rank is the one priced. An uneven split gives the first
		// experts % width ranks one expert more (vllm/model_executor/layers/fused_moe/
		// expert_map_manager.py:67-69 at v0.31.0), and that rank bounds both answers: its
		// memory fills first, and the dispatch and combine wait for its experts to finish.
		// An earlier form priced the lighter rank, understating minimax-m2.5 at ep72 --
		// 256 experts, 40 ranks holding 4 -- by a quarter of its expert weights.
		k.expertsPerRank = float64(heaviest)
		// The share of the model's experts that rank holds, from its count rather than as
		// 1/width.
		k.localExpertShare = float64(heaviest) / float64(experts+redundant)
		k.totalExperts = experts + redundant
	}

	// KV geometry. Both the cache dtype and the tensor-parallel width divide it, and the
	// head count floors at one because a head is never split across ranks.
	nkv, headDim, layers, kind := kvGeometry(g)
	k.kvBytesPerToken = price.KVBytesPerToken(nkv, k.layout.TP, headDim, layers, cacheBytes,
		latentAttention(kind))
	// The packed DeepSeek sparse-MLA layouts state their own page cell: fp8_ds_mla packs the
	// latent with its scales into 656 bytes a token and nvfp4_ds_mla into 352, against 576
	// elements by the per-element formula (state_content_bytes,
	// vllm/model_executor/layers/attention/mla_attention.py:1361-1363 at v0.31.0). On a DSA
	// model the layout is the one its backend allocates, which can differ from the stated
	// dtype, and a layout no backend serves is refused (sparseMLABackend). Elsewhere it is
	// taken where the deployment states one.
	layout := k.pool.Engine.CacheDType
	if isDSA(g) {
		heads := 0
		for _, l := range k.plan.Layers {
			if l.AttnKind == model.AttentionSparseMLA && l.AttnQHeads > 0 {
				heads = max(heads, l.AttnQHeads)
			}
		}
		backend, dsaLayout, err := sparseMLABackend(sparseMLABackendRequest{
			cache:        k.pool.Engine.CacheDType,
			headsPerRank: heads / max(k.layout.TP, 1),
			sm100:        k.chip.NVFP4Peak > 0,
			preHopper:    preHopperParts[k.chip.Name],
			tp:           k.layout.TP,
			dcp:          k.layout.DCP,
			pcp:          k.layout.PCP,
			dcpComm:      k.decodeContext.CommBackend,
		})
		if err != nil {
			return err
		}
		if backend == backendFlashInferSM90 {
			k.assume("sparse_mla_backend", backend,
				"chosen as vLLM v0.31.0 would choose it, which needs FlashInfer 0.6.18 or "+
					"later installed (flashinfer_mla_sparse_sm90.py:194-199)")
		}
		if stated := k.pool.Engine.CacheDType; dsaLayout != stated && stated != "" {
			k.layout.Overrides = append(k.layout.Overrides, resolve.Override{
				Field: "cache_dtype", Requested: stated, Resolved: dsaLayout,
				Reason: backend + " serves this cache as " + dsaLayout +
					" (_canonicalize_sparse_mla_kv_cache_dtype, " +
					"vllm/model_executor/layers/attention/mla_attention.py:358-375)",
			})
		}
		layout = dsaLayout
	}
	if b, ok := dsMLAStateBytes[layout]; ok && latentAttention(kind) {
		k.kvBytesPerToken = b * float64(layers)
	}
	// The layer count kvBytesPerToken is spread over. kvGeometry counts only layers that
	// HOLD KV, so this is the divisor that recovers one layer's share of the cache -- on a
	// hybrid stack it is not the total layer count, and the two differ ninefold on
	// Nemotron-3-Ultra (12 KV layers of 108).
	k.kvLayers = layers
	// Whether EVERY KV-holding layer is a kind decode-context parallelism shards. A
	// whole-model verdict, because kvBytesPerToken is a whole-model figure: see
	// SequenceVariableBytes for what that costs and why it costs nothing real.
	k.dcpShardsAllKVLayers = true
	for _, l := range k.plan.Layers {
		if l.AttnQHeads > 0 && !dcpShardsKV(l.AttnKind) {
			k.dcpShardsAllKVLayers = false
			break
		}
	}
	return k.liftMemory(c, g)
}

// liftMemory resolves the per-rank occupancy outside the KV budget, every magnitude from
// the registry. computeFixedBytes composes them.
//
// THE COMPOSITION IS NVIDIA'S, as blis-registry's cost-model-memory set states it
// (blis-registry v0.1.1, methodology section 11): per rank,
//
//	weights + activation + nccl_communicator_bytes_<N>rank + engine_workspace_bytes
//	        + cudagraph_capture_bytes_<mode>
//
// where every term ADDS -- the communicator figure is the whole collective-buffer
// reservation, the workspace is a separate allowance beside it, and neither contains the
// graph capture -- and
//
//	activation = max(activation_buffer_count_<dense|moe>_<N>rank
//	                     * max_num_batched_tokens * h * 2,
//	                 activation_scratch_floor_bytes)
//
// with N the tensor-parallel width snapped to a declared width (snapDeclared). Each
// magnitude is REQUIRED: the registry carries all of them for every catalog part, so an
// absent one means the scenario did not list cost-model-memory, and charging zero for it
// would overstate how many sequences fit by up to several gigabytes per rank.
//
// They replace three constants this kernel carried -- 512 MiB of capture, 392 MiB of
// communicator, four buffers of activation -- which no Provenance entry could see, and the
// engine workspace, which was loaded and never read.
func (k *Kernel) liftMemory(c *resolve.Coefficients, g *model.Graph) error {
	need := func(name string) (float64, error) {
		v, err := c.Value(name)
		if err != nil {
			return 0, fmt.Errorf("memory occupancy needs %s; list cost-model-memory and "+
				"cost-model-primitives in the scenario's coefficients: %w", name, err)
		}
		return v, nil
	}
	// A communicator exists wherever some collective group is wider than one rank. The
	// figure is keyed on the tensor-parallel width, as NVIDIA's descriptor keys it, and
	// snapped to a declared width; a layout with tp=1 but another group (expert, prefill-
	// or decode-context) still holds a communicator, and is charged the narrowest declared
	// one rather than none.
	if k.layout.TP > 1 || k.layout.MoEGroup() > 1 || k.layout.PCP > 1 || k.layout.DCP > 1 {
		comm, err := need(fmt.Sprintf("nccl_communicator_bytes_%drank",
			snapDeclared(c, "nccl_communicator_bytes_%drank", k.layout.TP)))
		if err != nil {
			return err
		}
		k.commBytes = int64(comm)
	}
	workspace, err := need("engine_workspace_bytes")
	if err != nil {
		return err
	}
	k.workspaceBytes = int64(workspace)

	capture, err := need("cudagraph_capture_bytes_" + k.graphModeName)
	if err != nil {
		return err
	}
	k.captureBytes = int64(capture)

	// The activation multiple is keyed by whether the model routes tokens to experts,
	// which is the only family distinction the kernel can draw without a model name.
	family := "dense"
	if k.totalExperts > 0 {
		family = "moe"
	}
	format := "activation_buffer_count_" + family + "_%drank"
	if k.activationBuffers, err = need(fmt.Sprintf(format,
		snapDeclared(c, format, min(k.layout.TP, 8)))); err != nil {
		return err
	}
	if k.activationFloor, err = need("activation_scratch_floor_bytes"); err != nil {
		return err
	}
	k.activationWidth = activationWidth(k.plan, g)

	k.batchedTokens = k.pool.Engine.MaxNumBatchedTokens
	if k.batchedTokens <= 0 {
		k.batchedTokens = defaultMaxNumBatchedTokens(k.chip)
		k.assume("max_num_batched_tokens", strconv.Itoa(k.batchedTokens),
			"vLLM v0.31.0's OpenAI-API-server default for this part's memory "+
				"(EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887); "+
				"the offline LLM class defaults higher on every part below 160 GiB")
	}
	return nil
}

// activationWidth is h in the activation form: num_attention_heads x head_dim.
//
// That is NVIDIA's definition (AISimulate sdk/backends/base_backend.py: h =
// model._num_heads * model._head_size), where head_size is the config's head_dim, or
// hidden_size / heads when the config states none (sdk/utils.py). It equals hidden_size
// unless a config declares a head_dim that differs, and several do: qwen3-30b-a3b is
// 32 x 128 = 4,096 over a 2,048 hidden size.
//
// The graph carries it for a full or windowed attention layer, whose d_h is the config's
// head_dim. A LATENT layer's d_h is not a head_dim but the cache width (kv_lora_rank +
// qk_rope_head_dim), so for one the width is hidden_size. That is exact for every
// DeepSeek-V3-derived config, which states no head_dim, and it is the one place this can
// differ from NVIDIA's figure: a latent config that DOES state a head_dim (glm-5's is 64,
// deepseek-v4's 512) has an h the graph does not record.
//
// Taken as the widest over the stack, so a model whose attention kinds disagree is
// charged its widest scratch.
func activationWidth(p *price.Plan, g *model.Graph) float64 {
	var h int
	for _, l := range p.Layers {
		if l.AttnQHeads <= 0 {
			continue
		}
		w := l.AttnQHeads * l.AttnHeadDim
		if latentAttention(l.AttnKind) {
			w = g.Global.HiddenSize
		}
		h = max(h, w)
	}
	if h == 0 {
		h = g.Global.HiddenSize
	}
	return float64(h)
}

// lcm is the least common multiple of two positive integers.
func lcm(a, b int) int {
	x, y := a, b
	for y != 0 {
		x, y = y, x%y
	}
	return a / x * b
}

// defaultMaxNumSeqs is vLLM v0.31.0's default sequence cap for a part in the
// OpenAI-API-server context (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:
// 2858-2887): 1,024 on a part with at least 70 GiB that is not an A100, 256 otherwise.
func defaultMaxNumSeqs(chip hardware.Chip) int {
	if chip.MemoryGiB >= 70 && !strings.Contains(strings.ToLower(chip.Name), "a100") {
		return 1024
	}
	return 256
}

// defaultMaxNumBatchedTokens is vLLM v0.31.0's default token budget for a part, in the
// OpenAI-API-server usage context an llm-d deployment serves through
// (EngineArgs.get_batch_defaults, vllm/engine/arg_utils.py:2858-2887): 16,384 on a part
// with at least 160 GiB, 8,192 on one with at least 70 GiB that is not an A100, 2,048
// otherwise.
func defaultMaxNumBatchedTokens(chip hardware.Chip) int {
	switch {
	case chip.MemoryGiB >= 160:
		return 16384
	case chip.MemoryGiB >= 70 && !strings.Contains(strings.ToLower(chip.Name), "a100"):
		return 8192
	}
	return 2048
}

// liftCollectiveFloors resolves one floor and one rate per collective operation.
//
// The entry names carry operation, dtype and rank count because the coefficient
// schema's scope has no member for any of the three. A deployment whose exact
// combination was not measured is an error rather than a substitution: borrowing
// another width's floor would misprice by up to 2.7x in the measured set, and doing
// so silently is what this refuses.
func (k *Kernel) liftCollectiveFloors(c *resolve.Coefficients) error {
	// The dtype the collectives move, which selects which measured sweep prices them: the
	// 16-bit one, always. Collectives carry activations, which are bf16 whatever the weights
	// are served in (price.ActivationBytes has the citation, and the one exception this does
	// not model: a quantized MoE dispatch); an int8-served model's
	// all-reduce is a bf16 all-reduce, and the int8 sweep describes a payload no layer here
	// sends. An earlier form picked the int8 sweep for an int8-served model.
	const dtype = "fp16"
	chip := strings.ReplaceAll(k.chip.Name, "-", "_")
	k.collectiveFloors = map[collKey]time.Duration{}
	k.collectiveRates = map[collKey]float64{}
	k.collectiveTransitions = map[collKey]float64{}
	names := map[model.Op]string{
		model.OpAllReduce:     "all_reduce",
		model.OpAllGather:     "all_gather",
		model.OpReduceScatter: "reduce_scatter",
		model.OpAll2All:       "alltoall",
	}
	// One triple per (op, group), because a group is what picks the width and two groups
	// can run the same op. The tensor-parallel and expert-parallel axes cover everything
	// a model graph emits; the decode-context axis is resolved ONLY when dcp exceeds one,
	// so a deployment without DCP asks the registry for exactly the coefficients it
	// always did and cannot newly fail construction.
	keys := []collKey{
		{Op: model.OpAllReduce, Group: price.GroupTP},
		{Op: model.OpAllGather, Group: price.GroupTP},
		{Op: model.OpReduceScatter, Group: price.GroupTP},
		{Op: model.OpAll2All, Group: price.GroupExpert},
	}
	if k.layout.DCP > 1 {
		// Exactly the collectives the resolved combine launches over the decode-context
		// group (dcpDecodeCollectives has the table and the citations): an all-gather
		// always -- the query with PCP off, and the log-sum-exp under ag_rs -- then a
		// reduce-scatter for ag_rs, an all-reduce for ag_rs under PCP, or one all-to-all
		// carrying output and log-sum-exp together for a2a. Resolving only those means a part
		// missing an unused op is not refused.
		combine := model.OpReduceScatter
		switch {
		case k.decodeContext.CommBackend == resolve.DCPAllToAll:
			combine = model.OpAll2All
		case k.layout.PCP > 1:
			combine = model.OpAllReduce
		}
		keys = append(keys,
			collKey{Op: model.OpAllGather, Group: price.GroupDCP},
			collKey{Op: combine, Group: price.GroupDCP})
	}
	if k.layout.PCP > 1 {
		// One KV all-gather per layer that holds KV, so every rank keeps a full cache
		// replica of what the split prefill wrote.
		keys = append(keys, collKey{Op: model.OpAllGather, Group: price.GroupPCP})
	}
	for _, key := range keys {
		op := key.Op
		measured := names[op]
		// The rank count a collective spans, snapped to a width this chip was
		// actually measured at. The widths come from the registry rather than from a
		// constant here, so a part swept at 2 and 4 ranks only (a Grace-Blackwell
		// tray is four GPUs) is not asked for an 8-rank figure that does not exist.
		ranks, err := k.groupWidth(c, key, measured, dtype, chip)
		if err != nil {
			return err
		}
		stem := fmt.Sprintf("%s_%s_%drank_%s", measured, dtype, ranks, chip)
		floor, err := c.Value("collective_floor_" + stem)
		if err != nil {
			return fmt.Errorf("no measured floor for a %d-rank %s on %s: %w",
				ranks, measured, k.chip.Name, err)
		}
		rate, err := c.Value("collective_peak_rate_" + stem)
		if err != nil {
			return fmt.Errorf("no measured rate for a %d-rank %s on %s: %w",
				ranks, measured, k.chip.Name, err)
		}
		k.collectiveFloors[key] = time.Duration(floor * float64(time.Microsecond))
		k.collectiveRates[key] = rate * 1e6 // bytes per microsecond to bytes per second
		// The transition rate is what actually prices a forward pass's collectives; the
		// peak above is only a ceiling. Absent, the pricing falls back to the
		// two-parameter form, which is optimistic in the transition region — so its
		// absence is reported rather than silently tolerated.
		transition, err := c.Value("collective_transition_rate_" + stem)
		if err != nil {
			return fmt.Errorf(
				"no measured transition rate for a %d-rank %s on %s; the two-parameter "+
					"form understates a collective by up to 3.8x in the message range a "+
					"forward pass produces: %w", ranks, measured, k.chip.Name, err)
		}
		k.collectiveTransitions[key] = transition * 1e6
	}
	return nil
}

// snapDeclared snaps a width to one the registry declares for a family of per-width
// entries, named by format with a %d for the width: the widest declared width at or below
// it, or the narrowest declared where it is below them all.
//
// Separate from groupWidth because these families are vendor descriptor figures --
// NVIDIA's communicator sizes and activation multiples -- not measured collective sweeps,
// and a slightly mis-sized memory allowance is not a mispriced latency term, so snapping is
// right here where groupWidth refuses. NVIDIA's own model indexes its tables at
// min(tp, 8) (AISimulate base_backend.py). The registry declares both families at widths
// up to 8, so snapping a wider tp already lands on 8; the activation caller also clamps to
// 8 explicitly, and the communicator caller would take a wider entry if one were declared.
func snapDeclared(c *resolve.Coefficients, format string, width int) int {
	var declared []int
	for _, w := range []int{1, 2, 4, 8, 16} {
		if c.Has(fmt.Sprintf(format, w)) {
			declared = append(declared, w)
		}
	}
	if len(declared) == 0 {
		return width
	}
	best := declared[0]
	for _, w := range declared {
		if w <= width {
			best = w
		}
	}
	return best
}

// candidateRankWidths are the group widths any AISimulate comm sweep in this project has
// ever carried. It is the SEARCH SPACE groupWidth discriminates against — the widths for
// which "some part was swept here, so fit it for yours" is a fair demand — and nothing
// else. Which widths a given part actually carries is read from the registry by
// measuredWidths, so this list does not gate what can be used; a sweep at a width absent
// from it is still found and still priced.
//
// It is a constant because it encodes a claim about the DATA LANDSCAPE rather than about
// any registry state: 2, 4, 8 and 16 are the widths NVIDIA's comm sweeps cover across the
// SKUs this project reads. Extend it when a sweep at a new width appears upstream, which
// turns "past all available data, clamp" into "this part has a gap, fit it" for that
// width. Leaving it stale is the safe direction — a wider group clamps with the
// approximation visible in provenance rather than erroring.
var candidateRankWidths = []int{2, 4, 8, 16}

// measuredWidths returns the widths this chip carries a complete coefficient triple
// for, ascending. Completeness matters: liftCollectiveFloors needs floor, peak rate
// AND transition rate, and a width holding only some of the three cannot price a
// collective, so it is not a measured width for this purpose.
//
// Discovered from the coefficient names the registry actually carries, rather than probed
// against a list of widths this file knows about. The difference matters the moment the
// registry grows: a sweep at a width no constant here mentions would otherwise be
// invisible — present in the registry, never looked for, and the deployment that needs it
// priced by clamping to a narrower figure or refused outright. Reading the names means a
// width becomes usable by being committed upstream, with no change here.
func measuredWidths(c *resolve.Coefficients, measured, dtype, chip string) []int {
	// collective_floor_<op>_<dtype>_<N>rank_<chip>
	prefix := "collective_floor_" + measured + "_" + dtype + "_"
	suffix := "rank_" + chip

	var out []int
	for _, name := range c.Names() {
		digits, ok := strings.CutPrefix(name, prefix)
		if !ok {
			continue
		}
		digits, ok = strings.CutSuffix(digits, suffix)
		if !ok {
			continue
		}
		w, err := strconv.Atoi(digits)
		if err != nil || w < 1 {
			continue
		}
		stem := fmt.Sprintf("%s_%s_%drank_%s", measured, dtype, w, chip)
		if c.Has("collective_peak_rate_"+stem) &&
			c.Has("collective_transition_rate_"+stem) {
			out = append(out, w)
		}
	}
	sort.Ints(out)
	return out
}

// groupWidth returns the rank count a collective spans, snapped to a width this chip
// was measured at.
//
// Two cases, and the discriminator is the SEARCH SPACE rather than this part's own
// widths. A group wider than every width any sweep in this project covers
// (candidateRankWidths) is past all available data: it is priced at the widest
// measured figure, which understates its floor, and the clamp is recorded in
// provenance because refusing would decline to price a deployment the data merely
// does not reach. A group whose width IS in the search space but was not swept for
// this part is an error. That is the rack-scale case — a Grace-Blackwell tray is four
// GPUs, so GB200 and GB300 carry 2- and 4-rank sweeps while an 8-rank group is a
// routine tp=8 deployment — and substituting the 4-rank floor there understates an
// 8-rank all-reduce by 1.53x to 1.91x across the seven parts measured at both widths.
// Making that substitution silently is what liftCollectiveFloors' doc comment
// promises not to do.
func (k *Kernel) groupWidth(
	c *resolve.Coefficients, key collKey, measured, dtype, chip string,
) (int, error) {
	// The group decides the width, not the op: an all-gather spans tp ranks on the
	// tensor-parallel axis and dcp ranks on the decode-context-parallel one. An axis with
	// no width is refused here rather than resolved against a substitute, which is the
	// same standard this function already holds for an unmeasured width.
	width, ok := k.groupSize(key.Group)
	if !ok {
		return 0, fmt.Errorf(
			"collective %s names parallelism axis %d, whose width this kernel cannot "+
				"determine; pricing it at the tensor-parallel width would understate a "+
				"narrower group's floor by up to 1.59x", key.Op, key.Group)
	}
	widths := measuredWidths(c, measured, dtype, chip)
	if len(widths) == 0 {
		return 0, fmt.Errorf(
			"no measured %s widths at all for %s on %s: the registry carries no "+
				"complete floor/peak/transition triple for any rank count",
			measured, dtype, k.chip.Name)
	}
	// Narrower than anything measured: the narrowest is the only defensible figure,
	// and a 1-rank group does no collective the caller reaches this path for.
	if width <= widths[0] {
		return widths[0], nil
	}
	// Exact hit on a width this part was swept at.
	for _, w := range widths {
		if w == width {
			return width, nil
		}
	}
	// Past the whole search space: clamp to this part's widest measured figure.
	if width > candidateRankWidths[len(candidateRankWidths)-1] {
		return widths[len(widths)-1], nil
	}
	return 0, fmt.Errorf(
		"no %d-rank %s measurement for %s: the registry carries %v-rank figures for "+
			"this part, and borrowing a narrower width's floor understates an 8-rank "+
			"all_reduce by 1.53x to 1.91x across the parts measured at both widths. "+
			"Fit a %d-rank %s coefficient for %s, or run this deployment at a "+
			"measured width",
		width, measured, k.chip.Name, widths, width, measured, k.chip.Name)
}

// dtypeFit maps a weight dtype to its coefficient-name suffix and the chip's peak rate for
// it. A format the chip does not support natively has no peak here: pricing it at a
// nominal rate the hardware reaches only through a dequantize path would overstate it.
func dtypeFit(d model.DType, c hardware.Chip) (suffix string, peak float64) {
	switch d {
	case model.DTypeBF16, model.DTypeFP16:
		return "bf16", c.BF16Peak * 1e12
	case model.DTypeFP8:
		if c.FP8Peak > 0 {
			return "fp8", c.FP8Peak * 1e12
		}
		return "bf16", c.BF16Peak * 1e12
	case model.DTypeNVFP4, model.DTypeMXFP4:
		if c.NVFP4Peak > 0 {
			// The nvfp4 suffix, not fp8. This returned "fp8" while the registry carried
			// six fitted gemm_*_nvfp4 entries, so those were never read and a four-bit
			// deployment was priced with the fp8 asymptote -- 0.717 against the 0.504
			// fitted for nvfp4 on B200 and 0.441 on B300, over-pricing efficiency by
			// 1.42x and 1.73x on top of the NVFP4 peak those fractions are taken of.
			return "nvfp4", c.NVFP4Peak * 1e12
		}
		// Four-bit weights on a part without native support are dequantized to a wider
		// format before the matmul, so the achievable rate is that wider format's.
		if c.FP8Peak > 0 {
			return "fp8", c.FP8Peak * 1e12
		}
		return "bf16", c.BF16Peak * 1e12
	case model.DTypeINT8:
		return "fp8", c.FP8Peak * 1e12
	case model.DTypeINT4:
		// W4A16: the weights are four-bit but the activations are not quantized at all
		// (compressed-tensors leaves input_activations null), so the matmul runs at the
		// compute dtype's rate after dequantizing the weights. That is BF16, not a
		// four-bit rate -- the narrow storage buys memory traffic, not FLOPs. Stated
		// rather than left to the fallthrough, because the two agree only by coincidence
		// and a reader should see which one is intended.
		return "bf16", c.BF16Peak * 1e12
	}
	return "bf16", c.BF16Peak * 1e12
}

// validateDocuments runs blis-schemas' FIELD validation over the documents New was given.
//
// Why here rather than in Open: both constructors end in New, so putting it here means a
// caller that assembles Inputs directly cannot skip it. Why at all: nothing in this
// repository called any Validate() before, while Inputs' doc comment claimed the documents
// "arrive already validated" — so a caller was promised a check that did not happen.
// LoadModelGraph only deserializes. Strict decoding catches an unparseable file or a
// misspelled key, not a cyclic layer graph, a self-edge, a duplicate layer-kind id or an
// unrecognized emit condition; model.Graph.Validate holds 47 such checks and was called by
// nothing.
//
// It also closes a gap the v0.2.0 document split opened. Before the split, Scenario carried
// Pools, so one Validate covered the pool/cluster coupling. After it, those checks live in
// deployment.ValidateAgainstCluster — that a deployment's pools fill the cluster its
// scenario declares, that each local data-parallel width divides a node, that offload tiers
// draw from the declared storage inventory — and blisschemas.Validate is what supplies the
// cluster to them. Nothing ran it outside this repository's own fixture tests, so an
// external caller could build a kernel for a deployment that does not fit its hardware.
//
// THE FIELD LAYER ONLY, and that boundary is deliberate. blis-schemas separates field
// problems, which are version-independent and always the author's to fix, from
// version-scoped RULE problems, where a finding may mean the document is right and the
// engine version is wrong. Measured over the committed corpus: every one of the 687 bundles
// and all 32 catalog model graphs pass the field layer, while the rules layer rejects 213 of
// them — the 0.29.0 pack does not accept minimax_m3_mtp as a speculative method. Enforcing
// rules here would refuse a quarter of the deployments this project scores against measured
// data, for a reason that belongs in the catalog or the rules pack rather than in a
// constructor.
//
// Warnings are not failures: Problems.OK ignores them by design, so an out-of-tree
// quantization name reported as "not in-tree for 0.29.0" still builds.
func validateDocuments(in Inputs) error {
	rep := schemas.Validate(schemas.Bundle{
		Scenario: in.Scenario, Deployment: in.Deployment, Model: in.Model,
		Chip: in.Chip, Fabric: in.Fabric, Devices: in.Devices,
		Coefficients: in.Coefficients,
	})
	if !rep.Field.OK() {
		return fmt.Errorf("the documents do not validate:\n%w", rep.Field)
	}
	return nil
}

// servedDType resolves the format the linear layers are served in.
//
// A quantization the kernel does not recognize is an error rather than a fallback to the
// checkpoint's width: silently serving a request for, say, awq at bf16 would report weight
// bytes and a compute peak that are both wrong, with nothing saying so.
func servedDType(checkpoint model.DType, quantization string) (model.DType, error) {
	switch quantization {
	case "":
		return checkpoint, nil
	case "fp8", "ptpc_fp8", "fbgemm_fp8", "modelopt":
		return model.DTypeFP8, nil
	case "modelopt_fp4", "nvfp4":
		return model.DTypeNVFP4, nil
	case "mxfp4":
		return model.DTypeMXFP4, nil
	case "bitsandbytes", "awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors":
		// These are weight-only integer formats whose packed width depends on a
		// per-checkpoint bit count the flag does not carry. A checkpoint serving one of
		// them records the width in its own quantization_config, which the graph already
		// read, so the flag adds nothing and the graph governs.
		return checkpoint, nil
	}
	return "", fmt.Errorf(
		"quantization %q is not one this kernel prices; add it to servedDType with its "+
			"weight width rather than letting it fall through to %s",
		quantization, checkpoint)
}

// cacheDTypeBytes resolves the KV or state cache width.
//
// "auto" follows the model's COMPUTE dtype: vLLM sizes an auto cache at
// model_config.dtype (vllm/platforms/interface.py:861-862, vllm/utils/torch_utils.py:527-529
// at v0.31.0), which is bf16 or fp16 whatever the weights are stored in. That is not the
// weight storage width on a quantized checkpoint, and the difference is not cosmetic. A
// W4A16 checkpoint -- Kimi-K2.5 is compressed-tensors int4 at group_size 32 -- stores
// weights at four bits and computes in bf16, so following the weight width gave a
// half-byte KV element and, once paged, a per-block figure of zero. An fp8-served model is
// the same case: its linears return out_dtype=x.dtype, so its activations and an auto
// cache stay bf16, and an earlier form that sized it at one byte halved its cache.
//
// WHAT THIS CANNOT SEE. "auto" is resolved against the checkpoint before it reaches the
// cache: a quantization config declaring a KV algorithm (a ModelOpt checkpoint with
// kv_cache_quant_algo) turns it into that dtype (resolve_kv_cache_dtype_string,
// vllm/utils/torch_utils.py:503-518), and DeepSeek-V4 turns it into fp8_ds_mla when its
// backend uses that layout (vllm/models/deepseek_v4/attention.py:107-120). (DeepSeek-V3.2
// keeps "auto" at the model dtype, vllm/model_executor/models/config.py:37-39, whatever the
// cache.py:127 docstring says.) The graph carries neither, so such a deployment should
// state its cache dtype; "auto" here is the model dtype.
func cacheDTypeBytes(declared string, fallback model.DType) float64 {
	switch declared {
	case "", "auto":
		// A weight format narrower than 16 bits is a storage width, not a compute width:
		// the model computes in bf16 (or fp16) and the cache follows that. A 16- or 32-bit
		// format is its own compute width and passes through.
		if fallback.Bytes() < 2 {
			return model.DTypeBF16.Bytes()
		}
		return fallback.Bytes()
	case "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla":
		return 1
	case "bfloat16", "float16":
		return 2
	case "nvfp4", "nvfp4_4over6":
		// Four-bit data plus an fp8 scale per 16 elements: head_size/2 + head_size/16
		// bytes a head (nvfp4_kv_cache_full_dim, vllm/utils/torch_utils.py:543-545).
		return 0.5 + 1.0/16
	case "nvfp4_ds_mla":
		// Its 352-byte cell is applied where it is served (dsMLAStateBytes); this is only
		// the per-element width a latent model never prices with.
		return 0.5
	case "float32":
		return 4
	}
	// checkCacheDTypes has refused every other name, so this is unreachable from New.
	return fallback.Bytes()
}

// The KV cache dtypes vLLM v0.31.0 accepts (CacheDType, vllm/config/cache.py:39-58), split by
// whether this kernel prices them. The turboquant and per-token-head formats are real but
// carry per-token scales or packings whose page size this kernel does not compute
// (MLACommonBackend.customize_spec, mla_attention.py:1580-1593, for the per-token-head
// ones), so they are refused rather than priced at a width they do not have. Some of the
// priced ones are refused for a model as well (checkCacheDTypes).
var (
	pricedCacheDTypes = map[string]bool{
		"": true, "auto": true, "float16": true, "bfloat16": true, "fp8": true,
		"fp8_e4m3": true, "fp8_e5m2": true, "fp8_inc": true, "fp8_ds_mla": true,
		"nvfp4_ds_mla": true, "nvfp4": true, "nvfp4_4over6": true,
	}
	unpricedCacheDTypes = map[string]bool{
		"turboquant_k8v4": true, "turboquant_4bit_nc": true, "turboquant_k3v4_nc": true,
		"turboquant_3bit_nc": true, "int4_per_token_head": true,
		"int8_per_token_head": true, "fp8_per_token_head": true,
	}
	// The recurrent state dtypes (MambaDType, vllm/config/cache.py:61).
	mambaCacheDTypes = map[string]bool{
		"": true, "auto": true, "float32": true, "float16": true, "bfloat16": true,
	}
)

// checkCacheDTypes refuses a cache dtype the engine does not accept, one it accepts that
// this kernel does not price, and one no backend serves for this model's attention kind:
//
//   - fp8_inc is listed by no CUDA attention backend (it appears only in
//     vllm/config/cache.py:46 and vllm/utils/torch_utils.py:46);
//   - fp8_ds_mla and nvfp4_ds_mla are listed only by the sparse-MLA backends;
//   - plain nvfp4 on a latent model is refused outright (validate_nvfp4_kv_cache_with_mla,
//     vllm/config/vllm.py:3414-3430).
//
// An unchecked name used to fall through to the weight width, which on an nvfp4-served
// model charged an unknown cache at half a byte.
func checkCacheDTypes(cache, mamba string, kind model.AttentionKind) error {
	switch {
	case cache == "fp8_inc":
		return fmt.Errorf("cache_dtype fp8_inc is served by no CUDA attention backend in " +
			"vLLM v0.31.0")
	case strings.HasSuffix(cache, "_ds_mla") && kind != model.AttentionSparseMLA:
		return fmt.Errorf("cache_dtype %q is served only by the sparse-MLA backends, and "+
			"this model's attention is %s", cache, kind)
	case strings.HasPrefix(cache, "nvfp4") && !strings.HasSuffix(cache, "_ds_mla") &&
		latentAttention(kind):
		return fmt.Errorf("cache_dtype %q on a latent-attention model: vLLM v0.31.0 "+
			"refuses plain nvfp4 with MLA (vllm/config/vllm.py:3414-3430)", cache)
	case unpricedCacheDTypes[cache]:
		return fmt.Errorf("cache_dtype %q is a vLLM v0.31.0 format this kernel does not "+
			"price: its page carries scales or a packing the kernel does not size", cache)
	case !pricedCacheDTypes[cache]:
		return fmt.Errorf("cache_dtype %q is not a vLLM v0.31.0 cache dtype "+
			"(vllm/config/cache.py:39-58); the engine refuses it", cache)
	case !mambaCacheDTypes[mamba]:
		return fmt.Errorf("mamba_cache_dtype %q is not a vLLM v0.31.0 recurrent state "+
			"dtype (vllm/config/cache.py:61); the engine refuses it", mamba)
	}
	return nil
}

// kvGeometry returns the KV head count, head dimension and layer count that hold KV. A
// recurrent layer holds no KV, so it does not contribute.
//
// A layer is counted ONCE however many attention kernels it launches, and only the
// attention that reads the engine's cache sets the geometry. Both matter on a
// block-sparse layer: its indexer is a second Attention node with its own narrow cache
// (MiniMax-M3 scores with 4 heads over 128 where the layer reads 4 KV heads over 128),
// so counting per node doubled that model's KV-holding layers to 117 of 60 and left the
// geometry set by whichever node the loop saw last.
//
// The kind returned is the cache-holding attention's, taken from the same node as the
// geometry, which decides whether a token holds a key and a value or one latent vector. The
// catalog states no stack that mixes latent and non-latent caches.
func kvGeometry(g *model.Graph) (nkv, headDim, layers int, kind model.AttentionKind) {
	counts := map[string]int{}
	for _, id := range g.Stack.Expand() {
		counts[id]++
	}
	for _, lk := range g.LayerKinds {
		holds := false
		for _, n := range lk.Nodes {
			if n.Op != model.OpAttention || n.Role != "" {
				continue
			}
			if !holds {
				nkv, headDim, kind = n.NumKVHeads, n.HeadDim, n.AttentionKind
				holds = true
			}
		}
		if holds {
			layers += counts[lk.ID]
		}
	}
	return nkv, headDim, layers, kind
}

// computeFixedBytes sums occupancy independent of the request set.
func (k *Kernel) computeFixedBytes(g *model.Graph) kernel.MemoryBreakdown {
	var weights float64
	for _, l := range k.plan.Layers {
		c := float64(l.Count)
		weights += c * l.DenseWeightBytes / float64(max(k.layout.TP, 1))
		if l.ExpertWeightBytesPerExpert > 0 {
			// Exactly one axis divides an expert's weights: expert parallelism gives a
			// rank whole experts (expertTensorShards 1, expertsPerRank a fraction of the
			// total), and without it every rank holds every expert as a tensor slice
			// (expertTensorShards tp x dp, expertsPerRank the full count). Dividing by
			// expertTensorShards is what makes this a PER-RANK figure in the second case;
			// omitting it reported the whole model's expert weights on every rank, which
			// on GLM-5 at tp=8 was 675 GiB against a 141 GiB part.
			//
			// The step-time path applies the same division (kernel.go), and the two must
			// agree: they are the same bytes, read once per step and resident throughout.
			weights += c * l.ExpertWeightBytesPerExpert * k.expertsPerRank /
				k.expertTensorShards
			weights += c * l.SharedExpertWeightBytes / float64(max(k.layout.TP, 1))
		}
	}
	// Embeddings and the head are tensor-parallel sharded but not expert-parallel sharded.
	// Embeddings are not quantized by --quantization in vLLM: the flag applies to the
	// linear layers, and the embedding table stays at the checkpoint's width. Sizing it
	// at the served width would understate a large-vocabulary model by the ratio.
	weights += (k.plan.Head.DenseWeightBytes +
		float64(g.Global.VocabSize)*float64(g.Global.HiddenSize)*
			g.Global.WeightDType.Bytes()) / float64(max(k.layout.TP, 1))

	// Activation scratch at the batched-token bound: NVIDIA's live-buffer multiple of one
	// bf16 activation row per token, floored (see liftMemory).
	//
	// The 2 is the ACTIVATION width, which is the model's compute dtype and does not follow
	// the served weight format: vLLM's fp8 and fp4 linears quantize their input transiently
	// and return out_dtype=x.dtype (vllm/model_executor/kernels/linear/scaled_mm/
	// cutlass.py at v0.31.0), so an fp8 deployment's residual stream and inter-layer
	// buffers stay bf16. Sizing it at the served width would halve this term on an fp8
	// deployment and understate its occupancy.
	const activationBytes = price.ActivationBytes
	act := math.Max(
		k.activationBuffers*float64(k.batchedTokens)*k.activationWidth*activationBytes,
		k.activationFloor)

	// EPLB's redundant replicas, reported apart from the weights rather than on top of
	// them. expertsPerRank is the PHYSICAL count -- the model's experts plus the redundant
	// ones, divided over the group -- so the expert term above already holds them; the
	// replicas' share is moved out of Weights into EPLBRedundant, leaving Total what a rank
	// actually holds. An earlier form added them here as well, counting every replica twice.
	var eplb float64
	if k.pool.Engine.EPLB != nil && k.pool.Engine.EPLB.Enabled &&
		k.layout.ExpertWidth > 0 {
		for _, l := range k.plan.Layers {
			if l.ExpertWeightBytesPerExpert == 0 {
				continue
			}
			eplb += float64(l.Count) * l.ExpertWeightBytesPerExpert *
				float64(k.pool.Engine.EPLB.NumRedundantExperts) /
				float64(k.layout.ExpertWidth) / k.expertTensorShards
		}
		weights -= eplb
	}
	return kernel.MemoryBreakdown{
		Weights:        int64(weights),
		ActivationPeak: int64(act),
		CUDAGraph:      k.captureBytes,
		EPLBRedundant:  int64(eplb),
		// The collective-buffer reservation AND the engine workspace. blis-schemas v0.2.2's
		// MemoryBreakdown has no field for the workspace -- NVIDIA's misc.other_mem: CUDA
		// context, cuBLAS workspace and allocator slack -- so it is reported here rather
		// than left out of Total, which is what a capacity verdict reads. Provenance names
		// both coefficients, so the split is recoverable.
		CommBuffers: k.commBytes + k.workspaceBytes,
	}
}

// buildProvenance records every coefficient used, so a prediction can state its evidence.
//
// It records the kernel's own assumptions as well, after the registry's entries: see
// assume. A figure the kernel supplied itself would otherwise be invisible to Provenance
// and Evidence however its comment described it, which is how 1.3 GB of hardcoded memory
// occupancy once went unreported beside "101 measured of 133".
func (k *Kernel) buildProvenance(c *resolve.Coefficients) {
	for _, name := range c.Names() {
		r, ok := c.Entry(name)
		if !ok {
			continue
		}
		k.origins = append(k.origins, kernel.CoefficientOrigin{
			Name: name, Set: r.Set, Method: string(r.Entry.Method),
			Scope: scopeString(r.Entry.Scope),
		})
	}
	k.origins = append(k.origins, k.assumptions...)
	var overrides []kernel.Override
	for _, o := range k.layout.Overrides {
		overrides = append(overrides, kernel.Override{
			Field: o.Field, Requested: o.Requested, Resolved: o.Resolved, Reason: o.Reason,
		})
	}
	backend := "nccl"
	if k.layout.CustomAllReduce {
		backend = "custom"
	}
	cascade := false
	if k.pool.Engine.DisableCascadeAttn != nil {
		cascade = !*k.pool.Engine.DisableCascadeAttn
	}
	async := true
	if k.pool.Engine.AsyncScheduling != nil {
		async = *k.pool.Engine.AsyncScheduling
	}
	k.resolution = kernel.Resolution{
		// The layout this kernel priced, each width floored at one as the schema requires:
		// a consumer multiplies by these, so zero must never stand for "unset".
		TensorParallelWidth: max(k.layout.TP, 1),
		DataParallelWidth:   max(k.layout.DP, 1),
		ExpertParallelWidth: max(k.layout.ExpertWidth, 1),
		AllReduceBackend:    backend,
		AsyncScheduling:     async,
		CascadeAttention:    cascade,
		SequenceParallelMoE: k.layout.SequenceParallelMoE,
		Overrides:           overrides,
	}
}

// KernelAssumptionSet is the Set a Provenance entry carries when the kernel supplied the
// value itself rather than reading it from a registry set or the deployment. No registry
// set can take this name, since registry sets are named for their coefficient family, so a
// consumer can tell the two apart without parsing anything else.
const KernelAssumptionSet = "blis-latency-kernel"

// EngineBehaviourVersion is the vLLM release whose engine behaviour this kernel encodes.
const EngineBehaviourVersion = "0.31.0"

// optional reads a coefficient the kernel can price without, returning zero when the
// scenario's sets do not carry it. Where the deployment needs it, its absence is recorded as
// an assumption, naming what is priced without it, so a registry that drops an entry moves
// Evidence rather than silently cheapening the step.
func (k *Kernel) optional(c *resolve.Coefficients, name string, needed bool,
	without string) float64 {
	if c.Has(name) {
		return c.ValueOr(name, 0)
	}
	if needed {
		k.assume(name, "absent", "not in the scenario's coefficient sets: "+without)
	}
	return 0
}

// assume records a value the kernel filled in on the deployment's behalf -- an engine
// default for a setting the deployment left unstated, where the default is a judgement the
// kernel made rather than a fact it read. It reaches Provenance as an entry from
// KernelAssumptionSet with method "assumed", so Evidence counts it in the total and lists
// it among the assumptions instead of leaving the prediction's footing overstated.
//
// name is the setting; scope carries the value and the reason, since a reader of the
// provenance trail needs both and CoefficientOrigin has no other free-text field.
func (k *Kernel) assume(name, value, reason string) {
	k.assumptions = append(k.assumptions, kernel.CoefficientOrigin{
		Name: name, Set: KernelAssumptionSet, Method: string(vocab.MethodAssumed),
		Scope: value + ": " + reason,
	})
}

func scopeString(s coefficient.Scope) string {
	if len(s.Hardware) > 0 {
		return "hardware: " + s.Hardware[0]
	}
	if len(s.Model) > 0 {
		return "model: " + s.Model[0]
	}
	return "unscoped"
}

// Evidence reports how much of the resolved coefficient set rests on measurement, so a
// caller can state a prediction's footing.
func (k *Kernel) Evidence() (measured, total int, assumed []string) {
	for _, o := range k.origins {
		total++
		if vocab.Method(o.Method).Evidenced() {
			measured++
		} else if vocab.Method(o.Method) == vocab.MethodAssumed {
			assumed = append(assumed, o.Name)
		}
	}
	return measured, total, assumed
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
