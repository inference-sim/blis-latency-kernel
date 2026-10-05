package latencykernel

import (
	"fmt"
	"strings"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Inputs are the documents a kernel is built from. They arrive already parsed and
// validated: loading and validation belong to blis-schemas, and a kernel that re-validated
// would either duplicate those rules or diverge from them.
type Inputs struct {
	Scenario *scenario.Scenario
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
func New(in Inputs) (*Kernel, error) {
	if in.Scenario == nil || in.Model == nil || in.Chip == nil {
		return nil, fmt.Errorf("a kernel needs a scenario, a model graph and a chip")
	}
	if in.PoolIndex < 0 || in.PoolIndex >= len(in.Scenario.Pools) {
		return nil, fmt.Errorf("pool index %d is outside the scenario's %d pool(s)",
			in.PoolIndex, len(in.Scenario.Pools))
	}
	if in.Rules == nil {
		return nil, fmt.Errorf("a kernel needs engine rules for version %q",
			in.Scenario.EngineVersion)
	}
	pool := in.Scenario.Pools[in.PoolIndex]

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

	cacheBytes := cacheDTypeBytes(pool.Engine.CacheDType, served)
	stateBytes := cacheDTypeBytes(pool.Engine.MambaCacheDType, model.DTypeFP32)

	plan, err := price.BuildPlan(in.Model, emitter{layout}, served.Bytes(), stateBytes)
	if err != nil {
		return nil, fmt.Errorf("planning the model graph: %w", err)
	}
	k.plan = plan

	if err := k.lift(coeffs, in.Model, cacheBytes); err != nil {
		return nil, err
	}
	k.tiers = map[string]hardware.StorageDevice{}
	for _, d := range in.Devices {
		if d != nil {
			k.tiers[d.Name] = *d
		}
	}
	k.buildProvenance(coeffs)
	k.fixed = k.computeFixedBytes(in.Model, cacheBytes)
	return k, nil
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
	k.recurrent = map[model.RecurrentKind]floorRate{}
	for kind, suffix := range map[model.RecurrentKind]string{
		model.RecurrentMamba2: "mamba2",
		model.RecurrentKDA:    "kda",
		model.RecurrentGDN:    "gdn",
	} {
		floor := c.ValueOr("recurrent_decode_floor_"+suffix, 0)
		rate := c.ValueOr("recurrent_decode_rate_"+suffix, 0)
		if floor <= 0 || rate <= 0 {
			continue
		}
		k.recurrent[kind] = floorRate{
			floor: time.Duration(floor * float64(time.Microsecond)),
			rate:  rate * 1e6, // tokens per microsecond to tokens per second
		}
	}

	// The attention primitive's measured floor and rate. Absent for a part with no
	// generation-attention sweep, in which case the FLOPs fallback runs and says so.
	k.attentionFloor = time.Duration(
		c.ValueOr("attention_decode_floor", 0) * float64(time.Microsecond))
	k.attentionRate = c.ValueOr("attention_decode_rate", 0) * 1e6

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
	k.attentionByKind = map[model.AttentionKind]floorRate{}
	for kind, suffix := range map[model.AttentionKind]string{
		model.AttentionSWA: "swa",
	} {
		floor := c.ValueOr("attention_decode_floor_"+suffix, 0)
		rate := c.ValueOr("attention_decode_rate_"+suffix, 0)
		if floor <= 0 || rate <= 0 {
			continue
		}
		k.attentionByKind[kind] = floorRate{
			floor: time.Duration(floor * float64(time.Microsecond)),
			rate:  rate * 1e6, // bytes per microsecond to bytes per second
		}
	}
	k.attentionPrefillFloor = time.Duration(
		c.ValueOr("attention_prefill_floor", 0) * float64(time.Microsecond))
	k.attentionPrefillScale = c.ValueOr("attention_prefill_work_scale", 0)

	k.hostBytesPerSecond = c.ValueOr("host_link_bandwidth", 0) * 1e6
	k.moeImbalance = c.ValueOr("moe_routing_imbalance_median", 1.0)

	k.admissionPerToken = time.Duration(
		c.ValueOr("host_admission_per_token", 0) * float64(time.Microsecond))
	k.outputTokenCost = time.Duration(
		c.ValueOr("host_output_token", 0) * float64(time.Microsecond))
	k.completionCost = time.Duration(
		c.ValueOr("host_completion", 0) * float64(time.Microsecond))
	k.launchPerLayer = time.Duration(
		c.ValueOr("host_launch_eager_per_layer", 0) * float64(time.Microsecond))
	// Per-kernel dispatch, which dominates a single-request decode step. Required rather
	// than defaulted: a zero here silently removes the term that two published runs say is
	// most of a small step's cost.
	k.launchPerKernel = time.Duration(
		c.ValueOr("host_launch_per_kernel", 0) * float64(time.Microsecond))
	k.replayPerStep = time.Duration(
		c.ValueOr("host_replay_graph_per_step", 0) * float64(time.Microsecond))
	// How much of a step a captured graph covers, which sets the launch count rather
	// than only whether capture happens. vLLM's default is PIECEWISE, which splits at
	// every attention op, so the common case is one replay per layer and not one per
	// step. An empty setting means the engine's default.
	switch strings.ToUpper(k.pool.Engine.CUDAGraphMode) {
	case "", "PIECEWISE":
		k.graphMode = graphModePiecewise
	case "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE":
		// FULL_AND_PIECEWISE captures a full graph for uniform decode and falls back to
		// piecewise otherwise. Priced as full here, which is the decode case a step-time
		// model is usually asked about; a caller pricing its prefill steps should say
		// PIECEWISE.
		k.graphMode = graphModeFull
	case "NONE":
		k.graphMode = graphModeEager
	default:
		return fmt.Errorf(
			"cudagraph_mode %q is not one this kernel prices; add it with its launch "+
				"count rather than letting it fall through",
			k.pool.Engine.CUDAGraphMode)
	}
	k.graphCaptured = k.graphMode != graphModeEager

	// Communicator reservation and engine workspace, both stated per part by
	// NVIDIA's own descriptor. These replace order-of-magnitude constants an earlier
	// version of this kernel carried; the communicator figure is per rank count.
	k.commBytes = int64(c.ValueOr(fmt.Sprintf("nccl_communicator_bytes_%drank",
		k.groupWidth(model.OpAllReduce)), 0))
	k.workspaceBytes = int64(c.ValueOr("engine_workspace_bytes", 0))

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
	// is on. vLLM sets ep_size = tp and tp_size = 1 when it is, and ep_size = 1 with
	// tp_size = tp when it is not (fused_moe/config.py). So exactly one of the two axes
	// divides an expert, never both.
	k.localExpertShare = 1
	k.expertTensorShards = 1
	if k.layout.ExpertWidth <= 1 {
		// No expert parallelism: every rank holds every expert, tensor-sharded.
		k.expertTensorShards = k.tp
	}
	if experts > 0 {
		redundant := 0
		if k.pool.Engine.EPLB != nil && k.pool.Engine.EPLB.Enabled {
			redundant = k.pool.Engine.EPLB.NumRedundantExperts
		}
		// With expert parallelism off the width is 1, so every rank holds every expert
		// — which is right: the division is by tensor shard, applied at pricing time.
		base, _, _, imbalance := price.ExpertsPerRank(experts+redundant, k.layout.ExpertWidth)
		if base == 0 {
			return fmt.Errorf(
				"expert-parallel width %d exceeds the model's %d physical experts, so some "+
					"ranks would hold none", k.layout.ExpertWidth, experts+redundant)
		}
		k.expertsPerRank = float64(base)
		k.expertImbalance = imbalance
		// The share of the model's experts one rank holds. Computed from the local
		// count rather than as 1/width, so an uneven split prices each rank by what it
		// actually holds.
		k.localExpertShare = float64(base) / float64(experts+redundant)
		k.totalExperts = experts + redundant
	}

	// KV geometry. Both the cache dtype and the tensor-parallel width divide it, and the
	// head count floors at one because a head is never split across ranks.
	nkv, headDim, layers := kvGeometry(g)
	k.kvBytesPerToken = price.KVBytesPerToken(nkv, k.layout.TP, headDim, layers, cacheBytes)
	k.blockSize = k.pool.Engine.BlockSize
	if k.blockSize <= 0 {
		k.blockSize = 16
	}
	return nil
}

// liftCollectiveFloors resolves one floor and one rate per collective operation.
//
// The entry names carry operation, dtype and rank count because the coefficient
// schema's scope has no member for any of the three. A deployment whose exact
// combination was not measured is an error rather than a substitution: borrowing
// another width's floor would misprice by up to 2.7x in the measured set, and doing
// so silently is what this refuses.
func (k *Kernel) liftCollectiveFloors(c *resolve.Coefficients) error {
	// The dtype the collectives move. Activations cross at the served width, and the
	// sweeps measure fp16 and int8; an fp8 payload is priced at the fp16 rate, which
	// is the nearer of the two in both width and reduction cost.
	dtype := "fp16"
	if k.servedDType == model.DTypeINT8 {
		dtype = "int8"
	}
	chip := strings.ReplaceAll(k.chip.Name, "-", "_")
	k.collectiveFloors = map[model.Op]time.Duration{}
	k.collectiveRates = map[model.Op]float64{}
	k.collectiveTransitions = map[model.Op]float64{}
	for op, measured := range map[model.Op]string{
		model.OpAllReduce:     "all_reduce",
		model.OpAllGather:     "all_gather",
		model.OpReduceScatter: "reduce_scatter",
		model.OpAll2All:       "alltoall",
	} {
		// The rank count a collective spans: the group width, clamped to the widths
		// the sweep measured. A group wider than any measured width resolves to the
		// widest, which understates its floor — recorded in provenance rather than
		// silently corrected, because the alternative is refusing to price a
		// deployment the data merely does not reach.
		ranks := k.groupWidth(op)
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
		k.collectiveFloors[op] = time.Duration(floor * float64(time.Microsecond))
		k.collectiveRates[op] = rate * 1e6 // bytes per microsecond to bytes per second
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
		k.collectiveTransitions[op] = transition * 1e6
	}
	return nil
}

// measuredRankWidths are the group widths AISimulate's NCCL sweeps cover.
var measuredRankWidths = []int{2, 4, 8}

// groupWidth returns the rank count a collective spans, snapped to a measured width.
func (k *Kernel) groupWidth(op model.Op) int {
	width := k.layout.TP
	if op == model.OpAll2All {
		width = k.layout.ExpertWidth
	}
	if width < measuredRankWidths[0] {
		return measuredRankWidths[0]
	}
	// The largest measured width at or below the group's, so a 16-rank group is
	// priced at the 8-rank floor rather than extrapolated past the data.
	best := measuredRankWidths[0]
	for _, w := range measuredRankWidths {
		if w <= width {
			best = w
		}
	}
	return best
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
// "auto" follows the model's COMPUTE dtype, which is what vLLM documents: "If auto,
// will use model data type" (config/cache.py). That is not the weight storage width on
// a quantized checkpoint, and the difference is not cosmetic. A W4A16 checkpoint --
// Kimi-K2.5 is compressed-tensors int4 at group_size 32 -- stores weights at four bits
// and computes in bf16, so following the weight width gave a half-byte KV element and,
// once paged, a per-block figure of zero. A budget cannot be divided by that, and the
// kernel refused twelve sweeps rather than guess.
//
// vLLM does offer 4-bit KV (int4_per_token_head, turboquant_4bit_nc), but only when
// named. "auto" never selects one, so neither does this.
func cacheDTypeBytes(declared string, fallback model.DType) float64 {
	switch declared {
	case "", "auto":
		// A sub-byte weight format is a storage width, not a compute width. The cache
		// follows what the kernel computes in, which for every such checkpoint in this
		// catalog is bf16; a format that is already at least a byte wide is its own
		// compute width and passes through.
		if fallback.Bytes() < 1 {
			return model.DTypeBF16.Bytes()
		}
		return fallback.Bytes()
	case "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla":
		return 1
	case "bfloat16", "float16":
		return 2
	case "nvfp4", "nvfp4_4over6":
		return 0.5
	}
	// An unrecognized cache dtype falls back to the model's width rather than guessing a
	// narrower one, which would overstate capacity.
	return fallback.Bytes()
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
func kvGeometry(g *model.Graph) (nkv, headDim, layers int) {
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
				nkv, headDim = n.NumKVHeads, n.HeadDim
				holds = true
			}
		}
		if holds {
			layers += counts[lk.ID]
		}
	}
	return nkv, headDim, layers
}

// computeFixedBytes sums occupancy independent of the request set.
func (k *Kernel) computeFixedBytes(g *model.Graph, cacheBytes float64) kernel.MemoryBreakdown {
	var weights float64
	for _, l := range k.plan.Layers {
		c := float64(l.Count)
		weights += c * l.DenseWeightBytes / float64(max(k.layout.TP, 1))
		if l.ExpertWeightBytesPerExpert > 0 {
			// Exactly one axis divides an expert's weights: expert parallelism gives a
			// rank whole experts (expertTensorShards 1, expertsPerRank a fraction of the
			// total), and without it every rank holds every expert as a tensor slice
			// (expertTensorShards tp, expertsPerRank the full count). Dividing by
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

	batched := k.pool.Engine.MaxNumBatchedTokens
	if batched <= 0 {
		batched = 8192
	}
	// Activation scratch at the batched-token bound, across a few live buffers.
	act := float64(batched) * float64(g.Global.HiddenSize) * 2 * 4

	var eplb float64
	if k.pool.Engine.EPLB != nil && k.pool.Engine.EPLB.Enabled &&
		k.layout.ExpertWidth > 0 {
		for _, l := range k.plan.Layers {
			if l.ExpertWeightBytesPerExpert == 0 {
				continue
			}
			eplb += float64(l.Count) * l.ExpertWeightBytesPerExpert *
				float64(k.pool.Engine.EPLB.NumRedundantExperts) /
				float64(k.layout.ExpertWidth)
		}
	}
	return kernel.MemoryBreakdown{
		Weights:        int64(weights),
		ActivationPeak: int64(act),
		CUDAGraph:      k.graphCaptureBytes(),
		EPLBRedundant:  int64(eplb),
		CommBuffers:    k.commBufferBytes(),
	}
}

// graphCaptureBytes returns the memory a CUDA-graph capture holds. Zero when capture is
// off, which is what makes the graph mode a memory term as well as a host-time one.
func (k *Kernel) graphCaptureBytes() int64 {
	if !k.graphCaptured {
		return 0
	}
	// A capture holds one buffer set per captured shape. No public measurement of the
	// count exists, so this is an order-of-magnitude figure and is reported as such in
	// provenance rather than presented as sourced.
	return 512 << 20
}

// commBufferBytes returns the collective buffers a rank reserves.
func (k *Kernel) commBufferBytes() int64 {
	if k.layout.TP <= 1 && k.layout.ExpertWidth <= 1 {
		return 0
	}
	// Also an order-of-magnitude figure; see graphCaptureBytes.
	return 392 << 20
}

// buildProvenance records every coefficient used, so a prediction can state its evidence.
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
		ExpertParallelWidth: k.layout.ExpertWidth,
		AllReduceBackend:    backend,
		AsyncScheduling:     async,
		CascadeAttention:    cascade,
		SequenceParallelMoE: k.layout.SequenceParallelMoE,
		Overrides:           overrides,
	}
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
