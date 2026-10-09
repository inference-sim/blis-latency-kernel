// Package latencykernel implements the blis-schemas Kernel interface: it prices GPU work
// for one resolved deployment.
//
// Construction resolves everything — catalog facts, registry coefficients, the engine
// settings a layout overrides, and the model graph flattened to a per-kind plan. After
// that every method is a pure function of its arguments, so a discrete-event simulator can
// call StepTime at any simulated instant, from any goroutine, and get the same answer.
//
// Two constructors, for the two shapes a caller comes in. Open takes a scenario filename
// and the roots its names resolve against, which is what a simulator or a scoring command
// has. New takes the documents already parsed, which is what a configuration search
// producing deployment variants that were never written to disk has. Open ends in New, so
// both validate identically and neither can price a deployment the other would refuse.
//
// Speed is a requirement rather than a nicety: a simulator calls StepTime once per
// simulated step, millions of times over a run. So the hot path does arithmetic over a
// short pre-aggregated slice and allocates nothing. The plan holds one entry per distinct
// layer kind rather than one per layer, which for the deepest model in the catalog is three
// entries instead of 108.
package latencykernel

import (
	"fmt"
	"math"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Kernel prices one deployment. Obtain one from New; the zero value is not usable.
type Kernel struct {
	// Resolved configuration, frozen at construction.
	layout resolve.Layout
	fabric resolve.Fabric
	plan   *price.Plan
	pool   deployment.Pool
	chip   hardware.Chip

	// servedDType is the weight format the linear layers actually run in, after the
	// scenario's --quantization is applied over the checkpoint's own format.
	servedDType model.DType

	// Constants lifted out of the coefficient set once, so the hot path reads fields
	// rather than hashing strings.
	hbmBytesPerSecond     float64
	computeFLOPsPerSecond float64
	// nHalf and kHalf extend the efficiency ramp to the GEMM's own shape. Zero means the
	// registry carries no shape-aware entry for this part and dtype, and the kernel keeps
	// the token-count ramp -- so a partially migrated registry prices as it did before
	// rather than pricing a GEMM at zero.
	nHalf, kHalf         float64
	gemmShapeAware       bool
	epsMax, mHalf        float64
	nvlinkBytesPerSecond float64
	nicBytesPerSecond    float64
	hostBytesPerSecond   float64
	moeImbalance         float64

	// Per-operation collective floors and peak rates, resolved once. Keyed by op
	// because an all-reduce floor and an all-to-all floor differ by more than 2x on
	// the same part, and by rank count inside the lookup because a ring-shaped
	// collective's floor grows with group width where a shuffle's barely does.
	//
	// Keyed by GROUP as well as op, because the op alone stopped identifying a triple
	// once decode-context parallelism arrived: a tensor-parallel all-gather at tp=8 and
	// a DCP all-gather at dcp=2 are the same primitive at two widths, and the registry
	// carries a separate floor/peak/transition for each. One entry per op would have
	// priced one of the two at the other's width. For the ops the context-parallel axes
	// add, the 8-rank floor is 1.35x to 1.60x the 4-rank figure across the nine parts
	// this registry carries at both; see price.GroupAxis for the per-op spread.
	collectiveFloors      map[collKey]time.Duration
	collectiveRates       map[collKey]float64
	collectiveTransitions map[collKey]float64

	// Recurrent families' measured floor and rate, keyed by kind. A kind absent from the
	// map has no measurement for this part and contributes nothing.
	recurrent map[model.RecurrentKind]floorRate

	// The attention primitive's own floor and rate, measured per part. A zero rate means
	// this part has no measured entries and the FLOPs fallback runs.
	attentionFloor time.Duration
	attentionRate  float64
	// attentionByKind holds per-kind decode terms where the registry carries them. A kind
	// absent from the map uses attentionFloor and attentionRate, so the unsuffixed pair stays
	// the default and adding a kind never changes a deployment that has no fit for it.
	attentionByKind map[model.AttentionKind]floorRate
	// Prefill attention's own floor and the fraction of the dense ramp it reaches. A zero
	// scale means this part has no measured prefill entries and the FLOPs fallback runs.
	attentionPrefillFloor time.Duration
	attentionPrefillScale float64

	// Memory occupancy outside the KV budget, every magnitude from the registry (see
	// liftMemory): the communicator reservation, the engine workspace, the graph capture,
	// and the activation form's multiple, floor, width and token bound.
	commBytes         int64
	workspaceBytes    int64
	captureBytes      int64
	activationBuffers float64
	activationFloor   float64
	activationWidth   float64
	batchedTokens     int

	// Per-rank expert counts, derived once.
	expertsPerRank  float64
	expertImbalance float64

	// totalExperts is the model's physical expert count, which the expected-coverage
	// term needs alongside the local count.
	totalExperts int
	// expertTensorShards is how many ranks one expert's weights are split across: the
	// tensor-parallel width when expert parallelism is off, and 1 when it is on, because
	// an expert-parallel rank owns whole experts. With expert parallelism off and dp or
	// pcp above one, vLLM v0.31.0 shards an expert over dp x pcp x tp ranks, not tp (see
	// the KNOWN DIVERGENCE in lift).
	expertTensorShards float64

	// tp and localExpertShare are the per-rank divisors, held as floats so the hot path
	// multiplies rather than converting. localExpertShare is the fraction of the model's
	// experts one rank holds, which is what divides routed FLOPs.
	tp               float64
	localExpertShare float64

	// KV geometry, derived once from the graph and the cache dtype.
	kvBytesPerToken float64
	blockSize       int
	// dcpShardsAllKVLayers is whether every KV-holding layer is a kind DCP shards, which
	// is what lets a whole-model capacity figure carry the shard at all.
	dcpShardsAllKVLayers bool
	// kvLayers is how many layers hold KV, which is the count kvBytesPerToken is summed
	// over and so the divisor for one layer's share of the cache.
	kvLayers int

	// Host overheads.
	admissionPerToken time.Duration
	// admissionPerRequest is the length-INDEPENDENT host cost before a request can be
	// scheduled. Separate from admissionPerToken because the two have different
	// dimensions: InferenceX's c=1 anchors scale as L^0.575 between 1k and 8k inputs,
	// so the deficit is not proportional to prompt length and charging it per token
	// adds steepness to a prefill term that is already linear in L.
	admissionPerRequest time.Duration
	outputTokenCost     time.Duration
	completionCost      time.Duration
	launchPerLayer      time.Duration
	launchPerKernel     time.Duration
	replayPerStep       time.Duration
	// graphDecode is the graph mode a uniform decode batch runs under, graphOther the
	// mode every other batch does; they differ under FULL_AND_PIECEWISE and
	// FULL_DECODE_ONLY. graphModeName is the resolved mode in vLLM's lowercase spelling,
	// which keys the capture's memory. uniformDecodeWidth is the tokens per request that
	// make a decode batch uniform.
	graphDecode        graphMode
	graphOther         graphMode
	graphModeName      string
	uniformDecodeWidth int

	// recurrentCacheMode is how a hybrid model's recurrent state is cached, as resolved.
	recurrentCacheMode price.RecurrentCacheMode

	// decodeContext is how the decode-context-parallel combine runs: its backend and the
	// stripe each rank holds. Meaningful only when the layout's DCP exceeds one.
	decodeContext resolve.DecodeContext

	// Fixed occupancy, computed once.
	fixed kernel.MemoryBreakdown

	// Provenance, assembled once. assumptions are the values the kernel supplied itself,
	// appended to origins after the registry's entries.
	origins     []kernel.CoefficientOrigin
	assumptions []kernel.CoefficientOrigin
	resolution  kernel.Resolution

	// tiers maps a tier name to its device facts, for TierTime.
	tiers map[string]hardware.StorageDevice
}

// collKey identifies one measured collective triple: the primitive AND the group it runs
// across.
//
// Op alone is not enough. Two different groups can run the same primitive -- a
// tensor-parallel all-gather spans tp ranks while a decode-context-parallel one spans dcp
// -- and each width has its own measured floor, peak rate and transition rate in the
// registry. Collapsing them would silently charge one group the other's width.
type collKey struct {
	Op    model.Op
	Group price.GroupAxis
}

// floorRate is a primitive's two-parameter measured form: a fixed setup cost plus a rate.
type floorRate struct {
	floor time.Duration
	// rate is tokens per second, converted from the registry's tokens-per-microsecond.
	rate float64
}

// Compile-time proof that this type satisfies the published interface. If the interface
// gains a method, this line fails to build rather than the mismatch surfacing at a call
// site.
var _ kernel.Kernel = (*Kernel)(nil)

// FixedBytes returns occupancy independent of the request set, per rank.
func (k *Kernel) FixedBytes() kernel.MemoryBreakdown { return k.fixed }

// SequenceFixedBytes returns per-sequence bytes that do not vary with prompt length:
// recurrent state, where the model has it and the cache mode keeps it bounded.
func (k *Kernel) SequenceFixedBytes() int64 {
	var total float64
	mode := k.recurrentCacheMode
	if mode.ProportionalToContext() {
		// Under this mode the state scales with the context bound, so it belongs in the
		// variable term and this one reports none of it.
		return 0
	}
	for _, l := range k.plan.Layers {
		if l.RecurrentStateBytes == 0 {
			continue
		}
		spec := 0
		if k.pool.Engine.Speculative != nil {
			spec = k.pool.Engine.Speculative.NumSpecTokens
		}
		per := price.RecurrentStateBytes(l.RecurrentStateBytes, mode,
			k.pool.Engine.MaxModelLen, k.blockSize, spec, 0)
		// The state is replicated across context-parallel ranks and sharded by tensor
		// parallelism through its head count, which the graph already reflects.
		total += per * float64(l.Count)
	}
	return int64(total)
}

// SequenceVariableBytes returns per-sequence KV bytes for a token count, quantized to the
// engine's page size.
func (k *Kernel) SequenceVariableBytes(tokens int) int64 {
	if tokens <= 0 {
		return 0
	}
	// Decode-context parallelism shards the TOKEN COUNT, and the paging happens after the
	// division rather than before it. That ordering is the engine's:
	// FullAttentionSpec.max_memory_usage_bytes divides max_model_len by dcp_world_size and
	// THEN rounds to blocks (vllm/v1/kv_cache_interface.py:578-583 at v0.31.0), and
	// max_num_blocks_per_req is cdiv(max_len, block_size * kv_shard_count) (:545-550).
	// page_size_bytes never sees dcp.
	//
	// Dividing kvBytesPerToken instead would be the smaller edit and it would be wrong in
	// two ways: it would make a page fractional, understating occupancy by up to dcp below
	// block_size*dcp tokens (8x at 16 tokens and dcp 8, 2x at 64, exact from 128 up), and
	// the same field prices PDTransferTime, which DCP does not shard at all.
	//
	// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. kvBytesPerToken is ONE whole-model
	// scalar, summed by kvGeometry over layers of every attention kind, and this function
	// has no per-kind context to spend. So the shard applies only when every KV-holding
	// layer is a kind DCP shards. A hybrid stack -- gpt-oss-120b alternates a 128-token
	// swa layer with a full gqa one -- is left UNSHARDED, which overstates what it needs.
	// That is the conservative direction, and under the engine's default hybrid KV-cache
	// manager it costs nothing real: the engine refuses the deployment, asserting
	// dcp == 1 with "DCP not support sliding window" (kv_cache_interface.py:868-872). With
	// the hybrid manager disabled -- explicitly, or by a KV connector that does not support
	// it (vllm/config/vllm.py) -- the windowed layers are promoted to full attention and
	// sharded, which this overstates. Expressing the hybrid case exactly would mean
	// splitting kvBytesPerToken per kind -- a larger refactor, for configurations that
	// either do not start or run only with the hybrid manager off.
	//
	// AND THIS DISAGREES WITH THE DECODE READ ON THAT SAME STACK, deliberately. The read
	// is applied per layer kind, so on a hybrid it shards the full-attention layers and
	// leaves the windowed ones whole; capacity, having no per-kind context, shards
	// nothing. A consumer reading both therefore gets two models of one cache for a
	// configuration the engine will not run. Each term is individually defensible -- the
	// read is exact where it can be, capacity errs high where it cannot -- and making
	// them agree would mean either coarsening the read to a whole-model verdict, which
	// loses accuracy on every deployment that DOES run, or the per-kind refactor above.
	// Neither is worth doing for an inadmissible layout, so the disagreement is recorded
	// rather than resolved.
	shardedTokens := tokens
	if k.layout.DCP > 1 && k.dcpShardsAllKVLayers {
		shardedTokens = (tokens + k.layout.DCP - 1) / k.layout.DCP
	}
	total := price.PagedBytes(shardedTokens, k.blockSize, k.kvBytesPerToken)

	// Under the all-positions recurrent cache mode the state is proportional to the
	// context bound, so it is a variable cost rather than a fixed one.
	//
	// Sized from the UNSHARDED token count, which is deliberate: "Mamba state is
	// replicated across DCP/PCP ranks, never sharded"
	// (vllm/v1/kv_cache_interface.py:1097-1098 at v0.31.0). A context-parallel rank holds
	// every position's recurrent state even while holding only its shard of the KV cache.
	mode := k.recurrentCacheMode
	if mode.ProportionalToContext() {
		spec := 0
		if k.pool.Engine.Speculative != nil {
			spec = k.pool.Engine.Speculative.NumSpecTokens
		}
		for _, l := range k.plan.Layers {
			if l.RecurrentStateBytes == 0 {
				continue
			}
			total += int64(price.RecurrentStateBytes(l.RecurrentStateBytes, mode,
				tokens, k.blockSize, spec, 0) * float64(l.Count))
		}
	}
	return total
}

// StepTime prices one forward pass. This is the hot path.
//
// The returned StepEstimate carries a freshly allocated PerResource map. A simulator that
// calls this per simulated step, millions of times, can use StepTimeInto instead and reuse
// one map across calls, which makes the path allocation-free.
func (k *Kernel) StepTime(b kernel.Batch) kernel.StepEstimate {
	return k.stepTime(b, nil)
}

// StepTimeInto prices one forward pass, writing the per-resource breakdown into per
// instead of allocating a map.
//
// per is cleared first, so a caller reuses one map for every step. Passing nil is allowed
// and yields an estimate with no breakdown, for a caller that wants only the duration.
//
// The estimate's own fields are values, so nothing else escapes: this form allocates
// nothing at all, which is what lets a long simulation avoid the garbage collector on its
// innermost loop.
func (k *Kernel) StepTimeInto(b kernel.Batch,
	per map[kernel.Resource]time.Duration) kernel.StepEstimate {
	for r := range per {
		delete(per, r)
	}
	return k.stepTime(b, per)
}

func (k *Kernel) stepTime(b kernel.Batch,
	into map[kernel.Resource]time.Duration) kernel.StepEstimate {
	tokens := b.Tokens()
	if tokens == 0 {
		// An empty step launches no work but still costs the host its per-step overhead.
		host := k.hostPerStep(b)
		per := into
		if per == nil {
			per = make(map[kernel.Resource]time.Duration, 1)
		}
		per[kernel.ResourceHost] = host
		return kernel.StepEstimate{
			// Both edges are the host cost on an empty step: there is no device work for
			// them to disagree about.
			Overlap: host, NoOverlap: host,
			Bottleneck:  kernel.ResourceHost,
			PerResource: per,
		}
	}

	// Attention work depends on each request's own context length, so it is the one term
	// that walks the batch. Everything else multiplies a pre-aggregated scalar.
	//
	// Prefill and decode are separated because they are different regimes with their own
	// measured constants: prefill is compute-bound and quadratic in prompt length, decode is
	// bandwidth-bound in context. A request is in the prefill regime when it schedules more
	// than the batch's decode threshold -- the cost model's batch-region classifier. vLLM's
	// MLA and FlashInfer backends make the same split (split_decodes_and_prefills,
	// vllm/v1/attention/backends/utils.py:799-870 at v0.31.0), each at its own threshold;
	// FlashAttention does not split, and runs one varlen kernel over the mixed batch
	// (flash_attn.py:1450-1478).
	//
	// causalFLOPs halves the naive count because attention attends only to earlier
	// positions. decodeKVTokens is the context those requests read.
	var causalFLOPs, decodeKVTokens float64
	// Each prefill chunk's shape is kept so a sliding-window layer can be charged its
	// own bounded pair count. causalFLOPs is the unwindowed total, which is what a
	// full-attention layer pays.
	var prefillChunks []chunk
	// Each decode request's own context, kept for the same reason prefillChunks is: a
	// sparse-MLA layer's read is bounded PER REQUEST (min(ctx, topk) plus a compressed
	// remainder), and a bound cannot be applied to a batch-wide sum without changing the
	// answer. min(sum) != sum(min) whenever any request sits on either side of the bound,
	// which in a mixed batch is the normal case.
	var decodeContexts []int
	var prefillRequests, decodeRequests int
	// decodeTokens is the query rows the decode regime runs, which is what crosses in a
	// decode-context combine: one row per scheduled token, so a speculative decode
	// verifying four tokens moves four rows, not one.
	var decodeTokens int
	// prefillTokens is the scheduled prefill tokens THIS RANK computes, which the KV
	// gather's payload is sized from. withheldByPCP is how many of the batch's scheduled
	// tokens the prefill split takes off this rank, which is what the step's token count
	// loses -- tracked as a DELTA rather than by re-summing the batch, so a request the
	// loop below skips keeps contributing exactly what it contributed before.
	var prefillTokens, withheldByPCP int
	for i := range b.Reqs {
		r := &b.Reqs[i]
		ctx := r.Computed + r.Scheduled
		if ctx <= 0 {
			continue
		}
		if r.Scheduled > b.DecodeThreshold {
			prefillRequests++
			// Attention pairs for a chunk of `Scheduled` tokens resuming on a prefix of
			// `Computed` already-computed ones. Every chunk token attends to the WHOLE
			// prefix -- causal masking does not reduce that, since the prefix is entirely
			// earlier -- and to its causal share within the chunk:
			//
			//	pairs = Scheduled*Computed + Scheduled^2 / 2
			//
			// This read `Scheduled * (Computed+Scheduled) * 0.5`, which halves the
			// prefix term. At Computed == 0 the two agree, so an unchunked prefill
			// cannot reveal it, and every prefill check in this repository was
			// unchunked. FPM's mixed rows are what exposed it: at a 1025-token chunk on
			// a 203,760-token prefix the old count is 1.995x low, and the measured
			// whole-forward latency grows 5.39x across that context sweep where the
			// prediction grew 2.85x.
			sched := float64(r.Scheduled)
			// Prefill-context parallelism splits this chunk across its ranks, and the
			// split is balanced in BOTH quantities the chunk contributes.
			//
			// The causal pairs divide by exactly pcp. That is the zigzag pairing's doing,
			// not an approximation: pairing chunk r with chunk 2*pcp-1-r gives every rank
			// the same pair count, verified equal to 1.000000 of the ideal share with zero
			// spread across the ranks. See pcpLocalTokens for the enumeration.
			//
			// The TOKEN count divides by the busiest rank's share, which carries the
			// ragged remainder when 2*pcp does not divide the chunk. Both are per request,
			// since each prefill is partitioned on its own.
			pairs := 2 * 2 * (sched*float64(r.Computed) + sched*sched*0.5)
			localSched := r.Scheduled
			if local := pcpLocalTokens(r.Scheduled, k.layout.PCP, k.layout.DCP); local >= 0 {
				localSched = local
				// The causal work of the BUSIEST rank, computed from the engine's own
				// partition rather than scaled by a ratio.
				//
				// Where 2*pcp divides the query the zigzag makes every rank's pair count
				// exactly equal, so this is the whole count over pcp. Where it does not,
				// neither 1/pcp nor the token share is right: at 100 tokens over 8 ranks
				// the busiest rank carries 0.1568 of the pairs where 1/pcp is 0.125 and
				// its token share is 0.14. A ragged partition loads one rank with two
				// short chunks from the EXPENSIVE end, and only counting them says by how
				// much.
				//
				// A replicated request -- one the engine hands whole to every rank -- is
				// left at its full count by pcpCausalPairs, since every rank computes all
				// of it.
				pairs = pcpCausalPairs(r.Scheduled, r.Computed,
					k.layout.PCP, k.layout.DCP)
			}
			causalFLOPs += pairs
			prefillTokens += localSched
			withheldByPCP += r.Scheduled - localSched
			// The chunks this rank owns, at their TRUE GLOBAL OFFSETS.
			//
			// A sliding-window layer's cost is positional, not a function of the token
			// count: a query at global offset 100 under a 128-token window reads 101 keys,
			// one at offset 100,000 reads 128. So the windowed term needs where this
			// rank's tokens SIT, not merely how many it has -- and under the zigzag a
			// rank owns two runs, one near the start and one near the end, which cannot
			// be described as a single run at the request's own prefix.
			//
			// Collapsing them to one run at `prefix` charges the cheap early end twice
			// and understates a windowed prefill by up to 7.9x (a 4,096-token chunk on no
			// prefix at pcp 8, window 2,176). The error vanishes once the prefix
			// saturates the window, since every query then reads exactly `window` keys
			// wherever it sits -- which is why a long-context fixture cannot reveal it.
			prefillChunks = k.appendPrefillRuns(prefillChunks,
				r.Scheduled, r.Computed)
			continue
		}
		decodeRequests++
		decodeTokens += r.Scheduled
		decodeKVTokens += float64(ctx)
		decodeContexts = append(decodeContexts, ctx)
		// A decode row is REPLICATED across prefill-context-parallel ranks, not split:
		// _iter_rank_chunks gives every rank chunk_indices = (0,) for a non-prefilling
		// request, so each computes the whole one-token query. PCP divides prefill only,
		// which is why nothing is withheld here.
	}
	// The token count this rank actually runs a forward over: the batch's own count less
	// whatever the prefill split withheld. Subtracting a delta rather than re-summing the
	// loop is deliberate -- the loop skips a request whose context is non-positive, so a
	// re-sum would silently drop tokens that request still contributes to every term
	// derived from the step's token count. That would reprice a batch holding one, which
	// a re-summed draft of this did: a two-token request resuming on a negative prefix
	// priced 11.64 ms against 12.87 ms before.
	//
	// THIS IS THE RANK-LOCAL COUNT, AND THE EFFICIENCY RAMP THEREFORE SEES IT. That is a
	// decision rather than a side effect, and the engine settles it: a prefill-context
	// rank fills its input_ids and positions buffers with only num_local_tokens_padded
	// rows (vllm/v1/worker/gpu/pcp_manager.py:519-532 at v0.31.0) and runs the forward on
	// that, so every GEMM in the layer genuinely sees a shorter m. A narrower GEMM reaches
	// a lower fraction of peak, so a PCP rank is LESS efficient per token than an
	// unsplit one -- measured here at 3.4% above the ideal half at pcp 2 and 10.1% above
	// the ideal quarter at pcp 4, on a 4,096-token chunk.
	//
	// Evaluating the ramp at the batch-wide count instead would credit a split rank with
	// an efficiency its kernel does not reach, and would make PCP look exactly linear when
	// it is not. The argument the ramp takes is unchanged for every deployment that does
	// not split -- this is the same scalar it always was -- so the finding recorded at the
	// ramp's own call site, that it is evaluated at the step's token count rather than a
	// per-term one, still holds: what changed is how many tokens the step HAS.
	tokens -= withheldByPCP

	smBudget := b.SMBudget
	if smBudget <= 0 || k.chip.SMCount == 0 {
		smBudget = k.chip.SMCount
	}
	smDerate := 1.0
	if k.chip.SMCount > 0 && smBudget > 0 && smBudget < k.chip.SMCount {
		// Fewer SMs available means the compute term takes proportionally longer.
		smDerate = float64(k.chip.SMCount) / float64(smBudget)
	}

	// Composed PER LAYER, not once over the whole step.
	//
	// The cost model's §5.1 rule is to partition a step into stages, where a stage is a
	// maximal set of operations that run concurrently; take the max over resources within
	// a stage, and SUM across stages. A layer is a stage: layer N+1 needs layer N's
	// output, so the two cannot overlap however idle a resource is.
	//
	// Applying one max over the whole step instead assumes all 72 layers overlap with
	// each other, which understates a step badly wherever no single resource dominates.
	// At single-request decode nothing dominates — every kernel is small — so that
	// understatement is largest exactly where a latency-sensitive deployment runs.
	//
	// Per-resource totals are still accumulated, because a caller wants to see where the
	// work went even though the step time is not their max.
	var perResource [numResources]float64
	var overlapSeconds float64
	tokensF := float64(tokens)
	for i := range k.plan.Layers {
		l := &k.plan.Layers[i]
		count := float64(l.Count)

		// One layer's work. Everything here is PER RANK, and which axis divides a term is
		// a property of the term rather than one global divisor: projections shard by
		// tensor parallelism, routed experts by expert-parallel width, and a term divided
		// by the wrong one is wrong by the ratio between them.
		// routedFLOPs is kept apart from flops because the two are priced against
		// different points on the efficiency ramp.
		// attnSeconds is apart from the FLOPs terms because attention is priced by a
		// measured floor-and-rate form rather than from FLOPs.
		var flops, routedFLOPs, weightBytes, elementwise, attnSeconds float64
		// denseShapedSeconds is the dense-GEMM time when each GEMM is priced at its own
		// shape's efficiency. It replaces the flops-and-one-ramp path, so exactly one of
		// the two is populated per layer.
		var denseShapedSeconds float64
		// The routed-expert term is kept apart from the dense terms because it does NOT
		// compose with its own compute the way the dense primitives do. See the composition
		// below.
		var routedWeightBytes float64
		// attnSMSeconds is prefill attention, which binds on compute where decode binds on
		// bandwidth. kvFallback records that a decode request had no measured form, so the
		// raw KV read is charged instead.
		var attnSMSeconds float64
		kvFallback := false
		// recurrentSeconds, like attnSeconds, is priced by a measured floor-and-rate form
		// rather than from FLOPs.
		var recurrentSeconds float64

		if k.gemmShapeAware {
			// Each dense GEMM is priced at its OWN efficiency. A ramp in the token count
			// alone gives every GEMM in a layer the same fraction of peak, which is wrong
			// by the k factor: after TP=8 a projection's reduction depth can be 1024 while
			// an MLP's is 6144, and the sweep says those differ by more than 2x in
			// achieved efficiency at the same m.
			for _, g := range l.DenseGEMMs {
				// Tensor parallelism splits ONE of the two dimensions, and which one is
				// a property of the projection. The plan classifies it from the graph:
				// a column-parallel GEMM splits its output width, a row-parallel one its
				// reduction. Sharding the wrong axis would price mlp_gate_up's 6144-deep
				// reduction as 768-deep at TP=8, which is the widest GEMM in the layer.
				n, kk := float64(g.N), float64(g.K)
				if g.ShardN {
					n /= k.tp
				} else {
					kk /= k.tp
				}
				eff := price.ShapeEfficiency(tokensF, n, kk,
					k.epsMax, k.mHalf, k.nHalf, k.kHalf)
				if eff <= 0 {
					continue
				}
				denseShapedSeconds += tokensF * g.FLOPsPerToken / k.tp /
					(k.computeFLOPsPerSecond * eff)
			}
		} else {
			flops += tokensF * l.DenseFLOPsPerToken / k.tp
		}
		weightBytes += l.DenseWeightBytes / k.tp

		if l.ExpertFLOPsPerTokenPerExpert > 0 {
			// A rank computes only the tokens routed to the experts it holds.
			//
			// NOT divided by expertTensorShards, and that is a KNOWN over-charge rather
			// than an oversight. Under pure tensor parallelism localExpertShare is 1 --
			// every rank does hold every expert -- and vLLM shards the intermediate
			// dimension, `intermediate_size_per_partition = intermediate_size // tp_size`
			// (vllm/model_executor/layers/fused_moe/config.py:1321-1323), so a rank's
			// routed work is its slice's. Charging the whole expert over-prices routed
			// compute by the tensor-parallel width on the 400 of 591 InferenceX scenarios
			// that are pure TP. The BYTES do divide by it (routedWeightBytes, below), so the
			// two terms disagree about the same experts.
			//
			// Dividing it was implemented, tested and measured: it improves per-step
			// accuracy against NVIDIA's FPM whole-forward set, mape 27.45% to 23.03% over
			// 9,161 held-out steps, and costs 3.57 points of TPOT mape and 12.29 of TTFT
			// mape end to end. It is retained in this form on the evaluation evidence,
			// because the over-charge cancels another term that is not yet identified --
			// the sixth such cancellation this model is known to rest on. Correcting it
			// alone makes the kernel worse, so it waits for the term it offsets.
			//
			// docs/perf-model/hypothesis-log.md records the measurements on both sides.
			// ATTENTION-DP FUNNEL. With attention data parallelism every DP rank's
			// tokens are concatenated before expert routing, so the grouped GEMM sees
			// the whole replica group's tokens rather than one rank's. vLLM states it
			// for the naive dispatch -- "all DP ranks' tokens are concatenated before
			// routing" (fused_moe/routed_experts_capturer.py) -- and
			// allgather_reducescatter, the default (config/parallel.py:202), gathers every
			// DP rank's tokens before the expert GEMM whether it runs as that dispatch or
			// inside the modular kernel.
			// NVIDIA's own simulator applies the same factor exactly once before its
			// perf lookup (crates/core/src/perfmodel/operators/moe.rs,
			// `num_tokens.saturating_mul(self.attention_dp_size.max(1))`).
			//
			// Omitting it under-prices the routed term by the DP width, which is what
			// FPM's mixed rows measure: on MiniMax-M2.7 h200 the signed error runs
			// -14.9% at dp=1 tep2, -34.0% at dp=2 and -46.7% at dp=4, ordering by dp.
			routedTokens := tokensF * k.moeDPFunnel()
			routedPerRank := routedTokens * float64(l.TopK) * k.localExpertShare
			// With expert parallelism off and dp above one, the funnel's dp-fold tokens
			// each reach a 1/dp slice of every expert beyond the tensor-parallel one (the
			// MoE is flattened over dp x tp, see expertTensorShards), so the work per rank
			// divides by dp again. Only the dp part is divided: the tensor-parallel part is
			// the KNOWN over-charge above, kept as recorded, and dividing it here would
			// change every pure tensor-parallel deployment in the corpus.
			dpSlice := 1.0
			if k.layout.ExpertWidth <= 1 {
				dpSlice = k.expertTensorShards / k.tp
			}
			routedFLOPs += routedPerRank * l.ExpertFLOPsPerTokenPerExpert * k.moeImbalance /
				dpSlice
			// Expert weights read: the DISTINCT local experts this step touches, times
			// each one's per-rank bytes. Two things are separate and vLLM keeps them
			// separate too (`fused_moe/config.py`): WHICH experts a rank holds, and how
			// much of each. With expert parallelism a rank owns whole experts; without
			// it, every expert at a tensor-parallel slice. The total is the same and the
			// coverage is not, which a small batch exposes.
			//
			// The count that matters is how many distinct experts the step reaches. At
			// the batch sizes the "all local experts are read at decode" assumption was
			// written for that is all of them; below those it is not, since one token
			// reaches at most top_k.
			// The token count driving coverage is the SAME funnelled count that drives
			// the routed FLOPs above: with attention data parallelism a rank's experts
			// receive every DP rank's tokens, so they are reached by `routedTokens`
			// tokens, not by one rank's. Using the unfunnelled count here while the
			// FLOPs term uses the funnelled one is an internal inconsistency, and it
			// under-reads expert weights exactly where coverage is partial -- at the
			// batch 1-4 prefill rows that make up FPM's attention-DP cells.
			routedWeightBytes += l.ExpertWeightBytesPerExpert / k.expertTensorShards *
				price.ExpertsTouched(int(routedTokens+0.5), k.totalExperts, l.TopK,
					k.expertsPerRank)
			// A shared expert is dense: every token pays it, sharded like any projection.
			flops += tokensF * l.SharedExpertFLOPsPerToken / k.tp
			weightBytes += l.SharedExpertWeightBytes / k.tp
		}

		// Attention is priced by its own measured form, not from FLOPs.
		//
		// A FLOPs-and-bandwidth model of attention is wrong by an order of magnitude in both
		// regimes, and for the same two reasons in each: it charges nothing for the kernel's
		// launch and setup, and it assumes an efficiency the kernel does not reach. Decode is
		// off by 9.7x on L40S to 35.8x on GB200; prefill by 12.9x to 32.4x. So each regime
		// carries its own measured floor and scale, fitted per part.
		//
		//	decode:  floor + kv_bytes / rate            (bandwidth-bound in context)
		//	prefill: floor + causal_flops / (peak * eff * work_scale)
		//
		// A step holding both kinds of request pays both floors. That is the engine's
		// behaviour on the MLA and FlashInfer backends, which split a mixed batch and
		// launch a kernel per regime. KNOWN DIVERGENCE: FlashAttention runs one varlen
		// kernel over the whole batch (vllm/v1/attention/backends/flash_attn.py:1450-1478 at v0.31.0), so on it
		// a mixed step pays one floor and this charges one too many. Decode is charged to
		// HBM and prefill to SM, which is what each regime binds on.
		if l.AttnQHeads > 0 {
			// This layer kind's own decode terms, falling back to the part-wide pair.
			floor, rate := k.attentionFloor, k.attentionRate
			if fr, ok := k.attentionByKind[l.AttnKind]; ok {
				floor, rate = fr.floor, fr.rate
			}
			if decodeRequests > 0 && rate > 0 {
				// A sparse layer does not read the whole context. Bounded per request,
				// because min(sum) != sum(min): in a batch holding one request below the
				// top-k and one far above it, bounding the batch-wide total gets both
				// wrong.
				tokens := decodeKVTokens
				if bounded := k.dcpDecodeTokens(l, l.AttnKind, decodeContexts); bounded >= 0 {
					tokens = bounded
				}
				// The FLOOR stays outside the shard, deliberately. It is a KERNEL
				// LAUNCH-AND-SETUP cost, paid once per rank per invocation however many
				// tokens that rank holds, and sharding a sequence does not make a launch
				// cheaper.
				//
				// The registry's own numbers are what settle this, and they settle it
				// the opposite way round from the obvious argument. An MLA decode floor
				// is 9.5-14.5us across the parts that carry one -- essentially the same
				// as the part-wide attention floor, because blis-registry's
				// attention_decode_floor_mla IS that figure reused: "this is this part's
				// own measured attention-kernel decode floor, reused for the MLA kind"
				// (cost-model-attention.yaml, attention_decode_floor_mla rationale). The
				// larger module-derived floors, 4.7x-6.2x those, were measured and
				// REJECTED, because they cover the whole MLA block including
				// down-projections this kernel already prices as separate GEMM nodes.
				//
				// So the reason not to divide is not that an MLA setup is unusually
				// expensive -- it is not -- but that a floor of 9.5-14.5us is a launch
				// cost at any shard width. Dividing it by 8 would put it at 1.2-1.8us,
				// below any measured attention floor on any part in the registry, for a
				// kernel that still has to be launched on every rank.
				attnSeconds += floor.Seconds() +
					tokens*k.kvBytesPerToken/float64(k.plan.TotalLayers)/
						rate
			} else if decodeRequests > 0 {
				// No measured decode form for this part: the KV read is charged below and
				// the floor is lost. Provenance says the entries are missing.
				kvFallback = true
			}
			// A sliding-window layer attends within its window, not over the whole
			// prefix. The window is a per-LAYER property -- gpt-oss-120b alternates a
			// 128-token swa layer with a full gqa one -- so it cannot be applied to the
			// batch-wide causalFLOPs computed before this loop.
			layerCausalFLOPs := causalFLOPs
			if l.AttnWindow > 0 {
				layerCausalFLOPs = windowedCausalFLOPs(prefillChunks, l.AttnWindow)
			}
			if prefillRequests > 0 && k.attentionPrefillScale > 0 {
				scale := price.Efficiency(tokensF, k.epsMax, k.mHalf) *
					k.attentionPrefillScale
				attnSMSeconds += k.attentionPrefillFloor.Seconds() +
					layerCausalFLOPs*float64(l.AttnQHeads)*float64(l.AttnHeadDim)/k.tp/
						(k.computeFLOPsPerSecond*scale)*smDerate
			} else if prefillRequests > 0 {
				// No measured prefill form: fall back to the ramp unmodified, which the
				// sweeps say understates by over ten times.
				flops += layerCausalFLOPs * float64(l.AttnQHeads) *
					float64(l.AttnHeadDim) / k.tp
			}

			// Secondary attention kernels this layer launches. A block-sparse layer
			// scores which blocks to read before reading them, and that scorer is a
			// separate kernel over a separate, narrow cache: its bytes come from its own
			// head geometry rather than from the engine's KV figure, and its read is
			// bounded by context where the attention it feeds is bounded by the top-k.
			//
			// Charged to HBM like the primary decode term. Omitting it made selection
			// free, which understates every block-sparse layer -- on MiniMax-M3 the
			// dropped read is 0.94x the attention read at 8K context and 15x at 131K.
			for _, a := range l.Attentions {
				if a.HoldsKV() || decodeRequests <= 0 {
					continue
				}
				floor, rate := k.attentionFloor, k.attentionRate
				if fr, ok := k.attentionByKind[a.Kind]; ok {
					floor, rate = fr.floor, fr.rate
				}
				if rate <= 0 {
					continue
				}
				// Scale the engine's per-token KV figure by this kernel's own head
				// geometry over the primary's. Derived rather than recomputed from the
				// cache dtype so the two cannot drift: kvBytesPerToken already carries
				// the cache width, the tensor-parallel division and the replication
				// floor that a head is never split across ranks.
				primary := float64(l.AttnKVHeads) * float64(l.AttnHeadDim)
				if primary <= 0 {
					continue
				}
				ratio := float64(a.KVHeads) * float64(a.HeadDim) / primary
				perToken := k.kvBytesPerToken / float64(k.plan.TotalLayers) * ratio
				tokens := decodeKVTokens
				if a.Window > 0 {
					tokens = math.Min(tokens, float64(a.Window)*float64(decodeRequests))
				}
				// A pooled scorer reads cached STATES, not tokens: the kpool indexer
				// compresses index_kpool tokens into one state and scores over those,
				// so the candidate count is the token count over the ratio. This is the
				// scan bound, not a width -- perToken already carries the state's own
				// head geometry. GLM-5.3-Flash pools 4:1, so an unpooled reading charges
				// four times the candidates on the one term in the layer that grows with
				// context. Applied after the window bound because the two compose: the
				// window fixes the span, the ratio how many states it holds.
				if a.CompressRatio > 1 {
					tokens /= float64(a.CompressRatio)
				}
				// This kernel's OWN kind decides whether its cache shards, not the
				// layer's primary: a block-index scorer keeps a separate cache, which is
				// what HoldsKV distinguishes. perToken is deliberately not touched -- it
				// carries head geometry and a cache width, and dividing bytes per token
				// rather than the token count is the error this shard exists to avoid.
				//
				// The scorer's bounds above are already applied to the batch total, so
				// the shard is taken on that total rather than per request. The
				// difference from per-request sharding is the rounding of one token per
				// request, on a term that is a fraction of the layer -- against the
				// primary read, where it is applied per request because that is where it
				// can matter.
				if k.layout.DCP > 1 && dcpShardsKV(a.Kind) {
					tokens = dcpLocalTokens([]int{int(tokens + 0.5)}, k.layout.DCP,
						k.decodeContext.Interleave)
				}
				attnSeconds += floor.Seconds() + tokens*perToken/rate
			}
		}
		elementwise += tokensF * l.ElementwiseBytesPerToken / k.tp

		// A recurrent state update, priced by its own measured floor and rate. The cost
		// model charged this at zero, which omits a real term on a hybrid stack: the
		// kernel fires on 48 of Nemotron-3-Ultra's 108 layers and on all 69 of Kimi-K3's.
		//
		// Charged to SM, because a recurrent update is a compute kernel whose cost is
		// launch and sequential arithmetic rather than a bandwidth read. It is sequential
		// in the state dimension, which is why its token rate is low against a GEMM's.
		//
		// A family with no measured entries contributes nothing and Provenance says the
		// entries are missing — the same failure the whole term used to have, but now
		// visible rather than silent.
		if l.RecurrentKind != "" {
			if fr, ok := k.recurrent[l.RecurrentKind]; ok {
				recurrentSeconds += fr.floor.Seconds() + tokensF/fr.rate
			}
		}

		// One layer's KV read. Only charged here when attention has no measured form for
		// this part; otherwise the attention term already carries these bytes and adding
		// them again would double-count.
		var kvBytes float64
		if l.AttnQHeads > 0 && kvFallback {
			tokens := decodeKVTokens
			if bounded := k.dcpDecodeTokens(l, l.AttnKind, decodeContexts); bounded >= 0 {
				tokens = bounded
			}
			kvBytes = tokens * k.kvBytesPerToken / float64(k.plan.TotalLayers)
		}

		// The efficiency ramp is evaluated at the STEP's token count for every term,
		// including the grouped GEMM, and that is a finding rather than a simplification.
		//
		// The cost model's §2.1 argues the grouped GEMM should see a smaller argument — the
		// rows routed to one expert rather than the whole batch — and the reasoning is
		// sound for a LOW-top_k model, one routing to 8 of a few hundred experts. Scored
		// against four published deployments the per-expert argument was worse overall: it
		// improved the two low-top_k arms (31% to 18% MAPE on one, 64% to 41% on Kimi-K3)
		// and ruined Nemotron-3-Ultra (87% to 158%), which routes each token to 22 of 512
		// experts. The aggregate went from 48% to 56%. Two of those four arms have since
		// left the corpus with their model, so the figures are kept as the record of why
		// this choice was made rather than as a re-runnable result.
		//
		// The reason the per-expert argument fails at high top_k is visible in its own
		// arithmetic: at one token routed to 22 experts it gives one row per expert and
		// prices the whole routed term at about 1% of peak, where the measurements say a
		// single-token MoE step reaches about 4%. A real grouped GEMM batches those
		// single-row problems into one kernel, so the machine sees more parallelism than
		// the per-expert view credits.
		//
		// Neither argument is measured. AISimulate's MoE tables are a shape-indexed latency
		// lookup rather than a FLOPs-and-efficiency model, so they state no grouped-GEMM
		// ramp to fit; an attempt to recover one yielded implied efficiencies above peak,
		// because the normalization those tables use is not documented with the data. So
		// this is the better of two assumptions on the evidence available, and the SM term
		// for a high-top_k MoE model is the least certain part of this kernel.
		ramp := price.Efficiency(tokensF, k.epsMax, k.mHalf)
		denseSM := flops/(k.computeFLOPsPerSecond*ramp)*smDerate +
			denseShapedSeconds*smDerate + recurrentSeconds + attnSMSeconds
		denseHBM := (weightBytes+kvBytes+elementwise)/k.hbmBytesPerSecond + attnSeconds

		// The routed-expert grouped GEMM composes its compute and memory as a SUM, where
		// every other primitive in this layer composes as a max.
		//
		// That asymmetry is measured, not assumed. Priced against NVIDIA's own MoE tables --
		// 3,223 shape comparisons over roughly 70 geometries per part, on the fastest kernel
		// lane, expert parallelism off, balanced routing -- the sum fits the term's growth in
		// token count better than the max on every part tested:
		//
		//	                max(c,m)    c+m
		//	h200  fp8         64.04%  55.98%
		//	h100  fp8         53.41%  44.91%
		//	b200  nvfp4      101.49%  79.70%
		//	b300  nvfp4      116.70%  91.95%
		//
		// The mechanism is visible in the shapes: measured MoE latency grows 3.58x from 4 to
		// 256 tokens where the max-composed prediction grows 8.38x. ExpertsTouched rises 30x
		// over that range, so above about 32 tokens the weight-read term dominates the max and
		// drags the prediction up. A real grouped GEMM amortises each expert's weight read
		// against the work done on that expert; a max between the two credits the whole read
		// as free whenever compute exceeds it, and the sum is the closer of the two available
		// approximations.
		//
		// The same test on DENSE GEMM says the opposite -- max 8.75% against sum 15.91% on
		// h200 fp8 -- which is why this is scoped to the routed term rather than applied to
		// the layer. A blanket sum improves the end-to-end score more (10.07% against 13.86%)
		// and is wrong for the dense primitives, so it is not taken.
		routedSM := routedFLOPs / (k.computeFLOPsPerSecond * ramp) * smDerate
		routedHBM := routedWeightBytes / k.hbmBytesPerSecond
		routed := routedSM + routedHBM

		// Each half of the routed cost is charged to the resource that actually does it, so
		// the per-resource breakdown remains a true account of where time went: the compute
		// half to SM, the memory half to HBM. An earlier version charged the WHOLE routed cost
		// to whichever half was larger, which doubled one resource and zeroed the other --
		// four existing tests caught it, including one asserting that withholding SMs cannot
		// move an HBM-bound step and one asserting routed compute falls as the expert group
		// widens.
		//
		// The sum is then realised in the stage composition below rather than here: the
		// routed halves are excluded from the per-layer max and added to it, which is what
		// makes the MoE term compose as a sum while every other term still composes as a max.
		sm := denseSM + routedSM
		hbm := denseHBM + routedHBM

		// Collectives in this layer, each priced against its own floor and rate. Every
		// launch pays its own floor: summing bytes across layers and applying one floor
		// understates a floor-dominated step by the layer count.
		var onNode, crossNode float64
		for _, c := range l.Collectives {
			bytes := tokensF * c.BytesPerToken
			if c.RoutedByTopK && l.TopK > 0 && k.routedAll2All() {
				bytes *= float64(l.TopK)
			}
			// Whether a collective leaves the node depends on ITS OWN group, not the
			// deployment's widest: a tensor-parallel reduction across 8 ranks stays on an
			// 8-GPU node even when expert parallelism spans nine of them.
			key := collKey{Op: c.Op, Group: c.Group}
			elapsed := 0.0
			if k.crossesNodes(key) {
				elapsed = k.collectiveSeconds(key, bytes*k.spanFor(key))
				crossNode += elapsed
			} else {
				elapsed = k.collectiveSeconds(key, bytes)
				onNode += elapsed
			}
		}

		// Decode-context parallelism's own collectives, which no model graph can emit.
		//
		// Sharding a sequence by token means no rank holds the whole context, so each
		// computes a PARTIAL attention output and the group must combine them. That combine
		// is the cost DCP trades for the capacity it buys, and omitting it would make DCP
		// look free in both directions. dcpDecodeCollectives prices exactly the set vLLM
		// v0.31.0 launches for the resolved backend and layout.
		//
		// NOT planned as a PlannedCollective: the plan is per-layer-kind and
		// shape-independent, while whether this fires depends on the batch holding decode
		// rows at all. It accumulates into the same onNode/crossNode totals the graph
		// collectives use, so composition and the per-resource breakdown treat it
		// identically.
		if decodeTokens > 0 && k.layout.DCP > 1 && dcpShardsKV(l.AttnKind) &&
			l.AttnQHeads > 0 {
			on, cross := k.dcpDecodeCollectives(l, decodeTokens)
			onNode += on
			crossNode += cross
		}

		// Prefill-context parallelism's own collectives, which no model graph emits either.
		//
		// A PCP rank computes only its share of a prefill, so it writes only its share of
		// the new KV. Every rank must still hold the WHOLE cache, because PCP "does not
		// increase the KV-cache shard count" (vllm/config/parallel.py:131-133 at v0.31.0)
		// -- that is what distinguishes it from DCP. So the ranks all-gather the cache
		// inputs they just computed, on the token dimension, carrying PREFILL tokens only:
		// "Keep replicated decode writes local and gather partitioned prefills"
		// (vllm/v1/attention/ops/pcp.py:16-50), since a replicated decode row is already
		// on every rank. See pcpPrefillGathers for what crosses.
		if prefillTokens > 0 && k.layout.PCP > 1 && l.AttnQHeads > 0 {
			on, cross := k.pcpPrefillGathers(l, prefillTokens)
			onNode += on
			crossNode += cross
		}

		// Max across resources within the layer, summed over the layers of this kind --
		// except that the routed-expert term composes as a sum rather than a max, for the
		// measured reason given above. So the max is taken over the layer WITHOUT the routed
		// halves, and the routed sum is added to it.
		stage := sm - routedSM
		for _, d := range [...]float64{hbm - routedHBM, onNode, crossNode} {
			if d > stage {
				stage = d
			}
		}
		stage += routed
		overlapSeconds += count * stage

		perResource[idxSM] += count * sm
		perResource[idxHBM] += count * hbm
		perResource[idxNVLink] += count * onNode
		perResource[idxNIC] += count * crossNode
	}

	// The head runs once per step rather than per layer, and shards over the vocabulary.
	headFLOPs := tokensF * k.plan.Head.DenseFLOPsPerToken / k.tp
	headSM := headFLOPs / (k.computeFLOPsPerSecond *
		price.Efficiency(tokensF, k.epsMax, k.mHalf)) * smDerate
	headHBM := k.plan.Head.DenseWeightBytes / k.tp / k.hbmBytesPerSecond
	perResource[idxSM] += headSM
	perResource[idxHBM] += headHBM
	if headSM > headHBM {
		overlapSeconds += headSM
	} else {
		overlapSeconds += headHBM
	}

	// The host is additive against everything: the CPU is a resource that cannot overlap
	// with itself, and its work is not hidden behind a stage the way device work is. Per
	// kernel dispatch joins it, for the same reason — a kernel cannot run before it has
	// been dispatched.
	host := k.hostPerStep(b) + seconds(k.launchSeconds())
	perResource[idxHost] = host.Seconds()

	// The breakdown map is the only thing that can allocate here. A caller that supplied
	// one gets no allocation at all; one that did not gets a map sized to the resources
	// actually in play.
	per := into
	if per == nil {
		var live int
		for _, elapsed := range perResource {
			if elapsed > 0 {
				live++
			}
		}
		per = make(map[kernel.Resource]time.Duration, live)
	}
	noOverlap, bottleneck := summarize(&perResource, per)
	overlap := seconds(overlapSeconds) + host

	if overlap > noOverlap {
		// Per-layer composition can exceed the sum of per-resource totals only through
		// rounding, since each stage's max is one of its own terms. Clamp rather than
		// report a band the wrong way round.
		overlap = noOverlap
	}
	// Which edge to believe, for a caller that wants one number: NoOverlap, on measured
	// evidence rather than preference. Over 219 points of NVIDIA's FPM whole-forward
	// dataset -- two models, two parts, five parallelism topologies -- Overlap's SIGNED
	// error is -13.45% against NoOverlap's -3.44%, and NoOverlap is closer on 158 of them.
	// A one-sided error of that size is a missing term rather than scatter, and it matches
	// the -13.10% deficit this kernel shows end to end on an independent serving corpus.
	// The physical reason is PIECEWISE cudagraph mode: attention runs eagerly between
	// captured segments, so per-layer overlap is structurally limited.
	//
	// A concentration-aware choice would refine this -- where one resource holds most of a
	// step, per-stage max IS right, and the single dissenting cell in that evidence is a
	// whole model on two GPUs. Five cells is enough to reject the optimistic edge as a
	// universal default and not enough to fit a blend.
	//
	// schemas v0.2.0 removed StepEstimate.Expected, so the kernel no longer names an edge
	// in the estimate and this paragraph is the only place the choice is recorded. It is
	// kept because it is measured rather than rederivable: a caller holding only Overlap
	// and NoOverlap cannot tell which the evidence favours, and the band is wide enough
	// that guessing picks a different number. A caller that wants one figure should read
	// NoOverlap.
	return kernel.StepEstimate{
		Overlap: overlap, NoOverlap: noOverlap,
		Bottleneck: bottleneck, PerResource: per,
	}
}

// collectiveTime prices one collective's bytes against its own measured floor and
// rate. Both are per operation: charging an all-to-all the all-reduce floor would
// overstate it by 2.2x on H200, and the rates differ by a similar factor.
func (k *Kernel) collectiveSeconds(key collKey, bytes float64) float64 {
	peak, ok := k.collectiveRates[key]
	if !ok || peak <= 0 {
		// Every op the plan can emit is resolved at construction, so this is
		// unreachable for a kernel New returned. Returning infinity rather than zero
		// keeps an unpriced collective visible if that ever changes.
		return math.Inf(1)
	}
	return price.CollectiveTime(bytes, k.collectiveTransitions[key], peak,
		k.collectiveFloors[key].Seconds())
}

// collectiveTime prices one invocation, for callers outside the step loop.
func (k *Kernel) collectiveTime(key collKey, bytes float64) time.Duration {
	return seconds(k.collectiveSeconds(key, bytes))
}

// crossesNodes reports whether a collective's own group spans more than one node.
//
// Per collective rather than per deployment: a tensor-parallel group of 8 fits inside an
// 8-GPU node however wide expert parallelism is, and an expert-parallel group of 72 does
// not. Using the deployment's widest group for both would move every reduction onto the
// NIC in a wide-EP layout, where in fact it never leaves the node.
//
// And by PLACEMENT rather than by width alone, because a group's members are not always
// neighbours. See groupPerNode: a two-rank prefill-context group at tp=8 has one member on
// each of two nodes, which a width test reads as fitting comfortably inside one.
func (k *Kernel) crossesNodes(key collKey) bool {
	if k.layout.GPUsPerNode <= 0 {
		return false
	}
	width, ok := k.groupSize(key.Group)
	if !ok {
		return false
	}
	return width > k.groupPerNode(key.Group)
}

// groupPerNode is how many of one group's ranks share a node, which is what decides both
// whether the group crosses a node boundary and how much of its traffic does.
//
// THE ENGINE'S RANK LAYOUT DECIDES IT. vLLM numbers ranks ExternalDP x DP x PP x PCP x TP
// with the tensor-parallel axis innermost (vllm/distributed/parallel_state.py:2045,
// :2054-2060 at v0.31.0), and a node holds consecutive ranks. So a group's members sit
// `stride` ranks apart, where the stride is the product of the axes inside the one the
// group runs along. With pp = 1, which every layout this kernel prices has (no collective
// spans the pipeline axis):
//
//	tensor-parallel     contiguous                                     stride 1
//	expert-parallel     one contiguous DP x PCP x TP block (:2212-2220)   stride 1
//	prefill-context     the PCP axis, at a fixed TP rank (:2155-2160)     stride tp
//	decode-context      within TP when pcp is 1; along the PCP axis when  stride 1, tp
//	                    dcp == pcp; the whole TP x PCP block when          or 1
//	                    dcp == tp*pcp (:2139-2144)
//
// and a node of g GPUs holds g/stride of them. At tp=8 on 8-GPU nodes a prefill-context
// group therefore has ONE member per node: every hop of its ring crosses the fabric. A
// width test priced exactly that gather on NVLink.
//
// A stride that does not divide the node (tp=6 on 8-GPU nodes) has no uniform answer, and
// is floored -- which charges more of the group to the fabric than some nodes carry, the
// conservative direction.
//
// COVERAGE LIMIT: a rack that is one NVLink domain (an NVL72-class part, four GPUs to a
// tray) is not consulted here, for any group. ResolveLayout treats such a rack as one node
// when it counts NodesSpanned, but a group spanning trays inside it is still priced as
// crossing, scaled by whatever inter-node ratio the deployment's fabric states.
func (k *Kernel) groupPerNode(group price.GroupAxis) int {
	gpn := k.layout.GPUsPerNode
	if gpn <= 0 {
		return math.MaxInt
	}
	stride := 1
	switch group {
	case price.GroupPCP:
		stride = max(k.layout.TP, 1)
	case price.GroupDCP:
		// dcp is 1, pcp or tp*pcp once pcp exceeds one (parallel.py:571-578); of those,
		// only dcp == pcp runs along the PCP axis rather than over a contiguous block.
		if k.layout.PCP > 1 && k.layout.DCP <= k.layout.PCP {
			stride = max(k.layout.TP, 1)
		}
	}
	return max(gpn/stride, 1)
}

// groupSize returns the rank count a collective's group spans.
//
// Read from the GROUP rather than inferred from the op, because the op no longer decides
// it: an all-gather spans tp ranks on the tensor-parallel axis and dcp ranks on the
// decode-context-parallel one.
func (k *Kernel) groupSize(group price.GroupAxis) (int, bool) {
	switch group {
	case price.GroupTP:
		return max(k.layout.TP, 1), true
	case price.GroupExpert:
		return max(k.layout.MoEGroup(), 1), true
	case price.GroupDCP:
		return max(k.layout.DCP, 1), true
	case price.GroupPCP:
		return max(k.layout.PCP, 1), true
	}
	// EXHAUSTIVE RATHER THAN DEFAULTING TO THE TENSOR-PARALLEL WIDTH. An axis this
	// function cannot name has no width, and pricing it at tp anyway is the exact failure
	// the composite key exists to prevent: an all-gather's 8-rank floor is 1.35x to 1.59x
	// its 4-rank figure across the nine parts measured at both, so a wrong axis is not a
	// rounding error.
	// Pipeline parallelism is the obvious future candidate -- Layout.PP exists and no
	// collective spans it yet -- and it should arrive as a refusal rather than as a
	// silently tensor-parallel price.
	//
	// This is the half of the resolve.Emits/Recognizes pair that a bare default would
	// have left out: that pair exists so an unrecognized condition can be refused rather
	// than quietly answered, and a width is no different.
	return 0, false
}

// spanFor returns the cross-node scaling a collective pays. A ring reduces as it travels
// and so crosses only a fraction of the fabric; a routed all-to-all must reach every peer.
func (k *Kernel) spanFor(key collKey) float64 {
	ratio := k.fabric.Ratio()
	group, _ := k.groupSize(key.Group)
	perNode := k.groupPerNode(key.Group)
	// An all-to-all is point to point unless it is the MoE dispatch over a backend that
	// moves ring-shaped volume. The decode-context a2a is all_to_all_single over its group
	// (dcp.py:1000-1005), every rank sending a slice to every other.
	if key.Op == model.OpAll2All && (key.Group != price.GroupExpert || k.routedAll2All()) {
		return price.All2AllSpan(group, perNode, ratio)
	}
	return price.RingSpan(group, perNode, ratio)
}

// routedAll2All reports whether the resolved MoE backend moves top-k-selected bytes point
// to point, rather than dense per-token bytes in two ring phases. The distinction changes
// both the volume and the cross-node scaling, and it is a property of the backend rather
// than of the primitive's name.
func (k *Kernel) routedAll2All() bool {
	if !k.pool.Parallel.EnableExpertParallel {
		// With expert parallelism off, a data-parallel MoE always falls back to the
		// all-gather/reduce-scatter dispatch, whatever backend is named
		// (vllm/model_executor/layers/fused_moe/all2all_utils.py:202-214 at v0.31.0).
		return false
	}
	switch k.pool.Engine.All2AllBackend {
	case "", "naive", "allgather_reducescatter":
		return false
	}
	return true
}

// chunk is one prefill request's shape within a step: `sched` new tokens resuming on a
// prefix of `prefix` already-computed ones.
type chunk struct {
	sched  int
	prefix int
}

// windowedCausalFLOPs is the attention FLOPs a sliding-window layer pays for these
// prefill chunks. A query at absolute position p attends keys in [p-window+1, p], so it
// reads min(p+1, window) of them rather than the whole prefix.
//
// For a chunk of `s` tokens resuming at prefix `c`, the queries whose own position still
// fits inside the window are the first max(0, min(s, window-c)); each later query reads
// exactly `window` keys. Summing the two regimes:
//
//	nSmall = max(0, min(s, window-c))
//	pairs  = nSmall*(c+1) + nSmall*(nSmall-1)/2 + (s-nSmall)*window
//
// The 2*2 factor matches the unwindowed term above: two FLOPs per multiply-accumulate,
// and two matmuls (scores, then the value-weighted sum).
//
// The bound SATURATES at s*window however long the prefix grows, where the unwindowed
// count grows linearly in the prefix without bound. The over-charge this corrects is
// therefore governed by PREFIX length, not chunk size: about 4x at an 8,192-token prefix
// for a 1,024-token chunk with a 2,176 window, 15x at 32,768 and 60x at 131,072.
// docs/perf-model/hypothesis-log.md records the brute-force verification.
func windowedCausalFLOPs(chunks []chunk, window int) float64 {
	if window <= 0 {
		return 0
	}
	w := float64(window)
	var pairs float64
	for _, c := range chunks {
		s, prefix := float64(c.sched), float64(c.prefix)
		nSmall := math.Min(s, w-prefix)
		if nSmall < 0 {
			nSmall = 0
		}
		// Queries still inside the window read their whole prefix plus themselves;
		// the rest read exactly `window`. The 0.5*nSmall subtraction drops the
		// diagonal's half-token so this matches the unwindowed term's continuum
		// convention (s*c + s^2/2) rather than exceeding it by s/2 when the window
		// is wide enough to be inactive.
		pairs += nSmall*(prefix+1) + nSmall*(nSmall-1)*0.5 - nSmall*0.5
		pairs += (s - nSmall) * w
	}
	return 2 * 2 * pairs
}

// selectedKVTokens is how many KV tokens a SPARSE layer's decode actually reads, summed
// over the batch. It returns -1 for a layer that reads its whole context, so a caller can
// tell "no bound" from "a bound that happens to equal the context".
//
// A sparse-MLA layer reads two tiers, which is what the measurement shows rather than
// what the name suggests:
//
//	selected = min(ctx, topk) + max(0, ctx-topk)/ratio
//
// The first tier is the top-k the indexer picked, read at full resolution and FLAT in
// context. The second is the remainder, present only where the architecture compresses it
// instead of discarding it, and divided down by that ratio. Both are capped by the context,
// since a 16-token context cannot yield 128 selected tokens -- which is why the measured
// curve is flat at small context rather than constant.
//
// A layer selecting by a sliding WINDOW states a window and no top-k, in which case the
// window is the selected count: that is how blis-catalog states DeepSeek-V4-Pro's
// csa128_moe layer, against csa4_moe's `index_topk: 1024`.
//
// WHY THIS IS NOT OPTIONAL. Without it the only available count is the full context, which
// overstates a 1M-token decode read on this architecture by 128x. Fitting a rate against
// that reading forces it to 1.00 of datasheet peak -- physically impossible, and the clamp
// is how the wrong byte count announces itself (blis-registry
// scripts/fit_attention_sparse_mla.py reports 1.698x error clamped, against 1.301x
// interior for the form above). Measured on h200, a batch-1 decode costs 9.8-13.7us flat
// from a 0 to a 16,384-token context and only 30.6us at 1,048,575.
//
// Summed per request rather than over the batch total, because min(sum) != sum(min): a
// batch holding one request below the top-k and one far above it gets both wrong if the
// bound is applied to the aggregate.
//
// NO LONGER ON THE PRICING PATH. dcpDecodeTokens composes this bound with the
// decode-context shard per request and is what stepTime calls; this remains as the
// selection bound stated on its own, which is what the sparse-MLA tests pin. The two
// share sparseTopK and selectedKVTokensFor, so what those tests establish still holds for
// the pricer.
func selectedKVTokens(l *price.PlannedLayer, contexts []int) float64 {
	// Delegates to sparseTopK rather than re-deriving the bound, so the claim that the
	// two cannot disagree about which layers are sparse is structural. A zero means
	// either a non-sparse kind or a sparse layer stating neither bound -- blis-schemas
	// rejects the latter, so reaching it means a graph bypassed validation, and charging
	// the full context is the conservative reading.
	topk := sparseTopK(l)
	if topk <= 0 {
		return -1
	}
	var total float64
	for _, ctx := range contexts {
		total += selectedKVTokensFor(l, ctx, topk)
	}
	return total
}

// selectedKVTokensFor is one request's selected count, which selectedKVTokens sums. Split
// out so a per-request shard can be applied to each request's own selected count rather
// than to the batch total: the two differ once a sharded count is rounded per request,
// and the engine itself keeps a per-request vector (dcp_local_seq_lens,
// vllm/v1/attention/backend.py:427-428, built per request in flash_attn.py:884-893).
func selectedKVTokensFor(l *price.PlannedLayer, ctx, topk int) float64 {
	sel := math.Min(float64(ctx), float64(topk))
	if l.AttnCompressRatio > 0 && ctx > topk {
		sel += float64(ctx-topk) / float64(l.AttnCompressRatio)
	}
	return sel
}

// sparseTopK is the bound a sparse-MLA layer selects within, or zero when this layer does
// not select.
//
// The single source of that decision: both selectedKVTokens and dcpDecodeTokens call it,
// so neither can disagree with the other about which layers are sparse or which bound
// they select within. An earlier shape of this file derived the bound twice, and the two
// copies were free to drift while a passing suite said nothing.
func sparseTopK(l *price.PlannedLayer) int {
	if l.AttnKind != model.AttentionSparseMLA {
		return 0
	}
	if l.AttnIndexTopK > 0 {
		return l.AttnIndexTopK
	}
	if l.AttnWindow > 0 {
		return l.AttnWindow
	}
	return 0
}

// pcpLocalTokens is how many scheduled tokens the BUSIEST prefill-context-parallel rank
// computes, for one prefill chunk of `sched` tokens. It returns -1 when prefill-context
// parallelism is off, so a caller keeps the whole chunk.
//
// PCP SPLITS THE PREFILL ITSELF, which is its defining purpose: vLLM calls it the "Number
// of ranks that split prefill sequence computation" (vllm/config/parallel.py:131-133 at
// v0.31.0). A rank builds a LOCAL batch of its own share and runs the forward on that
// (num_local_tokens, vllm/v1/worker/gpu/pcp_manager.py:485), so every term proportional to
// scheduled tokens falls with the split -- not only attention.
//
// THE SPLIT IS A ZIGZAG, NOT A SLICE, and that is the whole reason a single divisor is
// defensible here. _iter_rank_chunks (pcp_manager.py:235-273) cuts each prefill into
// 2*pcp chunks and gives rank r chunks r and 2*pcp-1-r, which its own docstring draws:
//
//	full:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     (pcp = 4)
//	rank 0:  0                           7
//	rank 1:      1                   6
//	rank 2:          2           5
//	rank 3:              3   4
//
// Pairing a low chunk with a high one is what balances CAUSAL work: an early chunk has few
// keys to its left and a late one has many, and the two sum to the same total on every
// rank. Verified by enumeration: at pcp 2, 4 and 8 the per-rank causal pair counts are
// EXACTLY equal -- max over ranks divided by the ideal share is 1.000000 with zero spread
// -- for unchunked prefills and for chunked ones alike (tested at 2,048 tokens on no
// prefix, 4,096 on 8,192, 1,024 on 131,072, 8,192 on none). A plain contiguous 1/pcp slice
// would NOT balance, because the last rank would carry the whole upper triangle.
//
// THE BUSIEST RANK BINDS, as it does for DCP: the ranks must gather before the next layer,
// so the step waits for the one with the most work. Where 2*pcp divides the query every
// rank holds 2*ceil(sched/(2*pcp)) tokens and the shards sum to sched exactly; where it
// does not, the ragged remainder lands unevenly and the maximum is taken over the ranks
// rather than assumed. The overhead over a nominal sched/pcp is bounded and small at any
// realistic chunk -- under 1.5% above 1,024 tokens and under 0.4% above 4,096, for pcp up to 8 -- and it is
// an OVERSTATEMENT, which is the safe direction.
//
// A SHORT PREFILL IS REPLICATED RATHER THAN SPLIT, BUT ONLY ALONGSIDE DCP. That gate is
// easy to miss and changes the answer, so it is taken from the engine verbatim:
// replicated_requests (pcp_manager.py:222-233) computes `drops_a_chunk` only inside
// `if self.dcp_world_size > 1`, so with PCP alone every prefill is partitioned however
// short it is, and a rank holding no chunk simply contributes nothing.
//
// The condition itself is not "shorter than 2*pcp" either, which is the obvious guess and
// wrong: it is `(2*pcp - 1) * ceil(sched/(2*pcp)) >= sched`, which also fires at lengths
// well above 2*pcp -- at pcp 8 it is true at 17 tokens and false at 16 -- because what it
// detects is a partition in which some chunk comes out empty.
//
// `dcp` is therefore a parameter rather than an assumption: it decides whether the
// replication branch exists at all.
func pcpLocalTokens(sched, pcp, dcp int) int {
	if pcp <= 1 || sched <= 0 {
		return -1
	}
	numChunks := 2 * pcp
	chunk := (sched + numChunks - 1) / numChunks
	if dcp > 1 && (numChunks-1)*chunk >= sched {
		// A chunk would come out empty, so the engine replicates instead: every rank
		// computes the whole query.
		return sched
	}
	// Rank r holds chunks r and 2*pcp-1-r, each truncated where it runs past the query.
	// Maximised over the ranks rather than taken as 2*chunk: that shortcut is right only
	// when every chunk is full, and it overstates a ragged partition -- at 2 tokens over
	// 2 ranks the chunks are one token each and the busiest rank holds ONE, where 2*chunk
	// capped at the query would say two.
	most := 0
	for rank := 0; rank < pcp; rank++ {
		local := 0
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := min(idx*chunk, sched)
			hi := min(lo+chunk, sched)
			if hi > lo {
				local += hi - lo
			}
		}
		if local > most {
			most = local
		}
	}
	return most
}

// appendPrefillRuns appends the prefill runs this rank computes for one request, reading
// the split widths from the resolved layout.
//
// A method rather than a bare call so that the chunk geometry stepTime depends on is
// reachable from outside stepTime: the quantity is a local, and the step time it feeds is
// too insensitive to pin the geometry through (a dense GEMM dominates a windowed attention
// term roughly five to one, so the difference between a positional reading and a
// by-token-count one is a few percent of SM). Exercising the same path the pricer takes is
// what makes that geometry testable at all.
func (k *Kernel) appendPrefillRuns(dst []chunk, sched, prefix int) []chunk {
	return appendPCPChunks(dst, sched, prefix, k.layout.PCP, k.layout.DCP)
}

// appendPCPChunks appends the prefill runs THIS RANK computes, each at its true global
// offset, so a positional cost law sees where the tokens sit rather than only how many
// there are.
//
// With prefill-context parallelism off, or for a request the engine replicates, that is
// the one run the caller would have appended anyway: the whole query at its own prefix.
// With the split on, it is the two zigzag chunks rank r owns -- chunk r near the cheap
// start and chunk 2*pcp-1-r near the expensive end -- at offsets `prefix + lo`.
//
// WHICH RANK, and why it is the same one the token count comes from: pcpLocalTokens
// maximises the token count over the ranks, so the chunks appended here are that rank's.
// For a sliding-window layer the busiest rank by token count is not necessarily the
// costliest by key count, but the two agree wherever the partition is even, and a window
// term driven by the wrong rank of an uneven partition is a second-order error against
// the 7.9x one this function exists to remove.
func appendPCPChunks(dst []chunk, sched, prefix, pcp, dcp int) []chunk {
	local := pcpLocalTokens(sched, pcp, dcp)
	if local < 0 || local == sched {
		// Not split, or replicated whole onto every rank.
		return append(dst, chunk{sched: sched, prefix: prefix})
	}
	numChunks := 2 * pcp
	size := (sched + numChunks - 1) / numChunks
	// The rank holding the most tokens, matching pcpLocalTokens' own aggregation.
	busiest, most := 0, 0
	for rank := 0; rank < pcp; rank++ {
		held := 0
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := min(idx*size, sched)
			hi := min(lo+size, sched)
			if hi > lo {
				held += hi - lo
			}
		}
		if held > most {
			busiest, most = rank, held
		}
	}
	for _, idx := range [...]int{busiest, numChunks - 1 - busiest} {
		lo := min(idx*size, sched)
		hi := min(lo+size, sched)
		if hi > lo {
			dst = append(dst, chunk{sched: hi - lo, prefix: prefix + lo})
		}
	}
	return dst
}

// pcpCausalPairs is the attention work the BUSIEST prefill-context-parallel rank does for
// one prefill chunk, in the same units the unsharded count uses: four times the
// query-key pairs, since there are two FLOPs per multiply-accumulate and two matmuls.
//
// Counted over the engine's own partition rather than scaled by a ratio, because no ratio
// is right in general. A query at chunk-local offset j attends its whole prefix plus its
// causal share within the chunk, so it reads prefix + j + 1 keys; rank r owns chunks r and
// 2*pcp-1-r. Where 2*pcp divides the query the zigzag pairing makes every rank's total
// EXACTLY equal and this reduces to the whole count over pcp -- verified equal to
// 1.000000 of the ideal share with zero spread across ranks, at pcp 2, 4 and 8, with and
// without a prefix. Where it does not divide, the busiest rank carries more than either
// 1/pcp or its token share: at 100 tokens over 8 ranks it carries 0.1568 of the pairs
// against 0.125 and 0.14 respectively, because a ragged partition hands one rank two short
// chunks from the expensive end.
//
// A replicated request keeps its whole count, since every rank computes all of it. The
// loop below produces that naturally: pcpLocalTokens returning the full length means the
// chunk indices cover the query.
func pcpCausalPairs(sched, prefix, pcp, dcp int) float64 {
	if pcp <= 1 || sched <= 0 {
		return 0
	}
	if pcpLocalTokens(sched, pcp, dcp) == sched {
		// Replicated: every rank attends the whole query, so this is the unsharded count.
		s := float64(sched)
		return 2 * 2 * (s*float64(prefix) + s*s*0.5)
	}
	numChunks := 2 * pcp
	chunk := (sched + numChunks - 1) / numChunks
	// Maximised over the ranks rather than read off one of them. WHICH rank is busiest
	// depends on where the ragged remainder falls -- enumerated over 40,000 random
	// (pcp, sched, prefix) draws it is rank 1 in 85% of cases, rank 0 in 14%, and some
	// other rank in the rest -- so assuming an index is wrong. The arithmetic below is
	// closed-form per rank and pcp is at most the tensor-parallel width, so this is a
	// handful of iterations rather than a walk over the query.
	var most float64
	for rank := 0; rank < pcp; rank++ {
		var pairs float64
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := min(idx*chunk, sched)
			hi := min(lo+chunk, sched)
			if n := hi - lo; n > 0 {
				// The keys each query in [lo, hi) reads: prefix + j + 1, summed in
				// closed form. The final -n/2 drops the diagonal's half-token so this
				// matches the unsharded term's CONTINUUM convention (s*c + s^2/2)
				// rather than exceeding it by s/2 -- the same correction
				// windowedCausalFLOPs makes, and for the same reason: without it an
				// inactive split would not reproduce the figure it replaces.
				pairs += float64(n)*float64(prefix+1) +
					float64(lo+hi-1)*float64(n)/2 - float64(n)/2
			}
		}
		if pairs > most {
			most = pairs
		}
	}
	return 2 * 2 * most
}

// dcpShardsKV reports whether decode-context parallelism shards THIS attention kind's
// cache.
//
// Per kind rather than per deployment, and vLLM states each case separately
// (vllm/v1/kv_cache_interface.py at v0.31.0):
//
//   - full attention, MLA and sparse MLA shard. AttentionSpec carries
//     `dcp_sharded: bool = True` (:488) and FullAttentionSpec.max_memory_usage_bytes
//     divides max_model_len by dcp_world_size (:578-583).
//   - a sliding window does NOT, and the engine refuses the combination outright:
//     SlidingWindowSpec.max_memory_usage_bytes asserts dcp == 1 with the message
//     "DCP not support sliding window" (:868-872). ChunkedLocalAttentionSpec likewise
//     divides by nothing.
//   - recurrent state does not, for a reason the engine also states: "Mamba state is
//     replicated across DCP/PCP ranks, never sharded" (:1097-1098). Nothing here has to
//     act on that, because a recurrent layer is priced through RecurrentStateBytes and
//     never reaches the KV term this function gates.
//
// The distinction is load-bearing rather than tidy: gpt-oss-120b ALTERNATES a 128-token
// swa layer kind with a full gqa one, so a deployment-wide divisor would shard half its
// stack against an assertion the engine would not have started under.
//
// An unrecognized kind returns false, which over-prices rather than under-prices. That is
// the same direction resolve.Emits takes for a condition it cannot answer: silently
// dropping a cost is the failure mode worth refusing.
func dcpShardsKV(kind model.AttentionKind) bool {
	switch kind {
	case model.AttentionGQA, model.AttentionMLA, model.AttentionSparseMLA:
		return true
	}
	return false
}

// dcpLocalTokens is how many KV tokens the SLOWEST decode-context-parallel rank holds,
// summed over these requests. It returns -1 when DCP shards nothing here, so a caller
// keeps the unsharded count -- the same sentinel contract selectedKVTokens uses.
//
// THREE PROPERTIES, each of which changes the answer by a large factor if dropped.
//
// IT DIVIDES TOKENS, NOT BYTES PER TOKEN. vLLM shards the cache by token position and
// leaves a page's size alone: max_num_blocks_per_req is
// `cdiv(max_len, block_size * kv_shard_count)` (kv_cache_interface.py:545-550) and
// page_size_bytes never sees dcp. Dividing a per-token byte figure instead would also
// divide PDTransferTime, which DCP does not shard, and would understate a paged capacity
// by up to dcp below block_size*dcp tokens -- 8x at 16 tokens and dcp 8, 2x at 64, exact
// from 128 up.
//
// THE SLOWEST RANK BINDS, NOT THE MEAN. The decode combine is a collective
// (cp_lse_ag_out_rs at vllm/v1/attention/ops/dcp.py:471), so every rank waits for the
// one holding the most tokens. Charging the mean would make a step cheaper than any rank
// can actually deliver.
//
// THERE IS NO REPLICATION FLOOR, which is exactly where this differs from the KV-head
// division next door. price.KVBytesPerToken floors at one head because a head is never
// split across ranks; tokens carry no such constraint and the shards sum to the whole
// context. That is also why DCP is the only axis that shards an MLA cache: a latent
// cache has one head, so no tensor-parallel width reduces it.
//
// THE STRIPING QUESTION, answered rather than left to omission. Tokens are striped across
// ranks in runs of `interleave` (cp_kv_cache_interleave_size, default 1), and the per-rank
// length is
//
//	base = (L / interleave / dcp) * interleave        // integer division
//	local(rank) = base + clip(L - base*dcp - rank*interleave, 0, interleave)
//
// (vllm/v1/attention/backends/utils.py:1143-1156). Rank 0 takes the largest clip, so the
// slowest rank holds base + min(L - base*dcp, interleave). Verified against the engine's
// own per-rank form over 18,000 (dcp, interleave, L) combinations: this equals the maximum
// across ranks exactly, and the shards sum to L exactly. So the interleave contributes a
// BOUNDED ADDITIVE term of at most one run, not a multiplicative one -- at interleave 32,
// dcp 8 and a 1,000-token context it is 128 against ceil(1000/8)=125, under 3%. Bounded is
// not small on a context short against the run: at interleave 64 a 100-token context puts
// 64 tokens on the slowest rank against ceil(100/8)=13, which is why the resolved interleave
// is passed in rather than assumed. At the default interleave of 1 the form reduces to
// ceil(L/dcp).
//
// PER REQUEST, not over the batch total, for the same reason selectedKVTokens is: the
// remainder term is per sequence, so sum-then-shard and shard-then-sum disagree whenever
// a context is not a multiple of interleave*dcp, which in a real batch is the normal case.
func dcpLocalTokens(contexts []int, dcp, interleave int) float64 {
	if dcp <= 1 {
		// Not sharded: one rank holds everything. -1 rather than the unsharded sum, so a
		// caller cannot confuse "no shard" with "a shard that happens to be the whole
		// context".
		return -1
	}
	if interleave < 1 {
		// The engine's own default. Guarded rather than trusted, because a zero here would
		// divide by zero below.
		interleave = 1
	}
	var total float64
	for _, ctx := range contexts {
		if ctx <= 0 {
			continue
		}
		base := ctx / interleave / dcp * interleave
		rest := ctx - base*dcp
		if rest > interleave {
			rest = interleave
		}
		local := base + rest
		if local < 1 {
			// A context shorter than the group still occupies one token on some rank.
			// Unreachable given the arithmetic above (rest is at least 1 when base is 0
			// and ctx is positive), asserted rather than assumed because a zero here
			// would make a decode read free.
			local = 1
		}
		total += float64(local)
	}
	return total
}

// dcpDecodeTokens is the KV token count a decode read charges for THIS layer, after both
// bounds that can narrow it: the top-k a sparse layer selects, and the shard a
// decode-context-parallel group holds. It returns -1 when neither applies, so a caller
// keeps the raw context sum -- the sentinel contract selectedKVTokens established.
//
// BOTH BOUNDS ARE APPLIED PER REQUEST, and the order is selection then sharding. Selection
// is a property of the sequence -- which positions this layer looks at -- and sharding a
// property of the layout: which of those positions live on this rank. So a rank reads its
// share of what the layer selected, not its share of the whole context.
//
// PER REQUEST RATHER THAN ON THE BATCH TOTAL, because the shard rounds. The engine keeps a
// per-request vector of local lengths for exactly this reason (dcp_local_seq_lens,
// vllm/v1/attention/backend.py:427-428, built per request at flash_attn.py:884-893 of
// v0.31.0), and sums it in the kernel. Scaling a batch total by one ratio instead agrees
// in the common case and does not in general: searched over 200,000 random (width, top-k,
// compression, context) draws, the largest disagreement is 4.3% -- at a heavily compressed
// sparse layer whose selected count is short enough that each request's rounding matters.
// Small, but it is the kind of error that compounds silently, so this computes the thing
// itself rather than an approximation of it.
func (k *Kernel) dcpDecodeTokens(l *price.PlannedLayer, kind model.AttentionKind,
	contexts []int) float64 {
	topk := sparseTopK(l)
	dcp := k.layout.DCP
	shards := dcp > 1 && dcpShardsKV(kind)
	if topk <= 0 && !shards {
		return -1
	}
	// The stripe as resolved: stated, pinned to the block size under NIXL, or the
	// engine's default of 1 (resolve.ResolveDecodeContext).
	interleave := k.decodeContext.Interleave
	var total float64
	for _, ctx := range contexts {
		if ctx <= 0 {
			continue
		}
		read := float64(ctx)
		if topk > 0 {
			read = selectedKVTokensFor(l, ctx, topk)
		}
		if shards {
			// Round before sharding: a rank holds whole cached positions, and the
			// selected count of a compressed layer is already a continuous expectation.
			read = dcpLocalTokens([]int{int(read + 0.5)}, dcp, interleave)
		}
		total += read
	}
	return total
}

// dcpDecodeCollectives prices one layer's decode-context-parallel collectives, returning
// the on-node and cross-node seconds. rows is the decode tokens in the step: one query row
// crosses per decode token (the MQA rows, num_mqa_tokens), and a prefill token takes no
// part in a decode combine.
//
// WHAT vLLM v0.31.0 LAUNCHES PER DECODE LAYER, by backend and by whether prefill-context
// parallelism is on. Three collectives at most, which is the "3 NCCL calls" its own
// documentation counts for ag_rs (vllm/config/parallel.py:371-380):
//
//	                    query gather            log-sum-exp gather   output combine
//	ag_rs, pcp off      all-gather over DCP     all-gather over DCP  reduce-scatter over DCP
//	a2a,   pcp off      all-gather over DCP     -- packed into one all-to-all over DCP --
//	ag_rs, pcp on       all-gather over TP,     all-gather over DCP  all-reduce over DCP
//	(latent only)       only when dcp = tp*pcp
//
//	query gather, pcp off  MLADCPManager._gather_query, dcp.py:1592-1596; flash_attn.py:1591
//	query gather, pcp on   deepseek_v32/attention.py:560-563 (none when dcp == pcp)
//	LSE gather             _cp_lse_common, dcp.py:458
//	ag_rs combine          cp_lse_ag_out_rs :493, or cp_lse_ag_out_ar :526 under PCP,
//	                       chosen by MLADCPManager._init_combine, dcp.py:1525-1531
//	a2a combine            dcp_a2a_lse_reduce, dcp.py:939-1010
//
// a2a with pcp on is refused at construction (pcp_manager.py:188-194), and so is pcp on a
// non-latent model ("MRV2 PCP currently supports MLA models only", pcp_manager.py:132-133)
// and pcp with dcp on anything but a DSA sparse-MLA layer (mla_attention.py:684-687), so
// none of those reaches here.
//
// THE PAYLOAD IS THE TENSOR THE COLLECTIVE MOVES, which is the convention every collective
// in this kernel prices against: an all-gather's output, a reduce-scatter's input, an
// all-reduce's buffer, an all-to-all's send buffer. A rank's tensor-parallel shard holds
// heads/tp query heads; the DCP query gather widens that to heads*dcp/tp, and under PCP the
// attention runs over heads/tp heads (dcp == pcp) or all heads (dcp == tp*pcp, after the
// tensor-parallel gather). The query crosses at the full head width -- for a latent layer
// kv_lora_rank + qk_rope_head_dim, which the catalog's d_h is -- and the output at the value
// width: kv_lora_rank for a latent layer (MLADCPManager's output_head_dim,
// mla_attention.py:697-708) and the head dimension otherwise. The a2a packs the fp32 LSE
// into two 16-bit lanes beside each output row (_dcp_a2a_lse_pack_dim, dcp.py:604-610).
//
// ACTIVATIONS CROSS AT 16 BITS. The query and the partial output are activations in the
// model's compute dtype, which stays bf16 under an fp8 or fp4 weight format: vLLM's
// quantized linears return out_dtype=x.dtype (vllm/model_executor/kernels/linear/scaled_mm/
// cutlass.py). The one exception is a backend that takes a quantized query alongside a
// quantized cache (query_dtype, mla_attention.py:690-696), which would move the query at
// one byte; this prices it at two, an overstatement of one of the three collectives.
//
// COVERAGE LIMITS, stated so they are not mistaken for completeness. A prefill row resuming
// on a DCP-sharded prefix needs that prefix's KV gathered back (MLA's chunked-context
// all-gather, mla_attention.py:3090-3104), and a full-attention DCP layer gathers the query
// of its context-prefill rows as well as its decode rows (flash_attn.py:1591). Neither is
// priced: this charges decode rows only.
func (k *Kernel) dcpDecodeCollectives(l *price.PlannedLayer, rows int) (onNode, crossNode float64) {
	const activationBytes = price.ActivationBytes
	tp := float64(max(k.layout.TP, 1))
	dcp := float64(k.layout.DCP)
	pcpOn := k.layout.PCP > 1
	heads := float64(l.AttnQHeads)
	queryWidth := float64(l.AttnHeadDim)
	outWidth := queryWidth
	if latentAttention(l.AttnKind) && l.AttnKVLoRARank > 0 {
		outWidth = float64(l.AttnKVLoRARank)
	}
	r := float64(rows)

	charge := func(key collKey, bytes float64) {
		if k.crossesNodes(key) {
			crossNode += k.collectiveSeconds(key, bytes*k.spanFor(key))
		} else {
			onNode += k.collectiveSeconds(key, bytes)
		}
	}

	// The heads each rank's attention runs over, after any query gather.
	attnHeads := heads / tp * dcp
	if pcpOn {
		attnHeads = heads / tp
		if k.layout.DCP > k.layout.PCP {
			// dcp == tp*pcp: the query is gathered over the tensor-parallel group first.
			attnHeads = heads
			charge(collKey{Op: model.OpAllGather, Group: price.GroupTP},
				r*attnHeads*queryWidth*activationBytes)
		}
	} else {
		charge(collKey{Op: model.OpAllGather, Group: price.GroupDCP},
			r*attnHeads*queryWidth*activationBytes)
	}

	if k.decodeContext.CommBackend == resolve.DCPAllToAll {
		const lsePack = 2 // an fp32 LSE in two 16-bit lanes
		charge(collKey{Op: model.OpAll2All, Group: price.GroupDCP},
			r*attnHeads*(outWidth+lsePack)*activationBytes)
		return onNode, crossNode
	}
	const lseBytes = 4 // fp32
	charge(collKey{Op: model.OpAllGather, Group: price.GroupDCP}, dcp*r*attnHeads*lseBytes)
	combine := model.OpReduceScatter
	if pcpOn {
		combine = model.OpAllReduce
	}
	charge(collKey{Op: combine, Group: price.GroupDCP}, r*attnHeads*outWidth*activationBytes)
	return onNode, crossNode
}

// pcpPrefillGathers prices one layer's prefill-context all-gathers, returning the on-node
// and cross-node seconds. localTokens is the prefill tokens THIS rank computed.
//
// ONE ALL-GATHER PER TENSOR, and a latent layer gathers two. _gather_prefill_cache_inputs
// launches one all_gather per tensor it is given (pcp.py:31-35), and the MLA cache write
// gives it kv_c_normed and k_pe separately (maybe_gather_mla_latent_cache_inputs,
// pcp.py:56-74; called before the cache update, mla_attention.py:781-800, and on a DSA layer
// deepseek_v32/attention.py:511-529). A layer with a
// sparse indexer gathers the indexer's key as well, in its own launch
// (maybe_gather_indexer_k, pcp.py:77-87, from sparse_attn_indexer.py:440). Each launch pays
// its own floor, which at a prefill chunk's payload is most of the cost.
//
// AT THE MODEL DTYPE, NOT THE CACHE'S. The gather runs before do_kv_cache_update quantizes
// into the cache (mla_attention.py:781-800; deepseek_v32/attention.py:511-529), so an fp8
// cache does not halve it: these are
// bf16 activations.
//
// SIZED AT THE GATHERED OUTPUT, pcp times what this rank contributes, which is the
// convention every all-gather in this kernel prices against (an all-gather's payload is
// the tensor it produces). An earlier form sized it at the rank's own contribution, which
// understated the bytes by the PCP width.
//
// PCP runs only where every layer is latent attention (resolveContextParallel), so the
// widths here are a latent layer's: kv_lora_rank for kv_c_normed and the remainder of the
// head width for k_pe. A latent node that states no kv_lora_rank is gathered as one tensor
// of its head width.
func (k *Kernel) pcpPrefillGathers(l *price.PlannedLayer, localTokens int) (onNode, crossNode float64) {
	const activationBytes = price.ActivationBytes
	rows := float64(k.layout.PCP) * float64(localTokens)
	key := collKey{Op: model.OpAllGather, Group: price.GroupPCP}
	gather := func(width float64) {
		bytes := rows * width * activationBytes
		if k.crossesNodes(key) {
			crossNode += k.collectiveSeconds(key, bytes*k.spanFor(key))
		} else {
			onNode += k.collectiveSeconds(key, bytes)
		}
	}
	if lora := l.AttnKVLoRARank; lora > 0 && lora < l.AttnHeadDim {
		gather(float64(lora))                 // kv_c_normed
		gather(float64(l.AttnHeadDim - lora)) // k_pe
	} else {
		gather(float64(l.AttnHeadDim))
	}
	for _, a := range l.Attentions {
		if !a.HoldsKV() && a.KVHeads > 0 && a.HeadDim > 0 {
			gather(float64(a.KVHeads * a.HeadDim)) // the indexer's key
		}
	}
	return onNode, crossNode
}

// moeDPFunnel is how many replica groups' tokens reach one rank's experts.
//
// Attention data parallelism replicates attention across DP ranks but SHARES the expert
// pool: each rank routes its own tokens and then every rank's tokens are gathered before
// the grouped GEMM, so the experts see dp times one rank's token count.
//
// Expert parallelism need not be on: vLLM v0.31.0 shares the MoE across data-parallel
// ranks either way -- with it off it shards every expert tensor-parallel over the dp x pcp
// x tp ranks (flatten_tp_across_dp_and_pcp, vllm/model_executor/layers/fused_moe/
// config.py:1090-1098, 1186-1200; "MoE layers will be sharded according to the product of
// the tensor, prefill-context, and data parallel sizes", vllm/config/parallel.py:134-136)
// and gathers every rank's tokens to it (all2all_utils.py:202-214). So the funnel is dp
// whenever there is an MoE group to gather into.
//
// COVERAGE LIMIT: prefill-context ranks are not counted here. A PCP rank's prefill tokens
// are its own share while its decode rows are replicated across the group, and how the MoE
// gathers that mixture is not modelled.
//
// Returns 1 rather than 0 when DP is unset, so a scenario that omits it prices as it did
// before this term existed.
func (k *Kernel) moeDPFunnel() float64 {
	if k.layout.MoEGroup() <= 1 || k.layout.DP <= 1 {
		return 1
	}
	return float64(k.layout.DP)
}

// hostPerStep returns the host cost of one step.
//
// The graph mode that runs decides how many launches a step makes, and it is chosen per
// batch: under FULL_AND_PIECEWISE -- vLLM v0.31.0's default -- and FULL_DECODE_ONLY a uniform
// decode batch replays one full graph, and every other batch runs piecewise or with no graph
// respectively. Those are each mode's decode and mixed halves (CUDAGraphMode.decode_mode
// and mixed_mode, vllm/config/compilation.py:62-69), which the default Model Runner V2
// builds its graph candidates from (vllm/v1/worker/gpu/cudagraph_utils.py:250-330).
//
//	NONE       one launch per layer's kernels, all eager.
//	PIECEWISE  the graph is split at every attention op, because attention takes a
//	           varying-shape KV cache that a captured graph cannot hold. So a step
//	           replays one segment per split point and runs attention eagerly between
//	           them: 73 launches on a 72-layer model, not one.
//	FULL       one replay for the whole step, which is what a single per-step cost
//	           describes.
//
// Charging one replay for a PIECEWISE step understates its host cost by the layer count.
// On a 72-layer model at the conventional 5-10 microsecond replay cost that is half a
// millisecond against seven microseconds — which matters at decode, where the whole step
// is a few milliseconds.
//
// The split points are vLLM's `_attention_ops` (vllm/config/compilation.py:764-782), and
// this counts one per layer, so the segment count is the layer count plus one.
//
// TWO KNOWN DIVERGENCES, kept because the host coefficients were fitted with this
// structure and changing either is a refit rather than a correction:
//
//   - A sparse-MLA layer splits twice: at its indexer (vllm::sparse_attn_indexer) as well
//     as its attention, so its stack has about twice the segments counted here.
//   - A batch of more tokens than max_cudagraph_capture_size runs with no graph at all,
//     whatever the mode: Model Runner V2 finds no captured candidate and dispatches NONE
//     (vllm/v1/worker/gpu/cudagraph_utils.py:499-529), as V1 does
//     (vllm/v1/cudagraph_dispatcher.py:270-279). A large prefill step is priced with
//     replays it does not make.
func (k *Kernel) hostPerStep(b kernel.Batch) time.Duration {
	mode := k.graphOther
	// An empty step runs no forward; it is charged as the decode loop's idle step.
	if len(b.Reqs) == 0 || b.UniformDecode(k.uniformDecodeWidth) {
		mode = k.graphDecode
	}
	switch mode {
	case graphModeFull:
		return k.replayPerStep
	case graphModePiecewise:
		segments := float64(k.plan.TotalLayers + 1)
		return time.Duration(float64(k.replayPerStep) * segments)
	default:
		return time.Duration(float64(k.launchPerLayer) * float64(k.plan.TotalLayers))
	}
}

// launchSeconds returns the per-kernel dispatch cost a step pays regardless of how much
// work each kernel does.
//
// This is the term that dominates a single-request decode step, and an earlier version of
// this model omitted it entirely. Two published runs of one 230B MoE model made the case:
// ITL at one concurrent request was 6.02 ms on H200 at 8k context and 6.13 ms on H100 at
// 707 tokens. Those parts differ in memory bandwidth by 1.43x and the contexts by 10x, and
// the measurement barely moved — so the step is dominated by something independent of
// both, which is dispatch. The residual against the rest of the model was 43.7 and 46.0
// microseconds per layer respectively, agreeing to 5% across that variation. Those reports
// are no longer in the corpus; the reasoning is recorded here because it is why the term
// exists, and the term is still checked by TestHostTermIsChargedOnEveryStep.
//
// Charged per KERNEL rather than per layer, because that is the quantity a deeper or
// shallower model scales with: the plan counts each layer's launches from its surviving
// graph nodes. Over roughly 20 launches per MoE layer the implied per-launch cost is about
// 2.3 microseconds, which is the conventional captured-graph dispatch figure the registry
// already cites for a single launch.
//
// It does NOT overlap with device work, and that is the point rather than a simplification:
// a dispatch that has not happened cannot have its kernel running. At large batch the
// device work per kernel dwarfs it and it vanishes into the noise, which is why a model
// fitted only at high concurrency would miss it.
func (k *Kernel) launchSeconds() float64 {
	return k.launchPerKernel.Seconds() * float64(k.plan.TotalKernels)
}

// graphMode is how much of a step a captured graph covers.
type graphMode int

const (
	// graphModeEager captures nothing: every layer's kernels are launched.
	graphModeEager graphMode = iota
	// graphModePiecewise captures between attention boundaries, so a step replays one
	// segment per layer and runs attention eagerly.
	graphModePiecewise
	// graphModeFull captures the whole step as one graph.
	graphModeFull
)

// TierTime prices one transfer between GPU and an offload tier.
func (k *Kernel) TierTime(tier string, dir kernel.Direction, bytes int64,
	queueDepth int) time.Duration {
	dev, ok := k.tiers[tier]
	if !ok {
		// An unknown tier is not a free transfer. Returning zero would make an offload
		// hit look costless, which is the opposite of what an unmodelled tier means.
		return time.Duration(math.MaxInt64)
	}
	rate := dev.ReadBandwidthMBs
	if dir == kernel.DirectionToTier {
		rate = dev.WriteBandwidthMBs
	}
	rate *= 1e6 // MB/s to bytes per second
	// A CPU tier crosses the host link as well as host memory, and the slower of the two
	// binds. Which one that is depends on the pairing rather than on either alone.
	if k.hostBytesPerSecond > 0 && rate > k.hostBytesPerSecond {
		rate = k.hostBytesPerSecond
	}
	// Storage bandwidth saturates under concurrency. The coefficient behind this is an
	// assumption, so the effect is linear in queue depth beyond the first request rather
	// than a fitted curve.
	if queueDepth > 1 {
		rate /= float64(queueDepth)
	}
	return seconds(price.FloorAndRate(float64(bytes), rate, dev.BaseLatencyUs*1e-6))
}

// PDTransferTime prices moving one request's KV between pools.
//
// Decode-context parallelism deliberately does NOT divide this, and the omission is the
// reason kvBytesPerToken is left as a per-token figure rather than being sharded once at
// construction. A prefill pool holds the whole cache for a request -- DCP shards the
// DECODE cache, and vLLM's own wording is "Number of ranks that shard the decode KV
// cache" (vllm/config/parallel.py:359-362 at v0.31.0) -- so the bytes that cross between
// pools are the whole request's, whatever the decode pool's dcp width is. A divisor here
// would make every PD transfer exactly dcp times too cheap.
func (k *Kernel) PDTransferTime(tokens int, from, to kernel.Placement) time.Duration {
	if tokens <= 0 {
		return 0
	}
	bytes := float64(tokens) * k.kvBytesPerToken
	rate := k.nicBytesPerSecond
	if from.Node == to.Node {
		// Same node: the transfer stays on the fast link.
		rate = k.nvlinkBytesPerSecond
	} else if k.fabric.RackIsOneDomain && from.Rack == to.Rack {
		// Multi-node NVLink makes a rack one domain, so an in-rack transfer moves at
		// near on-node speed.
		rate = k.nvlinkBytesPerSecond
	}
	return seconds(price.FloorAndRate(bytes, rate,
		k.collectiveFloors[collKey{Op: model.OpAll2All, Group: price.GroupExpert}].Seconds()))
}

// AdmissionOverhead returns host time before a request can be scheduled.
func (k *Kernel) AdmissionOverhead(promptTokens int) time.Duration {
	if promptTokens <= 0 {
		return 0
	}
	return k.admissionPerRequest +
		time.Duration(float64(k.admissionPerToken)*float64(promptTokens))
}

// OutputTokenOverhead returns host time per emitted token.
func (k *Kernel) OutputTokenOverhead() time.Duration { return k.outputTokenCost }

// CompletionOverhead returns fixed host time at request completion.
func (k *Kernel) CompletionOverhead() time.Duration { return k.completionCost }

// Provenance reports every coefficient this kernel used and its evidence.
func (k *Kernel) Provenance() []kernel.CoefficientOrigin { return k.origins }

// Resolved reports the configuration after resolution, including overridden requests.
func (k *Kernel) Resolved() kernel.Resolution { return k.resolution }

// Deployment returns the pool this kernel prices, as the document stated it.
//
// The request, not the resolution: a caller wanting a width or a backend reads Resolved,
// and reads this for the settings resolution does not touch -- the admission settings a
// scheduler sizes itself from. It is the pool New was given at PoolIndex, so a caller
// holding several kernels of one disaggregated deployment needs no index of its own to
// know which pool each prices.
//
// A deep copy, so a caller writing through one of the engine block's pointer fields
// changes its copy and not the kernel. blis-schemas permits returning a shallow copy and
// asking callers not to write; this is not a hot path, and a promise the type enforces is
// worth more than one a reader has to keep.
func (k *Kernel) Deployment() deployment.Pool { return clonePool(k.pool) }

// DecodeContextParallelWidth returns how many ranks shard the decode KV cache.
//
// A consumer sizing its own KV budget needs it and cannot derive it from anything else
// this kernel exposes: it is the only axis that shards a LATENT cache, because
// KVBytesPerToken floors at one KV head and a latent cache has exactly one, so no
// tensor-parallel width reduces it. A simulator that read Resolution's TensorParallelWidth alone would
// size an MLA deployment's cache as if DCP did nothing.
//
// From the resolved layout rather than the deployment document: a Parallelism states a
// request and resolution settles it.
//
// A method on this type rather than a field of kernel.Resolution only because blis-schemas
// v0.2.2's Resolution carries the tensor-, data- and expert-parallel widths and not the two
// context-parallel ones. The argument that put those three in Resolution -- a consumer
// that can read one width and not the others has to re-derive the rest -- applies to these
// two as well, so this method and its companion are interim, pending the same fields
// upstream.
func (k *Kernel) DecodeContextParallelWidth() int { return max(k.layout.DCP, 1) }

// PrefillContextParallelWidth returns how many ranks split a prefill sequence.
//
// The companion to DecodeContextParallelWidth, and the two are genuinely independent: PCP
// splits prefill computation and expands the process world size while leaving the KV cache
// replicated, where DCP shards the cache without expanding the world size -- it reuses the
// tensor-parallel ranks when pcp is 1, and spans the PCP axis or the whole TP x PCP block
// otherwise (vllm/config/parallel.py:359-362 at v0.31.0). A consumer
// sizing a deployment needs both, and neither can be derived from the other.
//
// Interim for the same reason as DecodeContextParallelWidth.
func (k *Kernel) PrefillContextParallelWidth() int { return max(k.layout.PCP, 1) }

// Resource indices for the hot path's fixed array. A map allocation per step would show
// up in a simulator calling this millions of times, so the accumulation is positional and
// converted to the published map once at the end.
const (
	idxSM = iota
	idxHBM
	idxNVLink
	idxNIC
	idxHost
	numResources
)

// summarize fills per from the positional totals and returns the serialized total and the
// busiest resource. Separated from StepTime so both the allocating and the caller-supplied
// forms share one definition of what the summary means.
func summarize(totals *[numResources]float64,
	per map[kernel.Resource]time.Duration) (time.Duration, kernel.Resource) {
	var noOverlap time.Duration
	var largest float64
	bottleneck := kernel.ResourceSM
	for i, elapsed := range totals {
		if elapsed <= 0 {
			continue
		}
		d := seconds(elapsed)
		if per != nil {
			per[resourceForIndex(i)] = d
		}
		noOverlap += d
		if elapsed > largest {
			largest, bottleneck = elapsed, resourceForIndex(i)
		}
	}
	return noOverlap, bottleneck
}

func resourceForIndex(i int) kernel.Resource {
	switch i {
	case idxSM:
		return kernel.ResourceSM
	case idxHBM:
		return kernel.ResourceHBM
	case idxNVLink:
		return kernel.ResourceNVLink
	case idxNIC:
		return kernel.ResourceNIC
	default:
		return kernel.ResourceHost
	}
}

func seconds(s float64) time.Duration {
	if math.IsNaN(s) || s < 0 {
		return 0
	}
	if math.IsInf(s, 1) {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(s * float64(time.Second))
}

func mustPositive(name string, v float64) error {
	if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%s resolved to %v, which cannot price anything", name, v)
	}
	return nil
}
