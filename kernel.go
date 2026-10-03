// Package latencykernel implements the blis-schemas Kernel interface: it prices GPU work
// for one resolved deployment.
//
// Construction resolves everything — catalog facts, registry coefficients, the engine
// settings a layout overrides, and the model graph flattened to a per-kind plan. After
// that every method is a pure function of its arguments, so a discrete-event simulator can
// call StepTime at any simulated instant, from any goroutine, and get the same answer.
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
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Kernel prices one deployment. Obtain one from New; the zero value is not usable.
type Kernel struct {
	// Resolved configuration, frozen at construction.
	layout resolve.Layout
	fabric resolve.Fabric
	plan   *price.Plan
	pool   scenario.Pool
	chip   hardware.Chip

	// servedDType is the weight format the linear layers actually run in, after the
	// scenario's --quantization is applied over the checkpoint's own format.
	servedDType model.DType

	// Constants lifted out of the coefficient set once, so the hot path reads fields
	// rather than hashing strings.
	hbmBytesPerSecond     float64
	computeFLOPsPerSecond float64
	epsMax, mHalf         float64
	nvlinkBytesPerSecond  float64
	nicBytesPerSecond     float64
	hostBytesPerSecond    float64
	moeImbalance          float64

	// Per-operation collective floors and peak rates, resolved once. Keyed by op
	// because an all-reduce floor and an all-to-all floor differ by more than 2x on
	// the same part, and by rank count inside the lookup because a ring-shaped
	// collective's floor grows with group width where a shuffle's barely does.
	collectiveFloors      map[model.Op]time.Duration
	collectiveRates       map[model.Op]float64
	collectiveTransitions map[model.Op]float64

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

	// Memory reservations NVIDIA's own system descriptor states per part.
	commBytes      int64
	workspaceBytes int64

	// Per-rank expert counts, derived once.
	expertsPerRank  float64
	expertImbalance float64

	// totalExperts is the model's physical expert count, which the expected-coverage
	// term needs alongside the local count.
	totalExperts int
	// expertTensorShards is how many ranks one expert's weights are split across: the
	// tensor-parallel width when expert parallelism is off, and 1 when it is on, because
	// an expert-parallel rank owns whole experts.
	expertTensorShards float64

	// tp and localExpertShare are the per-rank divisors, held as floats so the hot path
	// multiplies rather than converting. localExpertShare is the fraction of the model's
	// experts one rank holds, which is what divides routed FLOPs.
	tp               float64
	localExpertShare float64

	// KV geometry, derived once from the graph and the cache dtype.
	kvBytesPerToken float64
	blockSize       int

	// Host overheads.
	admissionPerToken time.Duration
	outputTokenCost   time.Duration
	completionCost    time.Duration
	launchPerLayer    time.Duration
	launchPerKernel   time.Duration
	replayPerStep     time.Duration
	graphCaptured     bool
	graphMode         graphMode

	// Fixed occupancy, computed once.
	fixed kernel.MemoryBreakdown

	// Provenance, assembled once.
	origins    []kernel.CoefficientOrigin
	resolution kernel.Resolution

	// tiers maps a tier name to its device facts, for TierTime.
	tiers map[string]hardware.StorageDevice
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
	mode := price.RecurrentCacheMode(k.pool.Engine.MambaCacheMode)
	if mode == "" {
		mode = price.RecurrentCacheNone
	}
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
	total := price.PagedBytes(tokens, k.blockSize, k.kvBytesPerToken)

	// Under the all-positions recurrent cache mode the state is proportional to the
	// context bound, so it is a variable cost rather than a fixed one.
	mode := price.RecurrentCacheMode(k.pool.Engine.MambaCacheMode)
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
		host := k.hostPerStep()
		per := into
		if per == nil {
			per = make(map[kernel.Resource]time.Duration, 1)
		}
		per[kernel.ResourceHost] = host
		return kernel.StepEstimate{
			// Both edges are the host cost on an empty step, so Expected is too: there is
			// no device work for them to disagree about.
			Overlap: host, NoOverlap: host, Expected: host,
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
	// than the engine's decode threshold — the cost model's batch-region classifier — which
	// is the same test vLLM's own kernel selection makes.
	//
	// causalFLOPs halves the naive count because attention attends only to earlier
	// positions. decodeKVTokens is the context those requests read.
	var causalFLOPs, decodeKVTokens float64
	var prefillRequests, decodeRequests int
	for i := range b.Reqs {
		r := &b.Reqs[i]
		ctx := r.Computed + r.Scheduled
		if ctx <= 0 {
			continue
		}
		if r.Scheduled > b.DecodeThreshold {
			prefillRequests++
			causalFLOPs += 2 * 2 * float64(r.Scheduled) * float64(ctx) * 0.5
			continue
		}
		decodeRequests++
		decodeKVTokens += float64(ctx)
	}

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

		flops += tokensF * l.DenseFLOPsPerToken / k.tp
		weightBytes += l.DenseWeightBytes / k.tp

		if l.ExpertFLOPsPerTokenPerExpert > 0 {
			// A rank computes only the tokens routed to the experts it holds.
			routedPerRank := tokensF * float64(l.TopK) * k.localExpertShare
			routedFLOPs += routedPerRank * l.ExpertFLOPsPerTokenPerExpert * k.moeImbalance
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
			routedWeightBytes += l.ExpertWeightBytesPerExpert / k.expertTensorShards *
				price.ExpertsTouched(tokens, k.totalExperts, l.TopK, k.expertsPerRank)
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
		// A step holding both kinds of request pays both, because the engine launches a
		// kernel for each. Decode is charged to HBM and prefill to SM, which is what each
		// regime binds on.
		if l.AttnQHeads > 0 {
			// This layer kind's own decode terms, falling back to the part-wide pair.
			floor, rate := k.attentionFloor, k.attentionRate
			if fr, ok := k.attentionByKind[l.AttnKind]; ok {
				floor, rate = fr.floor, fr.rate
			}
			if decodeRequests > 0 && rate > 0 {
				attnSeconds += floor.Seconds() +
					decodeKVTokens*k.kvBytesPerToken/float64(k.plan.TotalLayers)/
						rate
			} else if decodeRequests > 0 {
				// No measured decode form for this part: the KV read is charged below and
				// the floor is lost. Provenance says the entries are missing.
				kvFallback = true
			}
			if prefillRequests > 0 && k.attentionPrefillScale > 0 {
				scale := price.Efficiency(tokensF, k.epsMax, k.mHalf) *
					k.attentionPrefillScale
				attnSMSeconds += k.attentionPrefillFloor.Seconds() +
					causalFLOPs*float64(l.AttnQHeads)*float64(l.AttnHeadDim)/k.tp/
						(k.computeFLOPsPerSecond*scale)*smDerate
			} else if prefillRequests > 0 {
				// No measured prefill form: fall back to the ramp unmodified, which the
				// sweeps say understates by over ten times.
				flops += causalFLOPs * float64(l.AttnQHeads) *
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
			kvBytes = decodeKVTokens * k.kvBytesPerToken / float64(k.plan.TotalLayers)
		}

		// The efficiency ramp is evaluated at the STEP's token count for every term,
		// including the grouped GEMM, and that is a finding rather than a simplification.
		//
		// The cost model's §2.1 argues the grouped GEMM should see a smaller argument — the
		// rows routed to one expert rather than the whole batch — and the reasoning is
		// sound for a model like Granite-5, which routes to 8 of 224 experts. Scored
		// against four published deployments the per-expert argument is worse overall: it
		// improves Granite-5 on H200 (31% to 18% MAPE) and Kimi-K3 (64% to 41%) and ruins
		// Nemotron-3-Ultra (87% to 158%), which routes each token to 22 of 512 experts. The
		// aggregate goes from 48% to 56%.
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
			recurrentSeconds + attnSMSeconds
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
			elapsed := 0.0
			if k.crossesNodes(c.Op) {
				elapsed = k.collectiveSeconds(c.Op, bytes*k.spanFor(c.Op))
				crossNode += elapsed
			} else {
				elapsed = k.collectiveSeconds(c.Op, bytes)
				onNode += elapsed
			}
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
	host := k.hostPerStep() + seconds(k.launchSeconds())
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
	return kernel.StepEstimate{
		Overlap: overlap, NoOverlap: noOverlap,
		// Expected is NoOverlap on measured evidence, not preference. Over 219 points of
		// NVIDIA's FPM whole-forward dataset -- two models, two parts, five parallelism
		// topologies -- Overlap's SIGNED error is -13.45% against NoOverlap's -3.44%, and
		// NoOverlap is closer on 158 of them. A one-sided error of that size is a missing
		// term rather than scatter, and it matches the -13.10% deficit this kernel shows
		// end to end on an independent serving corpus. The physical reason is PIECEWISE
		// cudagraph mode: attention runs eagerly between captured segments, so per-layer
		// overlap is structurally limited.
		//
		// A concentration-aware choice belongs here eventually -- where one resource holds
		// most of a step, per-stage max IS right, and the single dissenting cell in that
		// evidence is a whole model on two GPUs. Five cells is enough to reject the
		// optimistic edge as a universal default and not enough to fit a blend, so this
		// stays a constant until there is more.
		Expected:   noOverlap,
		Bottleneck: bottleneck, PerResource: per,
	}
}

// collectiveTime prices one collective's bytes against its own measured floor and
// rate. Both are per operation: charging an all-to-all the all-reduce floor would
// overstate it by 2.2x on H200, and the rates differ by a similar factor.
func (k *Kernel) collectiveSeconds(op model.Op, bytes float64) float64 {
	peak, ok := k.collectiveRates[op]
	if !ok || peak <= 0 {
		// Every op the plan can emit is resolved at construction, so this is
		// unreachable for a kernel New returned. Returning infinity rather than zero
		// keeps an unpriced collective visible if that ever changes.
		return math.Inf(1)
	}
	return price.CollectiveTime(bytes, k.collectiveTransitions[op], peak,
		k.collectiveFloors[op].Seconds())
}

// collectiveTime prices one invocation, for callers outside the step loop.
func (k *Kernel) collectiveTime(op model.Op, bytes float64) time.Duration {
	return seconds(k.collectiveSeconds(op, bytes))
}

// crossesNodes reports whether a collective's own group spans more than one node.
//
// Per collective rather than per deployment: a tensor-parallel group of 8 fits inside an
// 8-GPU node however wide expert parallelism is, and an expert-parallel group of 72 does
// not. Using the deployment's widest group for both would move every reduction onto the
// NIC in a wide-EP layout, where in fact it never leaves the node.
func (k *Kernel) crossesNodes(op model.Op) bool {
	if k.layout.GPUsPerNode <= 0 {
		return false
	}
	return k.groupSize(op) > k.layout.GPUsPerNode
}

// groupSize returns the rank count a collective's group spans.
func (k *Kernel) groupSize(op model.Op) int {
	if op == model.OpAll2All {
		return k.layout.ExpertWidth
	}
	return k.layout.TP
}

// spanFor returns the cross-node scaling a collective pays. A ring reduces as it travels
// and so crosses only a fraction of the fabric; a routed all-to-all must reach every peer.
func (k *Kernel) spanFor(op model.Op) float64 {
	ratio := k.fabric.Ratio()
	group := k.layout.ExpertWidth
	if op == model.OpAllReduce || op == model.OpAllGather || op == model.OpReduceScatter {
		group = k.layout.TP
	}
	if op == model.OpAll2All && k.routedAll2All() {
		return price.All2AllSpan(group, k.layout.GPUsPerNode, ratio)
	}
	return price.RingSpan(group, k.layout.GPUsPerNode, ratio)
}

// routedAll2All reports whether the resolved MoE backend moves top-k-selected bytes point
// to point, rather than dense per-token bytes in two ring phases. The distinction changes
// both the volume and the cross-node scaling, and it is a property of the backend rather
// than of the primitive's name.
func (k *Kernel) routedAll2All() bool {
	switch k.pool.Engine.All2AllBackend {
	case "", "naive", "allgather_reducescatter":
		return false
	}
	return true
}

// hostPerStep returns the host cost of one step.
//
// The graph mode decides how many launches a step makes, and PIECEWISE — vLLM's default
// — is not one launch.
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
// The split points are vLLM's `_attention_ops` (`vllm/config/compilation.py`), which fire
// once per attention or recurrent-mixer layer, so the segment count is the layer count
// plus one.
func (k *Kernel) hostPerStep() time.Duration {
	switch k.graphMode {
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
// this model omitted it entirely. Two published Granite-5 runs make the case: ITL at one
// concurrent request is 6.02 ms on H200 at 8k context and 6.13 ms on H100 at 707 tokens.
// Those parts differ in memory bandwidth by 1.43x and the contexts by 10x, and the
// measurement barely moves — so the step is dominated by something independent of both,
// which is dispatch. The residual against the rest of the model is 43.7 and 46.0
// microseconds per layer respectively, agreeing to 5% across that variation.
//
// Charged per KERNEL rather than per layer, because that is the quantity a deeper or
// shallower model scales with: the plan counts each layer's launches from its surviving
// graph nodes. Over Granite-5's roughly 20 launches per MoE layer the implied per-launch
// cost is about 2.3 microseconds, which is the conventional captured-graph dispatch figure
// the registry already cites for a single launch.
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
		k.collectiveFloors[model.OpAll2All].Seconds()))
}

// AdmissionOverhead returns host time before a request can be scheduled.
func (k *Kernel) AdmissionOverhead(promptTokens int) time.Duration {
	if promptTokens <= 0 {
		return 0
	}
	return time.Duration(float64(k.admissionPerToken) * float64(promptTokens))
}

// OutputTokenOverhead returns host time per emitted token.
func (k *Kernel) OutputTokenOverhead() time.Duration { return k.outputTokenCost }

// CompletionOverhead returns fixed host time at request completion.
func (k *Kernel) CompletionOverhead() time.Duration { return k.completionCost }

// Provenance reports every coefficient this kernel used and its evidence.
func (k *Kernel) Provenance() []kernel.CoefficientOrigin { return k.origins }

// Resolved reports the configuration after resolution, including overridden requests.
func (k *Kernel) Resolved() kernel.Resolution { return k.resolution }

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
