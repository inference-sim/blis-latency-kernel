package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// A sparse-MLA layer does not read its whole context, and these pin that behaviourally.
//
// Before this, `sparse_mla` fell through to the full-attention byte count, so every layer
// of deepseek-v4-pro was priced as if it read the entire KV cache. That is wrong by up to
// 128x at long context, and it is not a small constant error: fitting a rate against the
// full-context reading forces it to 1.00 of datasheet peak -- physically impossible, which
// is how the wrong byte count announces itself (blis-registry
// scripts/fit_attention_sparse_mla.py reports 1.698x error clamped against 1.301x
// interior for the bounded form).
//
// Measured on h200 (AISimulate dsv4_hca_attn_module_perf, FLASHMLA_SPARSE_DSV4): a batch-1
// decode costs 9.8-13.7us flat from a 0 to a 16,384-token context and only 30.6us at
// 1,048,575. End-to-end on the InferenceX corpus the bound moved deepseek-v4-pro's TPOT
// error from 10.42% to 8.72%, against AISimulate's 12.82%.

// The headline property: step time must grow far more slowly in context for a sparse layer
// than the context grows. A model whose byte count is the full context would scale roughly
// linearly once past the floor.
func TestSparseMLADecodeDoesNotScaleWithTheWholeContext(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b200-fp4-vllm-tp8.yaml")

	var sparse *price.PlannedLayer
	for i := range k.plan.Layers {
		if k.plan.Layers[i].AttnKind == model.AttentionSparseMLA {
			sparse = &k.plan.Layers[i]
			break
		}
	}
	if sparse == nil {
		t.Skip("this fixture has no sparse_mla layer")
	}
	if sparse.AttnIndexTopK <= 0 && sparse.AttnWindow <= 0 {
		t.Fatalf("sparse layer carries neither index_topk nor window, so the plan is "+
			"dropping the geometry the catalog states: %+v", sparse)
	}

	at := func(ctx int) int64 {
		b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
			{Scheduled: 1, Computed: ctx, PromptLen: ctx},
		}}
		return k.StepTime(b).NoOverlap.Microseconds()
	}

	// A 64x context increase. A full-context byte count would push the attention term up
	// by about 64x; a bounded one leaves it nearly flat, so the step is dominated by the
	// terms that do not depend on context.
	short, long := at(16384), at(1048576)
	if short <= 0 || long <= 0 {
		t.Fatalf("both step times must be positive, got %d and %d", short, long)
	}
	// 1.8x, and the threshold is MEASURED rather than chosen for looseness. On this
	// deployment the bounded step moves 1.301x across that 64x context increase and the
	// unbounded one moves 2.330x, so 1.8 sits between them with room either side.
	//
	// Neither figure is near 64x because a step is not all attention: the GEMM and MoE
	// terms dominate here and are untouched by this bound, which compresses the whole
	// range. An earlier draft of this test used 8x on the reasoning that a full-context
	// read "would be near 64x" -- it passed under a mutation that ignored the bound
	// entirely, because the attention term was never 64x of the step to begin with. The
	// lesson is that a threshold argued from the term has to be checked against the
	// STEP, which is what the test actually measures.
	const maxRatio = 1.8
	if ratio := float64(long) / float64(short); ratio > maxRatio {
		t.Errorf("a 64x context increase moved the step %.3fx (%d -> %d us), above the "+
			"%.1fx bound; a sparse layer's read is capped at its selected top-k plus a "+
			"compressed remainder, so the step must grow far sub-linearly in context. "+
			"Measured: %.3fx with the bound, %.3fx without it",
			ratio, short, long, maxRatio, 1.301, 2.330)
	}
}

// The bound must be applied PER REQUEST, not to the batch-wide context total, because
// min(sum) != sum(min): a batch holding requests on both sides of the top-k gets both
// wrong if the cap is applied to the aggregate.
//
// Asserted on selectedKVTokens directly rather than through StepTime, and that choice is
// the finding. Through StepTime the distinction is UNOBSERVABLE on this model: an
// 8,600 us step is dominated by the GEMM and MoE terms, so a few hundred selected tokens
// across 60 layers does not move the microsecond-resolution total at all. Every shape pair
// tried -- 4x1024 against 4093+3x1, 4x128 against 509+3x1, 4x512 against 2045+3x1, at
// equal totals and equal request counts -- priced IDENTICALLY both with the correct
// per-request bound and with a deliberately broken batch-total one.
//
// So an end-to-end assertion here would be a test that cannot fail, which is worse than no
// test: it would report the property as protected while a batch-total regression shipped.
// The arithmetic is where the difference lives, so that is where it is pinned.
//
// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. This pins the FUNCTION, not the call
// site. A mutation replacing the pricer's `decodeContexts` with a one-element slice
// holding the batch total still passes every test in this file, because the resulting
// price difference is below microsecond resolution on every fixture here. Closing that
// would need either a fixture whose step is attention-dominated or an injection seam in
// the pricer, and neither exists today -- so the gap is recorded here instead of being
// hidden behind an assertion that happens to pass.
func TestSparseMLABoundsEachRequestNotTheBatchTotal(t *testing.T) {
	// csa4_moe's geometry: topk 1024, compressed remainder at 1/4.
	l := &price.PlannedLayer{
		AttnKind: model.AttentionSparseMLA, AttnIndexTopK: 1024, AttnCompressRatio: 4,
	}

	// Four requests of 1,024 tokens: every one sits exactly at the top-k, so nothing is
	// compressed and the total is 4 x 1024.
	perRequest := selectedKVTokens(l, []int{1024, 1024, 1024, 1024})
	if want := 4096.0; perRequest != want {
		t.Errorf("4 requests at the top-k selected %v tokens, want %v", perRequest, want)
	}

	// The same 4,096 tokens of context in ONE request: 1,024 selected plus 3,072/4
	// compressed = 1,792. Applying the cap to the batch total would give this same answer
	// for the case above, which is how the two differ.
	concentrated := selectedKVTokens(l, []int{4096})
	if want := 1024.0 + 3072.0/4.0; concentrated != want {
		t.Errorf("one request of 4,096 selected %v tokens, want %v", concentrated, want)
	}
	if perRequest == concentrated {
		t.Errorf("4x1,024 and 1x4,096 tokens of context both selected %v; the cap is "+
			"being applied to the batch total rather than per request", perRequest)
	}

	// And the direction matters: spreading the same context over more requests reads MORE,
	// because each request pays its own uncompressed top-k.
	if perRequest <= concentrated {
		t.Errorf("spreading context over 4 requests selected %v, not more than the %v one "+
			"request reads; each request pays its own top-k at full resolution",
			perRequest, concentrated)
	}
}

// A layer that is not sparse, or that states no bound, must report -1 so the caller keeps
// the full-context read. Returning 0 instead would make a non-sparse layer's attention
// free, which is silently cheap rather than visibly wrong.
func TestSelectedKVTokensRefusesWhatItCannotBound(t *testing.T) {
	for _, c := range []struct {
		name string
		l    price.PlannedLayer
	}{
		{"full attention", price.PlannedLayer{AttnKind: model.AttentionGQA}},
		{"dense MLA", price.PlannedLayer{AttnKind: model.AttentionMLA}},
		{"sparse with neither topk nor window", price.PlannedLayer{
			AttnKind: model.AttentionSparseMLA}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := selectedKVTokens(&c.l, []int{65536}); got != -1 {
				t.Errorf("got %v, want -1 so the caller charges the whole context", got)
			}
		})
	}

	// A window with no top-k IS a bound: that is how blis-catalog states csa128_moe.
	l := price.PlannedLayer{
		AttnKind: model.AttentionSparseMLA, AttnWindow: 128, AttnCompressRatio: 128,
	}
	if got := selectedKVTokens(&l, []int{65536}); got != 128.0+(65536.0-128.0)/128.0 {
		t.Errorf("a window-selected sparse layer bounded to %v, want %v",
			got, 128.0+(65536.0-128.0)/128.0)
	}
}

// Removing the geometry must change the price. This is the structural check the MLA
// dispatch test makes for its coefficient, in the form this change needs: if the plan
// stopped carrying index_topk and compress_ratio, the layer would silently revert to the
// full-context read while every other test still passed.
func TestSparseMLAGeometryReachesThePricer(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b200-fp4-vllm-tp8.yaml")

	batch := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 262144, PromptLen: 262144},
	}}
	with := k.StepTime(batch).NoOverlap

	// Strip the bound from every sparse layer and re-price.
	stripped := 0
	for i := range k.plan.Layers {
		l := &k.plan.Layers[i]
		if l.AttnKind != model.AttentionSparseMLA {
			continue
		}
		l.AttnIndexTopK, l.AttnWindow, l.AttnCompressRatio = 0, 0, 0
		stripped++
	}
	if stripped == 0 {
		t.Skip("this fixture has no sparse_mla layer")
	}
	without := k.StepTime(batch).NoOverlap
	if without <= with {
		t.Errorf("stripping the sparse geometry from %d layers did not RAISE the step "+
			"(%v with the bound, %v without); the unbounded read is the whole context and "+
			"must cost more, so the geometry is not reaching the pricer",
			stripped, with, without)
	}
}
