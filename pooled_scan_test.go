package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// scorersOf returns every block-index scorer the plan launches, by pointer so a test can
// vary its bound and re-price the same kernel.
func scorersOf(k *Kernel) []*price.PlannedAttention {
	var out []*price.PlannedAttention
	for i := range k.plan.Layers {
		for j := range k.plan.Layers[i].Attentions {
			if a := &k.plan.Layers[i].Attentions[j]; a.Role == "block_index_scores" {
				out = append(out, a)
			}
		}
	}
	return out
}

// A pooled indexer caches one state per index_kpool tokens and scores over those states,
// so the ratio bounds how many candidates its scan covers. The bound was dropped in
// planning, which priced every pooled scorer over the full token context.
//
// Asserted on STEP TIME, not on the plan: a plan that merely carries the ratio while the
// step ignores it is the failure mode a plan-level check cannot see -- the same trap
// TestSlidingWindowAndFullAttentionPriceDifferently documents for the per-kind rate.
//
// No committed catalog graph declares a pooled SCORER yet. deepseek-v4-pro carries
// compress_ratio on its PRIMARY attention (the compressed-tail read that
// selectedKVTokens prices), not on its scorer, and GLM-5.3-Flash is the first with a
// pooled one. So these tests vary the bound on a real fixture's own scorer rather than
// skipping: what is under test is the kernel's pricing, and a skip would leave the term
// unexercised in CI -- the silent-skip hole 7c556d7 closed elsewhere in this suite.
func TestAPooledScorersRatioChangesTheStepTime(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b300-fp4-vllm-tp4.yaml")
	scorers := scorersOf(k)
	if len(scorers) == 0 {
		t.Fatalf("the fixture's graph launches no block-index scorer, so the pooled-scan "+
			"term cannot be exercised; pick a fixture whose model has one")
	}

	b := longDecodeBatch()
	saved := saveRatios(scorers)
	defer restoreRatios(scorers, saved)

	// Unpooled: every token is its own candidate, which is what the kernel charged
	// before this term existed.
	setRatios(scorers, 1)
	unpooled := k.StepTime(b).Overlap

	// Pooled 4:1, GLM-5.3-Flash's index_kpool. Everything else held fixed.
	setRatios(scorers, 4)
	pooled := k.StepTime(b).Overlap

	if pooled == unpooled {
		t.Fatalf("step time is %v whether or not the scorer is pooled; the plan carries "+
			"the ratio but the step does not price with it", pooled)
	}
	if pooled >= unpooled {
		t.Fatalf("the pooled step (%v) is not below the unpooled one (%v); pooling "+
			"divides the candidate count, so a pooled scan must read LESS",
			pooled, unpooled)
	}
}

// METAMORPHIC: a larger pool must never cost more. The scan covers ceil(context/ratio)
// states, so priced time is monotonically non-increasing in the ratio. A kernel that
// multiplied instead of divided, or that clamped, passes a single-value check and fails
// this one.
func TestPooledScanTimeIsMonotonicInThePoolSize(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b300-fp4-vllm-tp4.yaml")
	scorers := scorersOf(k)
	if len(scorers) == 0 {
		t.Fatal("the fixture's graph launches no block-index scorer")
	}

	b := longDecodeBatch()
	saved := saveRatios(scorers)
	defer restoreRatios(scorers, saved)

	var prev float64
	seen := map[float64]bool{}
	for _, ratio := range []int{1, 2, 4, 8, 16, 64} {
		setRatios(scorers, ratio)
		got := k.StepTime(b).Overlap.Seconds()
		if prev != 0 && got > prev {
			t.Fatalf("ratio %d priced %.9fs, above the previous ratio's %.9fs; a larger "+
				"pool holds FEWER states per span, so the scan cannot cost more",
				ratio, got, prev)
		}
		prev = got
		seen[got] = true
	}
	// STRICT sensitivity: each DOUBLING of the pool must move the price, not merely
	// fail to raise it. Monotonicity alone cannot catch a kernel that divides by a
	// CONSTANT -- that is trivially non-increasing, and because the `> 1` guard still
	// makes ratio 1 differ, it even produces two distinct values. Requiring a distinct
	// time per ratio is what closes that: a constant divisor collapses 2, 4, 8, 16 and
	// 64 onto one time, and an off-by-one divisor (ratio+1) shifts every value but
	// still yields six distinct ones, so it is caught by the exact-arithmetic test
	// instead.
	if len(seen) != 6 {
		t.Fatalf("six pool sizes from 1 to 64 produced only %d distinct step times; a "+
			"scan that read its ratio gives a different time for each, so the field is "+
			"being ignored or collapsed to a constant", len(seen))
	}
}

// The scan bound must be the ratio EXACTLY, not merely something that decreases with it.
// Asserted against the arithmetic rather than against a recorded number: the scorer term
// is linear in the candidate count, so halving the pool size must halve the scorer's
// contribution to the step. Measured as a difference of differences, which cancels every
// other term in the step and leaves the scan.
func TestPooledScanBoundIsTheRatioExactly(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b300-fp4-vllm-tp4.yaml")
	scorers := scorersOf(k)
	if len(scorers) == 0 {
		t.Fatal("the fixture's graph launches no block-index scorer")
	}

	b := longDecodeBatch()
	saved := saveRatios(scorers)
	defer restoreRatios(scorers, saved)

	at := func(ratio int) float64 {
		setRatios(scorers, ratio)
		return k.StepTime(b).Overlap.Seconds()
	}

	// The scorer contributes S/r at ratio r, so t(r) = base + S/r. Then
	//   t(1) - t(2) = S/2  and  t(2) - t(4) = S/4,
	// whose ratio is exactly 2 for any S and any base. A divisor that is off by one, or
	// a constant, breaks that identity while still being monotone.
	d12 := at(1) - at(2)
	d24 := at(2) - at(4)
	if d24 <= 0 {
		t.Fatalf("the scan did not shrink between ratio 2 and 4 (delta %.9fs)", d24)
	}
	if got := d12 / d24; got < 1.9 || got > 2.1 {
		t.Fatalf("t(1)-t(2) over t(2)-t(4) is %.4f, want 2.0: the scan term is S/ratio, "+
			"so the two deltas are S/2 and S/4. A value away from 2 means the divisor is "+
			"not the ratio itself", got)
	}
}

// A ratio of zero or one means no pooling, and must price exactly alike: the bound is
// absent either way. This is the no-regression guarantee for every model that declares
// no index_kpool, which is all of them but one.
func TestAnAbsentPoolRatioPricesAsOne(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v4-pro-b300-fp4-vllm-tp4.yaml")
	scorers := scorersOf(k)
	if len(scorers) == 0 {
		t.Fatal("the fixture's graph launches no block-index scorer")
	}

	b := longDecodeBatch()
	saved := saveRatios(scorers)
	defer restoreRatios(scorers, saved)

	setRatios(scorers, 0)
	zero := k.StepTime(b).Overlap
	setRatios(scorers, 1)
	one := k.StepTime(b).Overlap

	if zero != one {
		t.Fatalf("an unset ratio priced %v and a ratio of 1 priced %v; both mean one "+
			"state per token, so a model declaring no pooling must be unaffected",
			zero, one)
	}
}

// longDecodeBatch is a decode batch at a long context, because the scan is the
// context-proportional term: at a short context the bound changes little and a test
// would not discriminate.
func longDecodeBatch() kernel.Batch {
	b := kernel.Batch{DecodeThreshold: 8, SMBudget: 132}
	for i := 0; i < 32; i++ {
		b.Reqs = append(b.Reqs, kernel.ReqShape{
			Scheduled: 1, Computed: 65536, PromptLen: 65536})
	}
	return b
}

func saveRatios(as []*price.PlannedAttention) []int {
	out := make([]int, len(as))
	for i, a := range as {
		out[i] = a.CompressRatio
	}
	return out
}

func setRatios(as []*price.PlannedAttention, r int) {
	for _, a := range as {
		a.CompressRatio = r
	}
}

func restoreRatios(as []*price.PlannedAttention, saved []int) {
	for i, a := range as {
		a.CompressRatio = saved[i]
	}
}
