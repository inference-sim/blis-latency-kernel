package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// An MLA layer and a full-attention layer are different kernels and must not price
// identically. The byte count was already right -- blis-catalog declares an MLA node as
// `n_kv: 1, d_h: 576`, so NumKVHeads*HeadDim is the latent width kv_lora_rank +
// qk_rope_head_dim and kvGeometry needs no special case -- but the RATE was not: an MLA
// decode floor is several times a GQA decode's (51.5-89.5us against 9.5-19.5us) because its
// per-call setup reads a latent cache, and it sustains a different fraction of peak
// (0.61-0.80 against 0.52-0.88).
//
// DeepSeek-V3 is the case that matters: every attention layer in its stack is `kind: mla`,
// so before this dispatch existed its whole attention term used the full-attention pair.
// Ten catalog models declare an mla or sparse_mla layer, and two of them (deepseek-v4-pro,
// kimi-k3) appear in both the FPM dataset and the InferenceX corpus, so this is on the
// scored path rather than hypothetical.
//
// Asserts BEHAVIOUR rather than a value: step time must move when the per-kind entry is
// removed. Checking that the coefficient resolves is what let a mutation deleting the SWA
// dispatch survive once -- the map was still populated while every layer used the part-wide
// rate.
func TestMLAAndFullAttentionPriceDifferently(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v3-b200-fp8-sglang-tp8.yaml")

	var mla *price.PlannedLayer
	for i := range k.plan.Layers {
		if k.plan.Layers[i].AttnKind == model.AttentionMLA {
			mla = &k.plan.Layers[i]
			break
		}
	}
	if mla == nil {
		t.Skip("this fixture carries no mla layer, so there is nothing to dispatch on")
	}
	fr, ok := k.attentionByKind[model.AttentionMLA]
	if !ok {
		t.Skip("the registry carries no per-kind mla fit for this part")
	}
	if fr.floor <= 0 || fr.rate <= 0 {
		t.Fatalf("the mla entry resolved to floor=%v rate=%v; both must be positive",
			fr.floor, fr.rate)
	}

	b := kernel.Batch{DecodeThreshold: 8, SMBudget: 132}
	for i := 0; i < 32; i++ {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 2048})
	}
	withKind := k.StepTime(b).Overlap

	saved := k.attentionByKind
	k.attentionByKind = map[model.AttentionKind]floorRate{}
	withoutKind := k.StepTime(b).Overlap
	k.attentionByKind = saved

	if withKind == withoutKind {
		t.Errorf("step time is %v both with and without the per-kind mla entry; the "+
			"dispatch resolves the coefficient but does not price with it", withKind)
	}
	t.Logf("mla floor=%v rate=%.0f B/s; step with kind %v, without %v",
		fr.floor, fr.rate, withKind, withoutKind)
}

// sparse_mla must NOT resolve the dense-MLA pair. It narrows the latent read to a selected
// top-k, so its byte count is a different function of the request, and the registry carries
// no fit for it. Mapping it onto the dense rate would charge a sparse read as a dense one --
// wrong in the expensive direction, and silently. Five catalog models declare sparse_mla, so
// this is a guard against a plausible future edit rather than a formality.
func TestSparseMLADoesNotBorrowTheDenseMLAFit(t *testing.T) {
	k := fixture(t, "aisimulate/deepseek-v3-b200-fp8-sglang-tp8.yaml")
	if _, ok := k.attentionByKind[model.AttentionSparseMLA]; ok {
		t.Error("sparse_mla resolved a per-kind entry; it has no fit in the registry and " +
			"must fall back to the part-wide terms rather than borrow the dense-MLA rate")
	}
}
