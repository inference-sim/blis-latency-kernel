package latencykernel

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/acceptance"
)

// Adaptive verification (speculative_config.enable_adaptive_verification, vLLM v0.32.0,
// vllm/config/speculative.py:551-553@5df63a72): the verifier runs a draft budget the engine
// chooses each step, rather than every draft the scheduler budgeted. The scheduler still
// budgets 1 + K tokens a request (nothing under vllm/v1/core reads the flag), the drafter still
// runs full blocks (vllm/v1/attention/selector.py:143-152@5df63a72), and only the target's
// forward runs the trimmed width.

// adaptiveVerify is the resolved adaptive-verification state; nil when the engine does not
// enable it.
type adaptiveVerify struct {
	// survival[i] is the predicted probability that a request's first i+1 drafts are all
	// accepted.
	survival []float64
}

// resolveAdaptive settles adaptive verification from the engine settings and the run's
// acceptance. With the flag off it leaves k.adaptive nil, and nothing else in the kernel reads
// the acceptance.
func (k *Kernel) resolveAdaptive(acc *acceptance.Resolution) error {
	s := k.engine.Speculative
	if s == nil || !s.EnableAdaptiveVerification || k.numSpecTokens() == 0 {
		return nil
	}
	if acc == nil {
		return fmt.Errorf("speculative.enable_adaptive_verification is on, and the kernel prices the " +
			"verify budget the engine chooses from per-position acceptance: Inputs.Acceptance, the " +
			"run's resolved acceptance (blis-schemas acceptance.Resolve), is required")
	}
	k.adaptive = &adaptiveVerify{}
	return nil
}

// VerifiedDrafts reports, for each request of b in order, how many of its scheduled drafts the
// verifier runs this step under adaptive verification: zero for a request that verifies none.
// It is nil when the engine does not enable adaptive verification, where every scheduled draft
// is verified. A simulator caps a request's accepted drafts at it, since a draft the verifier
// never ran cannot be accepted.
func (k *Kernel) VerifiedDrafts(b kernel.Batch) []int {
	if k.adaptive == nil {
		return nil
	}
	out := make([]int, len(b.Reqs))
	for i, r := range b.Reqs {
		out[i] = k.scheduledDrafts(r)
	}
	return out
}

// scheduledDrafts is how many drafts a request verifies as scheduled: a decode schedules its
// bonus position and its drafts (kernel.ReqShape.Scheduled), so Scheduled - 1 of them, at most
// num_spec_tokens; a prefill chunk verifies none.
func (k *Kernel) scheduledDrafts(r kernel.ReqShape) int {
	if r.Scheduled <= 1 || r.Computed < r.PromptLen {
		return 0
	}
	return min(r.Scheduled-1, k.numSpecTokens())
}
