package latencykernel

import "testing"

func TestConstructsFromCommittedArtifacts(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep16.yaml")
	r := k.Resolved()
	t.Logf("ep width %d, allreduce %s, sp-moe %v, overrides %d",
		r.ExpertParallelWidth, r.AllReduceBackend, r.SequenceParallelMoE,
		len(r.Overrides))
	measured, total, assumed := k.Evidence()
	t.Logf("coefficients: %d resolved, %d evidenced, %d assumed", total, measured,
		len(assumed))
}
