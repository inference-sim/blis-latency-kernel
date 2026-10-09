package latencykernel

import "testing"

// WHAT THE KERNEL DOES NOT PRICE, IT REFUSES. Each case is a deployment that either does not
// start in vLLM v0.31.0 or that this kernel cannot price truthfully, and each sits beside a
// control that differs in the one setting and does start, so a refusal for some other reason
// cannot pass for the right one.
func TestWhatTheKernelCannotPriceItRefuses(t *testing.T) {
	starts := func(fixtureName string, edit func(*Inputs)) bool {
		t.Helper()
		in := fixtureInputs(t, fixtureName)
		edit(&in)
		_, err := New(in)
		return err == nil
	}
	cache := func(c string) func(*Inputs) {
		return func(in *Inputs) { in.Deployment.Pools[0].Engine.CacheDType = c }
	}
	mamba := func(c string) func(*Inputs) {
		return func(in *Inputs) { in.Deployment.Pools[0].Engine.MambaCacheDType = c }
	}
	renamed := func(name string) func(*Inputs) {
		return func(in *Inputs) {
			chip := *in.Chip
			chip.Name = name
			in.Chip = &chip
		}
	}
	for _, c := range []struct {
		name    string
		fixture string
		edit    func(*Inputs)
		want    bool
	}{
		// Pipeline parallelism: no stage split, no inter-stage transfer.
		{"pp 1", dcpMLAFixture, func(in *Inputs) { in.Deployment.Pools[0].Parallel.PP = 1 },
			true},
		{"pp 2", dcpMLAFixture, func(in *Inputs) { in.Deployment.Pools[0].Parallel.PP = 2 },
			false},
		// Cache dtypes: the engine's closed list (vllm/config/cache.py:39-58), less the
		// formats whose page this kernel does not size.
		{"cache fp8_e5m2", dcpMLAFixture, cache("fp8_e5m2"), true},
		{"cache bogus", dcpMLAFixture, cache("bogus"), false},
		{"cache int8_per_token_head", dcpMLAFixture, cache("int8_per_token_head"), false},
		{"cache turboquant_k8v4", dcpMLAFixture, cache("turboquant_k8v4"), false},
		{"mamba float32", dcpMLAFixture, mamba("float32"), true},
		{"mamba fp8", dcpMLAFixture, mamba("fp8"), false},
		// A DSA model below SM90 has no sparse-MLA backend; a plain-MLA model on the same
		// renamed part is the control.
		{"plain MLA on l40s", dcpMLAFixture, renamed("l40s"), true},
		{"DSA on l40s", dcpSparseFixture, renamed("l40s"), false},
		{"DSA on a100-80", dcpSparseFixture, renamed("a100-80"), false},
		{"DSA on h200", dcpSparseFixture, renamed("h200"), true},
	} {
		if got := starts(c.fixture, c.edit); got != c.want {
			t.Errorf("%s: starts %v, want %v", c.name, got, c.want)
		}
	}
}
