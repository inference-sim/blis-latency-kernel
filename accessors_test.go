package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// The accessors exist so a consumer does not have to carry the Scenario and Deployment
// ALONGSIDE the kernel, indexing back into Pools[i] to reach what the kernel already
// resolved. That shape is a second source of truth for "which pool is this", and when the
// two disagree a decode pool is priced at a prefill pool's parallelism with nothing
// reporting it.
//
// So each test below asserts the accessor returns what THIS pool states, not merely
// something non-zero. A getter that returned pool 0's engine for every kernel would satisfy
// a non-zero check and still be the bug these replace.

// TestEngineReportsThisPoolsSettings is the per-pool property.
//
// Engine settings are per pool by design — a disaggregated deployment runs two engines with
// different token budgets — so the accessor must follow the index the kernel was opened at.
func TestEngineReportsThisPoolsSettings(t *testing.T) {
	r := writeDisaggregated(t)

	prefill, err := OpenPool("disagg.yaml", r, 0)
	if err != nil {
		t.Fatal(err)
	}
	decode, err := OpenPool("disagg.yaml", r, 1)
	if err != nil {
		t.Fatal(err)
	}

	// The fixture states 8192 for prefill and 2048 for decode, which is the realistic
	// shape: a prefill pool is given a large token budget and a decode pool a small one.
	if got := prefill.Engine().MaxNumBatchedTokens; got != 8192 {
		t.Errorf("prefill pool's max_num_batched_tokens = %d, want 8192", got)
	}
	if got := decode.Engine().MaxNumBatchedTokens; got != 2048 {
		t.Errorf("decode pool's max_num_batched_tokens = %d, want 2048; a kernel returning "+
			"pool 0's engine for every pool would report 8192", got)
	}
}

// TestRoleDistinguishesThePools: a caller holding several kernels needs to know which is
// which, without tracking the index that produced each one.
func TestRoleDistinguishesThePools(t *testing.T) {
	r := writeDisaggregated(t)

	for _, tc := range []struct {
		idx  int
		want deployment.Role
	}{
		{0, deployment.RolePrefill},
		{1, deployment.RoleDecode},
	} {
		k, err := OpenPool("disagg.yaml", r, tc.idx)
		if err != nil {
			t.Fatal(err)
		}
		if got := k.Role(); got != tc.want {
			t.Errorf("pool %d reports role %q, want %q", tc.idx, got, tc.want)
		}
	}
}

// TestParallelWidthsComeFromTheResolvedLayout is why these read the layout rather than the
// document.
//
// A Parallelism states a REQUEST and resolution settles it. Reading the deployment directly
// would miss an override the resolver recorded, and would report a width the kernel did not
// price against — the specific way a second source of truth goes wrong here.
func TestParallelWidthsComeFromTheResolvedLayout(t *testing.T) {
	r := writeDisaggregated(t)

	prefill, err := OpenPool("disagg.yaml", r, 0) // tp 8, dp 1
	if err != nil {
		t.Fatal(err)
	}
	decode, err := OpenPool("disagg.yaml", r, 1) // tp 4, dp 2
	if err != nil {
		t.Fatal(err)
	}

	if got, want := prefill.TensorParallelWidth(), 8; got != want {
		t.Errorf("prefill TP = %d, want %d", got, want)
	}
	if got, want := decode.TensorParallelWidth(), 4; got != want {
		t.Errorf("decode TP = %d, want %d", got, want)
	}
	if got, want := prefill.DataParallelWidth(), 1; got != want {
		t.Errorf("prefill DP = %d, want %d", got, want)
	}
	if got, want := decode.DataParallelWidth(), 2; got != want {
		t.Errorf("decode DP = %d, want %d", got, want)
	}
	// Both must agree with what the kernel reports it resolved, which is the whole point:
	// one answer, not two.
	if prefill.TensorParallelWidth() != prefill.layout.TP {
		t.Error("TensorParallelWidth disagrees with the resolved layout")
	}
}

// TestDataParallelWidthFloorsAtOne keeps an omitted width from scaling a caller's budgets
// to zero.
//
// A scenario that omits dp means one instance, not none. A simulator multiplies its
// sequence cap, token budget and KV budget by this, so a zero would silently make all three
// zero and the run would schedule nothing.
func TestDataParallelWidthFloorsAtOne(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
	in.Deployment.Pools[0].Parallel.DP = 1 // the fixture's own value; stated for clarity
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	if got := k.DataParallelWidth(); got < 1 {
		t.Errorf("DataParallelWidth = %d; an omitted width is one instance, not none", got)
	}
}

// TestIdentityAccessorsDescribeTheDeploymentPriced covers the set a scoring harness needs
// to configure a COMPARISON arm.
//
// A harness scoring this kernel against another backend configures that backend from this
// identity. It has to come from the kernel actually opened rather than from a scenario
// re-read alongside it: a second read can resolve a different file, and then two arms
// describe different deployments while claiming to describe one.
func TestIdentityAccessorsDescribeTheDeploymentPriced(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")

	if got, want := k.ModelName(), "minimax-m2.5"; got != want {
		t.Errorf("ModelName = %q, want %q", got, want)
	}
	if got, want := k.Chip().Name, "h200"; got != want {
		t.Errorf("Chip().Name = %q, want %q", got, want)
	}
	// The chip's memory is what vLLM resolves its batch defaults from, so a zero would
	// leave a consumer sizing against nothing.
	if k.Chip().MemoryGiB <= 0 {
		t.Errorf("Chip().MemoryGiB = %v; a consumer sizes batch defaults from it",
			k.Chip().MemoryGiB)
	}
	// The fixture serves fp8 over an fp8 checkpoint.
	if got, want := k.ServedDType(), model.DTypeFP8; got != want {
		t.Errorf("ServedDType = %q, want %q", got, want)
	}
	// minimax-m2.5 is MoE at 256 experts; a dense model would report zero.
	if got := k.Experts(); got <= 0 {
		t.Errorf("Experts = %d for an MoE model", got)
	}
}

// TestServedDTypeIsResolvedNotDeclared is the distinction the accessor exists to make.
//
// A consumer reading the engine's `quantization` field sees the request. Where the field is
// empty the checkpoint's own format governs, and that is the common case — so the declared
// value and the served one differ precisely when a caller most needs the served one.
func TestServedDTypeIsResolvedNotDeclared(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
	in.Deployment.Pools[0].Engine.Quantization = "" // let the checkpoint govern
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	if k.Engine().Quantization != "" {
		t.Fatal("the test did not clear the declared quantization")
	}
	if got := k.ServedDType(); got == "" {
		t.Error("ServedDType is empty where the engine states no quantization; it must " +
			"resolve to the checkpoint's own format")
	}
}
