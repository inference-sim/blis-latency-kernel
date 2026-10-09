package latencykernel

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"

	"github.com/inference-sim/blis-schemas/spec/coefficient"
)

// dropCoefficient returns the sets without the named entry.
func dropCoefficient(sets []*coefficient.Set, name string) []*coefficient.Set {
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			if e.Name != name {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		out[i] = &c
	}
	return out
}

// A MISSING OPTIONAL COEFFICIENT IS DISCLOSED WHERE IT MATTERS. The kernel prices without
// an optional coefficient, so dropping one the deployment needs must leave a step that
// differs from the full one AND a Provenance entry naming what is missing; dropping one it
// does not need -- a recurrent mixer's terms on a model with none -- must change neither.
// Before this, a registry missing kimi-k3's KDA rate priced its decode a tenth cheaper with
// nothing reported.
func TestAMissingOptionalCoefficientIsDisclosedWhereItMatters(t *testing.T) {
	decode := decodeBatch(32, 1, 4096)
	for _, c := range []struct {
		fixture, name string
		needed        bool
	}{
		{"kimi-k3-h100-nospec.yaml", "recurrent_decode_rate_kda", true},
		{dcpMLAFixture, "host_launch_per_kernel", true},
		{dcpMLAFixture, "attention_decode_rate_mla", true},
		{dcpMLAFixture, "recurrent_decode_rate_kda", false},
	} {
		full := fixture(t, c.fixture)
		in := fixtureInputs(t, c.fixture)
		in.Coefficients = dropCoefficient(in.Coefficients, c.name)
		k, err := New(in)
		if err != nil {
			t.Fatalf("%s without %s: %v", c.fixture, c.name, err)
		}
		if hasAssumption(full, c.name) {
			t.Errorf("%s with every coefficient discloses %s as missing", c.fixture, c.name)
		}
		moved := k.StepTime(decode).NoOverlap != full.StepTime(decode).NoOverlap
		if got := hasAssumption(k, c.name); got != c.needed {
			t.Errorf("%s without %s: disclosed %v, want %v", c.fixture, c.name, got, c.needed)
		}
		if moved != c.needed {
			t.Errorf("%s without %s: step moved %v, want %v", c.fixture, c.name, moved,
				c.needed)
		}
	}
}

// A DECODE READS THE CACHE IT HOLDS. Growing a one-request decode's context must add exactly
// the bytes the cache grew by (SequenceVariableBytes) over the attention decode rate to the
// HBM term -- on a hybrid stack as on a pure one. Nemotron-3-Ultra holds KV on 12 of its 108
// layers; dividing the whole-model KV figure by every layer, as the decode read once did,
// charged a ninth of it. Both fixtures' attention is full (gqa), so the part-wide rate
// applies; contexts are block-aligned, so the cache grows by whole pages. Equal to within
// the 2 ns that rounding to time.Duration allows.
func TestADecodeReadsTheCacheItHolds(t *testing.T) {
	for _, f := range []string{"nemotron3-ultra-h100-agg.yaml", "minimax-m25-h200-tp8.yaml"} {
		in, k := fixtureInputs(t, f), fixture(t, f)
		hbm := func(ctx int) float64 {
			return k.StepTime(decodeBatch(1, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		const short, long = 4096, 8192
		grew := float64(k.SequenceVariableBytes(long) - k.SequenceVariableBytes(short))
		want := grew / (registryValue(t, in, k, "attention_decode_rate") * 1e6)
		if got := hbm(long) - hbm(short); math.Abs(got-want) > 2e-9 {
			t.Errorf("%s: %d more tokens of context added %.3f us of HBM; the cache grew %.0f "+
				"bytes, which the attention rate reads in %.3f us", f, long-short, got*1e6,
				grew, want*1e6)
		}

		// With no measured attention rate the read falls back to HBM bandwidth -- the
		// chip's, derated by the registry's hbm_derate -- and must still be the cache's
		// growth, not a ninth of it.
		in.Coefficients = dropCoefficient(in.Coefficients, "attention_decode_rate")
		fb, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		fbHBM := func(ctx int) float64 {
			return fb.StepTime(decodeBatch(1, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		bw := in.Chip.MemoryBandwidthTBs * 1e12 * registryValue(t, in, fb, "hbm_derate")
		if got, want := fbHBM(long)-fbHBM(short), grew/bw; math.Abs(got-want) > 2e-9 {
			t.Errorf("%s without an attention rate: %d more tokens added %.3f us of HBM; the "+
				"cache grew %.0f bytes, which HBM reads in %.3f us", f, long-short, got*1e6,
				grew, want*1e6)
		}
	}
}

// THE ENGINE RELEASE PRICED IS DISCLOSED WHEN IT IS NOT THE ONE DECLARED. The fixtures declare
// 0.29.0, the one release with a rules pack, and the kernel encodes 0.31.0's behaviour; that
// must show in Provenance, and must not when the scenario declares 0.31.0 itself.
func TestTheEngineReleasePricedIsDisclosed(t *testing.T) {
	in := fixtureInputs(t, dcpMLAFixture)
	declared := func(v string) bool {
		t.Helper()
		sc := *in.Scenario
		sc.EngineVersion = v
		in2 := in
		in2.Scenario = &sc
		k, err := New(in2)
		if err != nil {
			t.Fatal(err)
		}
		return hasAssumption(k, "engine_version")
	}
	if !declared("0.29.0") {
		t.Error("a 0.29.0 scenario priced at 0.31.0's behaviour discloses nothing")
	}
	if declared(EngineBehaviourVersion) {
		t.Errorf("a %s scenario discloses a version mismatch", EngineBehaviourVersion)
	}
}
