package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/routing"
)

func routingValue(tok, exp float64) *routing.Resolution {
	return &routing.Resolution{Layer: routing.LayerScenario, Source: "test",
		Value: &routing.Value{TokenImbalance: tok, ExpertImbalance: &exp, Source: "test"}}
}

func withRouting(t *testing.T, scenario string, r *routing.Resolution) *Kernel {
	t.Helper()
	in := fixtureInputs(t, scenario)
	in.Routing = r
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// The busiest expert-parallel rank gates every MoE layer's combine, so a resolved token
// imbalance multiplies the routed compute and nothing else, and the expert imbalance
// multiplies the routed weight read and nothing else.
func TestRoutingImbalanceScalesTheBusiestRanksRoutedWork(t *testing.T) {
	const sc = "minimax-m25-h200-ep8.yaml"
	even := fixture(t, sc)
	tokOnly := withRouting(t, sc, routingValue(1.5, 1))
	expOnly := withRouting(t, sc, routingValue(1, 0.5))
	b := decodeBatch(256, 1, 8192)

	sm := func(k *Kernel) float64 { return k.StepTime(b).PerResource[kernel.ResourceSM].Seconds() }
	hbm := func(k *Kernel) float64 { return k.StepTime(b).PerResource[kernel.ResourceHBM].Seconds() }

	if sm(tokOnly) <= sm(even)*1.05 {
		t.Errorf("a 1.5x busiest-rank token load left SM at %.4g against %.4g ms", sm(tokOnly)*1e3, sm(even)*1e3)
	}
	if hbm(tokOnly) != hbm(even) {
		t.Errorf("token imbalance moved the HBM term: %v against %v", hbm(tokOnly), hbm(even))
	}
	if hbm(expOnly) >= hbm(even) {
		t.Errorf("halving the busiest rank's distinct experts did not cut the weight read: %v against %v",
			hbm(expOnly), hbm(even))
	}
	if sm(expOnly) != sm(even) {
		t.Errorf("expert imbalance moved the SM term")
	}
}

// Without expert parallelism there is no cross-rank term, whatever is supplied, and an
// expert-parallel run given nothing says in its provenance that it priced the ranks evenly.
func TestRoutingImbalanceAppliesOnlyUnderExpertParallelism(t *testing.T) {
	if k := withRouting(t, "minimax-m25-h200-tp8.yaml", routingValue(2, 2)); k.routing != nil ||
		k.layout.ExpertWidth > 1 {
		t.Errorf("a tensor-parallel MoE adopted a cross-rank routing imbalance")
	}

	even := fixture(t, "minimax-m25-h200-ep8.yaml")
	found := false
	for _, o := range even.Provenance() {
		if o.Name == "routing_imbalance" {
			found = true
			if o.Method != "assumed" || !strings.Contains(o.Scope, "evenly loaded") {
				t.Errorf("the default is not recorded as an even-load assumption: %+v", o)
			}
		}
	}
	if !found {
		t.Error("an expert-parallel run with no routing resolution does not record it")
	}
	pinned := withRouting(t, "minimax-m25-h200-ep8.yaml", routingValue(1.6, 1))
	for _, o := range pinned.Provenance() {
		if o.Name == "routing_imbalance" && !strings.HasPrefix(o.Set, "scenario") {
			t.Errorf("the resolved layer is not recorded: %+v", o)
		}
	}
}

// A curve is read at the step's own routed token count and phase: the same imbalance
// curve costs more at a small batch than a large one, and a prefill chunk reads the
// prefill curve.
func TestRoutingCurveIsReadAtTheStepsBatchAndPhase(t *testing.T) {
	prof := func(phase routing.Phase, small, large float64) *routing.Profile {
		return &routing.Profile{Model: "m", Category: "c", Phase: phase,
			Placement: routing.PlacementLinear, ExpertParallel: 8, Method: "measured",
			Points: []routing.Point{{BatchTokens: 1, TokenImbalance: small, ExpertImbalance: 1},
				{BatchTokens: 4096, TokenImbalance: large, ExpertImbalance: 1}}}
	}
	r := &routing.Resolution{Layer: routing.LayerScenario, Set: "s",
		Decode: prof(routing.PhaseDecode, 3, 1.5), Prefill: prof(routing.PhasePrefill, 4, 4)}
	k := withRouting(t, "minimax-m25-h200-ep8.yaml", r)
	small, large := k.shapeOf(decodeBatch(4, 1, 2048)), k.shapeOf(decodeBatch(1024, 1, 2048))
	ts, _ := k.routingAt(&small, small.tokensF*k.moeDPFunnel())
	tl, _ := k.routingAt(&large, large.tokensF*k.moeDPFunnel())
	if ts <= tl {
		t.Errorf("the decode curve was not read at the step's batch: %v at 4 tokens, %v at 1024", ts, tl)
	}
	pf := k.shapeOf(decodeBatch(1, 2048, 2048))
	if tp, _ := k.routingAt(&pf, pf.tokensF); tp != 4 {
		t.Errorf("a prefill chunk read %v, not the prefill curve's 4", tp)
	}
}
