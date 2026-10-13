package latencykernel

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// Step-time behaviour, as relations between steps, deployments and graphs.

func prefillOf(tokens, prefix int) kernel.Batch {
	return kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: tokens, Computed: prefix, PromptLen: prefix + tokens}}}
}

// A recurrent prefill runs the chunked scan, not the decode kernel's sequential one: the
// smaller GLM-5.3-Flash, 34 of whose layers are KDA, prefills no slower than GLM-5 on the same
// deployment, where pricing its prefill tokens at the decode scan's rate made it over twice as
// slow.
func TestRecurrentPrefillIsNotPricedAtTheDecodeRate(t *testing.T) {
	glm5 := fixture(t, "glm5-h200-tp8.yaml")
	in := fixtureInputs(t, "glm5-h200-tp8.yaml")
	withModel(t, &in, "glm-5.3-flash")
	flash := build(t, in)
	b := prefillOf(8192, 0)
	if f, g := flash.StepTime(b).NoOverlap, glm5.StepTime(b).NoOverlap; f > g {
		t.Errorf("an 8K GLM-5.3-Flash prefill takes %v, slower than the larger GLM-5's %v", f, g)
	}
}

// Running a step with no graph launches every kernel, so it is never cheaper on the host
// than replaying captured pieces: NONE >= PIECEWISE >= FULL for any batch.
func TestEagerLaunchesCostAtLeastAReplay(t *testing.T) {
	for _, scenario := range []string{"glm5-h200-tp8.yaml", "kimi-k3-h100-nospec.yaml", v41} {
		host := map[string]float64{}
		for _, mode := range []string{"NONE", "PIECEWISE", "FULL"} {
			in := fixtureInputs(t, scenario)
			in.Deployment.Pools[0].Engine.CUDAGraphMode = mode
			host[mode] = build(t, in).StepTime(decodeBatch(1, 1, 8000)).
				PerResource[kernel.ResourceHost].Seconds()
		}
		if host["NONE"] < host["PIECEWISE"] || host["PIECEWISE"] < host["FULL"] {
			t.Errorf("%s: host time NONE %.3g, PIECEWISE %.3g, FULL %.3g; capturing more "+
				"cannot cost the host more", scenario, host["NONE"], host["PIECEWISE"], host["FULL"])
		}
	}
}

// Adding a request never makes a step cheaper -- in particular across the CUDA-graph
// capture ceiling, past which the step runs with no graph. Checked on random batches
// straddling the ceiling, over dense, sparse-MLA, recurrent and V4.1 deployments, adding
// a decode or a short prefill.
func TestAddingARequestNeverLowersTheStep(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for _, scenario := range []string{"glm5-h200-tp8.yaml", "minimax-m25-h200-tp8.yaml",
		"kimi-k3-h100-nospec.yaml", v41} {
		k := fixture(t, scenario)
		ceiling := k.captureTokens
		for trial := 0; trial < 200; trial++ {
			var b kernel.Batch
			b.DecodeThreshold = k.uniformDecodeWidth
			total := ceiling - 12 + r.Intn(12)
			for total > 0 {
				s := k.uniformDecodeWidth
				if r.Intn(3) == 0 {
					s = 1 + r.Intn(min(total, 64))
				}
				s = min(s, total)
				ctx := 1 + r.Intn(16384)
				b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: s, Computed: ctx,
					PromptLen: ctx + s})
				total -= s
			}
			before := k.StepTime(b).NoOverlap
			extra := 1 + r.Intn(16)
			if r.Intn(2) == 0 {
				extra = k.uniformDecodeWidth
			}
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: extra, Computed: 4096,
				PromptLen: 4096 + extra})
			if after := k.StepTime(b).NoOverlap; after < before {
				t.Fatalf("%s: adding a %d-token request to a %d-token step lowered it from "+
					"%v to %v", scenario, extra, b.Tokens()-extra, before, after)
			}
		}
	}
}

// A top-k layer's prefill query attends its selection, not its whole prefix: a chunk's
// attention barely grows with how much context precedes it, where a full-attention chunk's
// grows with the prefix.
func TestSparsePrefillIsBoundedByItsSelection(t *testing.T) {
	k := fixture(t, "glm5-h200-tp8.yaml") // DSA, index_topk 2048
	sm := func(prefix int) float64 {
		return k.StepTime(prefillOf(1024, prefix)).PerResource[kernel.ResourceSM].Seconds()
	}
	if r := sm(131072) / sm(8192); r > 1.5 {
		t.Errorf("a 1,024-token chunk on a 128K prefix costs %.2fx one on an 8K prefix; "+
			"each query reads at most its top-k", r)
	}
}

// The indexer's scoring scan runs in prefill too: removing it makes a prefill cheaper on SM,
// not only by its launches.
func TestTheIndexerScanIsPricedInPrefill(t *testing.T) {
	with := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	if editNodes(in.Model, role("block_index_scores"),
		func(n *model.Node) { n.NumQHeads /= 2 }) == 0 {
		t.Fatal("the graph states no indexer")
	}
	narrower := build(t, in)
	b := prefillOf(8192, 0)
	if w, n := with.StepTime(b).PerResource[kernel.ResourceSM],
		narrower.StepTime(b).PerResource[kernel.ResourceSM]; w <= n {
		t.Errorf("an 8K prefill takes %v of SM with the indexer's heads and %v with half "+
			"of them; the scan is not priced in prefill", w, n)
	}
}

// Speculating costs the drafter's passes and holds its weights: the same verify step is
// dearer, and the deployment holds more, than the deployment that does not speculate. A parallel-block drafter
// runs once where an autoregressive one runs once per drafted token, so it is no dearer.
func TestTheDrafterIsPriced(t *testing.T) {
	spec := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	in.Deployment.Pools[0].Engine.Speculative = nil
	none := build(t, in)
	b := decodeBatch(32, 6, 8192)
	if spec.StepTime(b).NoOverlap <= none.StepTime(b).NoOverlap {
		t.Error("a speculating step costs no more than the same step with no drafter")
	}
	if spec.FixedBytes().Weights <= none.FixedBytes().Weights {
		t.Error("the drafter's weights are not resident")
	}

	in = fixtureInputs(t, v41)
	in.Model.Speculator.Drafting = model.DraftingAutoregressive
	in.Model.Speculator.SequentialHead = nil
	in.Model.Speculator.ContextStack = nil
	for i := range in.Model.LayerKinds {
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			if n.Window == 133 {
				n.Window = 128 // an autoregressive draft reads only its ring
			}
		}
	}
	ar := build(t, in)
	in = fixtureInputs(t, v41)
	in.Model.Speculator.SequentialHead = nil
	in.Model.Speculator.ContextStack = nil
	pb := build(t, in)
	if pb.StepTime(b).NoOverlap > ar.StepTime(b).NoOverlap {
		t.Error("one parallel-block pass costs more than five autoregressive ones")
	}

	// An aliased weight is read, not held: the drafter's LM head reads the target's.
	in = fixtureInputs(t, v41)
	for i := range in.Model.Speculator.Head {
		in.Model.Speculator.Head[i].Aliases = ""
		if in.Model.Speculator.Head[i].Op == model.OpGEMM {
			in.Model.Speculator.Head[i].WeightDType = model.DTypeBF16
		}
	}
	own := build(t, in)
	if own.FixedBytes().Weights <= spec.FixedBytes().Weights {
		t.Error("an LM head of the drafter's own holds no more than one aliasing the target's")
	}
	// And computes alike: an alias runs at its target's format, here BF16, exactly as a
	// BF16 copy of its own would, on every chip.
	for _, chip := range []string{"h200", "b200"} {
		a, o := fixtureInputs(t, v41), in
		if chip != "h200" {
			a.Chip, a.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
			o = fixtureInputs(t, v41)
			for i := range o.Model.Speculator.Head {
				o.Model.Speculator.Head[i].Aliases = ""
				if o.Model.Speculator.Head[i].Op == model.OpGEMM {
					o.Model.Speculator.Head[i].WeightDType = model.DTypeBF16
				}
			}
			o.Chip, o.Scenario.Cluster.Hardware = a.Chip, chip
		}
		if x, y := build(t, a).StepTime(b).NoOverlap, build(t, o).StepTime(b).NoOverlap; x != y {
			t.Errorf("%s: a drafter aliasing the BF16 LM head steps in %v, its own BF16 copy in %v",
				chip, x, y)
		}
	}
}

// FlashMLA on SM90 pads each rank's query heads to 64: at TP8 a 64-head V4.1 rank computes
// 64 heads whether the model has 64 or 32, so halving the attention's heads leaves the
// prefill unchanged there -- and on SM100, which does not pad, lowers it.
func TestFlashMLAPadsQueryHeadsOnSM90(t *testing.T) {
	halve := func(in *Inputs) {
		editNodes(in.Model, func(n model.Node) bool {
			return n.Op == model.OpAttention && n.Role == "" && n.NumQHeads == 64
		}, func(n *model.Node) { n.NumQHeads = 32 })
	}
	b := prefillOf(8192, 0)
	for _, c := range []struct {
		chip string
		pads bool
	}{{"h200", true}, {"b200", false}} {
		in := fixtureInputs(t, v41)
		half := fixtureInputs(t, v41)
		if c.chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, c.chip), c.chip
			half.Chip, half.Scenario.Cluster.Hardware = chipInputs(t, c.chip), c.chip
		}
		halve(&half)
		full, h := build(t, in).StepTime(b).NoOverlap, build(t, half).StepTime(b).NoOverlap
		if c.pads && full != h {
			t.Errorf("%s: 64 heads prefill in %v and 32 in %v; both pad to 64 a rank",
				c.chip, full, h)
		}
		if !c.pads && h >= full {
			t.Errorf("%s: 32 heads prefill in %v, no faster than 64's %v", c.chip, h, full)
		}
	}
}
