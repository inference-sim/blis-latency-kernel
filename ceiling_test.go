package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
)

// THE CAPTURE CEILING FOLLOWS THE ENGINE'S DEFAULT, on every input that sets it. vLLM v0.31.0
// captures token counts up to min(max_num_seqs x decode_query_len x 2, 512), or 1024 on
// data-center Blackwell (VllmConfig._set_cudagraph_sizes, vllm/config/vllm.py:2442-2460), and
// a step above that runs with no graph, a uniform decode included
// (vllm/v1/worker/gpu/cudagraph_utils.py:499-529). The fixture's own max_num_seqs, 256, puts
// it at 512 on h200, which fixes neither the Blackwell figure nor the seqs term; these cases
// move each:
//
//   - h200 marked Blackwell, seqs 1024: ceiling 1024.
//   - h200, seqs 64: ceiling 128 (the seqs term binds).
//   - h200, seqs 1024: ceiling 512, and a uniform decode of 513 requests -- admissible at
//     that cap -- runs with no graph under FULL_AND_PIECEWISE.
//
// At the ceiling a step prices as the mode's own graph; one token above, as NONE.
func TestTheCaptureCeilingFollowsTheEngineDefault(t *testing.T) {
	host := func(mode string, seqs int, blackwell bool, b kernel.Batch) time.Duration {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		e := &in.Deployment.Pools[0].Engine
		e.CUDAGraphMode, e.MaxNumSeqs = mode, seqs
		if blackwell {
			chip := *in.Chip
			chip.NVFP4Peak = 1
			in.Chip = &chip
		}
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.StepTime(b).PerResource[kernel.ResourceHost]
	}
	prefill := func(tokens int) kernel.Batch { return decodeBatch(1, tokens, tokens) }
	for _, c := range []struct {
		name      string
		seqs      int
		blackwell bool
		ceiling   int
	}{
		{"blackwell, seqs 1024", 1024, true, 1024},
		{"h200, seqs 64", 64, false, 128},
		{"h200, seqs 1024", 1024, false, 512},
	} {
		at, above := prefill(c.ceiling), prefill(c.ceiling+1)
		if a, b := host("PIECEWISE", c.seqs, c.blackwell, at),
			host("NONE", c.seqs, c.blackwell, at); a == b {
			t.Errorf("%s: a %d-token prefill, at the ceiling, priced like NONE", c.name,
				c.ceiling)
		}
		if a, b := host("PIECEWISE", c.seqs, c.blackwell, above),
			host("NONE", c.seqs, c.blackwell, above); a != b {
			t.Errorf("%s: a %d-token prefill, past the ceiling, priced %v against NONE's %v",
				c.name, c.ceiling+1, a, b)
		}
	}
	at, above := decodeBatch(512, 1, 2048), decodeBatch(513, 1, 2048)
	if a, b := host("FULL_AND_PIECEWISE", 1024, false, at),
		host("NONE", 1024, false, at); a == b {
		t.Error("a 512-request uniform decode, at the ceiling, priced like NONE")
	}
	if a, b := host("FULL_AND_PIECEWISE", 1024, false, above),
		host("NONE", 1024, false, above); a != b {
		t.Errorf("a 513-request uniform decode, past the ceiling, priced %v against NONE's %v",
			a, b)
	}
}

// A DCP GROUP SITS WHERE THE ENGINE'S RANK LAYOUT PUTS IT, seen in the step rather than the
// layout. vLLM numbers ranks DP x PP x PCP x TP with TP innermost
// (vllm/distributed/parallel_state.py:2054-2060), so a DCP group that spans the PCP axis has
// members tp ranks apart. On glm5 at tp=8, pcp=2, dcp=2 that is two nodes, so a decode step
// must charge the NIC: a decode step's only cross-node collective is that group's combine.
// Without the stride it would be priced as two adjacent ranks on one node's NVLink.
//
// glm5 states the ag_rs combine here and in the next test: GLM-5's model hook would
// otherwise choose a2a (vllm/model_executor/models/config.py:43-50), which PCP with DCP
// refuses (vllm/v1/worker/gpu/pcp_manager.py:187-194).
func TestADCPGroupSitsWhereTheRankLayoutPutsIt(t *testing.T) {
	k := mustDCPVariant(t, dcpSparseFixture, 8, 2, 2, agRS)
	if got := k.StepTime(decodeBatch(8, 1, 4096)).PerResource[kernel.ResourceNIC]; got <= 0 {
		t.Errorf("tp=8, pcp=2, dcp=2: a decode charged %v of NIC; the DCP group spans two "+
			"nodes", got)
	}
}

// UNDER PCP WITH DCP A SHORT PREFILL IS REPLICATED, and the step pays for it. With DCP on,
// MRV2 gives every PCP rank the whole query of a prefill too short to fill all 2 x pcp
// chunks: (2pcp - 1) x ceil(q / 2pcp) >= q (replicated_requests,
// vllm/v1/worker/gpu/pcp_manager.py:222-232 at v0.31.0). At pcp=4 a 17-token prefill drops a
// chunk (7 x 3 >= 17) and a 16-token one does not (7 x 2 < 16). So on glm5 at tp=2, pcp=4,
// the 17-token prefill must cost more SM at dcp=4 than at dcp=1, where nothing is
// replicated, and the 16-token one the same at both.
func TestUnderPCPWithDCPAShortPrefillIsReplicated(t *testing.T) {
	sm := func(dcp, tokens int) time.Duration {
		t.Helper()
		k := mustDCPVariant(t, dcpSparseFixture, 2, 4, dcp, agRS)
		return k.StepTime(decodeBatch(1, tokens, tokens)).PerResource[kernel.ResourceSM]
	}
	if a, b := sm(4, 17), sm(1, 17); a <= b {
		t.Errorf("a 17-token prefill at pcp=4 cost %v of SM with dcp=4 and %v without; "+
			"with DCP it is replicated to every rank", a, b)
	}
	if a, b := sm(4, 16), sm(1, 16); a != b {
		t.Errorf("a 16-token prefill at pcp=4 cost %v of SM with dcp=4 and %v without; it "+
			"fills every chunk, so DCP must not change its share", a, b)
	}
}

// agRS states the ag_rs DCP combine without a replicated query, which a GLM-5 deployment
// under DCP must state to start (see TestADCPGroupSitsWhereTheRankLayoutPutsIt).
func agRS(in *Inputs) {
	e := &in.Deployment.Pools[0].Engine
	if in.Deployment.Pools[0].Parallel.DCP > 1 {
		off := false
		e.DCPCommBackend, e.DCPQReplicate = "ag_rs", &off
	}
}

// A DECODE STEP DOES NOT LAUNCH THE GRAPH'S PREFILL-ONLY NODES. DeepSeek-V4.1 dequantizes and
// gathers its compressed KV, and its indexer gathers keys, only when the batch has prefills
// (vllm/models/deepseek_v41/nvidia/flashmla.py:143-165, 305, 317;
// vllm/model_executor/layers/sparse_attn_indexer.py:416-417, 481), and the graph marks those
// nodes phase: prefill. With no graph (NONE) every launch is a host charge, so adding one
// prefill row that samples nothing (no drafter pass joins it) must add exactly the eager plus
// dispatch cost of those launches, counted here from the graph -- and nothing on a graph
// that marks no node.
func TestADecodeStepDoesNotLaunchPrefillOnlyNodes(t *testing.T) {
	for _, c := range []struct {
		scenario string
		prefill  bool
	}{{v41, true}, {"minimax-m25-h200-tp8.yaml", false}} {
		in := fixtureInputs(t, c.scenario)
		in.Deployment.Pools[0].Engine.CUDAGraphMode = "NONE"
		want := 0
		for _, id := range in.Model.Stack.Expand() {
			for _, lk := range in.Model.LayerKinds {
				if lk.ID != id {
					continue
				}
				for _, n := range lk.Nodes {
					if n.Context != nil && n.Context.Phase == "prefill" {
						want++
					}
				}
			}
		}
		if (want > 0) != c.prefill {
			t.Fatalf("%s: %d prefill-only nodes in the graph", c.scenario, want)
		}
		k := build(t, in)
		dec := decodeBatch(4, 6, 90000)
		mixed := dec
		mixed.Reqs = append(append([]kernel.ReqShape{}, dec.Reqs...),
			kernel.ReqShape{Scheduled: 512, Computed: 1024, PromptLen: 8192})
		got := k.StepTime(mixed).PerResource[kernel.ResourceHost] -
			k.StepTime(dec).PerResource[kernel.ResourceHost]
		per := k.launchEager + k.launchPerKernel
		if d := got - time.Duration(want)*per; d < -time.Nanosecond || d > time.Nanosecond {
			t.Errorf("%s: a prefill row added %v of host time; its %d prefill-only "+
				"launches at %v each are %v", c.scenario, got, want, per,
				time.Duration(want)*per)
		}
	}
}

// THE EAGER COST CAN BE A MODEL'S OWN. With no graph DeepSeek-V4.1 pays Python dispatch per
// op, since vLLM runs it uncompiled (vllm/config/vllm.py:120), so the registry may scope the
// eager constant to the model. A model-scoped entry must win over the part-wide one whatever
// the file or set order, under either name; one scoped to another model must not apply; and
// host_launch_eager_per_kernel, the name that says what the value is, wins over the
// per-layer name at equal scope. The per-launch cost is read through the step: under NONE a
// decode step's host time is launches x (eager + dispatch) plus nothing else.
func TestAModelScopedEagerCostWinsWhateverItsOrder(t *testing.T) {
	entry := func(name string, v float64, model string) coefficient.Entry {
		e := coefficient.Entry{Name: name, Value: v, Units: "us_per_transfer",
			Method: "assumed", Rationale: "test", Scope: coefficient.Scope{
				Hardware: []string{"h200"}}}
		if model != "" {
			e.Scope.Model = []string{model}
		}
		return e
	}
	eager := func(before bool, extra ...coefficient.Entry) time.Duration {
		t.Helper()
		in := fixtureInputs(t, v41)
		in.Deployment.Pools[0].Engine.CUDAGraphMode = "NONE"
		set := &coefficient.Set{Kind: "CoefficientSet", Name: "test-eager",
			Coefficients: extra}
		switch {
		case len(extra) == 0:
		case before:
			in.Coefficients = append([]*coefficient.Set{set}, in.Coefficients...)
		default:
			in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...), set)
		}
		k := build(t, in)
		b := decodeBatch(1, 6, 90000)
		host := k.StepTime(b).PerResource[kernel.ResourceHost]
		_, draft := k.priceDraft(b, &stepShape{}, &[numResources]float64{})
		n := time.Duration(k.stepKernels(b) + draft)
		return host/n - k.launchPerKernel
	}
	us := func(v float64) time.Duration { return time.Duration(v * float64(time.Microsecond)) }
	generic := eager(false)
	const m = "deepseek-v4.1-flash"
	for _, c := range []struct {
		name   string
		before bool
		extra  []coefficient.Entry
		want   time.Duration
	}{
		{"per-layer, model-scoped, listed first", true,
			[]coefficient.Entry{entry("host_launch_eager_per_layer", 93.2, m)}, us(93.2)},
		{"per-layer, model-scoped, listed last", false,
			[]coefficient.Entry{entry("host_launch_eager_per_layer", 93.2, m)}, us(93.2)},
		{"per-layer, another model", false,
			[]coefficient.Entry{entry("host_launch_eager_per_layer", 93.2, "kimi-k3")}, generic},
		{"per-kernel, model-scoped, listed first", true,
			[]coefficient.Entry{entry("host_launch_eager_per_kernel", 91, m)}, us(91)},
		{"per-kernel generic beside per-layer generic", false,
			[]coefficient.Entry{entry("host_launch_eager_per_kernel", 11, "")}, us(11)},
		{"per-layer model-scoped beats per-kernel generic", false, []coefficient.Entry{
			entry("host_launch_eager_per_kernel", 11, ""),
			entry("host_launch_eager_per_layer", 93.2, m)}, us(93.2)},
	} {
		if got := eager(c.before, c.extra...); got-c.want > time.Nanosecond ||
			c.want-got > time.Nanosecond {
			t.Errorf("%s: eager cost per launch %v, want %v", c.name, got, c.want)
		}
	}
}
