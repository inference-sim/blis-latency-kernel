package latencykernel

import (
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
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
		return hostBusy(k, b)
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
		got := hostBusy(k, mixed) -
			hostBusy(k, dec)
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
		host := hostBusy(k, b)
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

// THE DSPARK DRAFTER RUNS UNDER ITS OWN GRAPHS. Its manager captures FULL graphs only, and
// only when the target's decode mode is FULL ("PIECEWISE cudagraphs are not supported for
// dflash", vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:126-150); otherwise it runs
// eagerly. A replay covers the context rows of one verify per request (:309-311), so a step
// with a prefill row stores the rest of its context eagerly first (:480-482). Each case is
// the drafter's host share: the step's host time less the same step's target-only host
// time (the target's launches, dispatch included, do not depend on the drafter).
func TestTheDrafterRunsUnderItsOwnGraphs(t *testing.T) {
	size := 64
	draftShare := func(mode string, b kernel.Batch) (got time.Duration, k *Kernel) {
		t.Helper()
		in := fixtureInputs(t, v41)
		e := &in.Deployment.Pools[0].Engine
		e.CUDAGraphMode, e.MaxCUDAGraphCaptureSize = mode, &size
		k = build(t, in)
		_, kernels := k.priceDraft(b, &stepShape{}, &[numResources]float64{})
		host := hostBusy(k, b)
		target := k.targetHost(b, 0) + seconds(k.launchSeconds(b, kernels))
		return host - target, k
	}
	d := func(k *Kernel) *price.DraftPlan { return k.plan.Draft }
	// A decode of 4 requests drafts 20 tokens: one replay under FULL_AND_PIECEWISE, every
	// launch eager under PIECEWISE, whose decode mode is not FULL.
	dec := decodeBatch(4, 6, 90000)
	if got, k := draftShare("FULL_AND_PIECEWISE", dec); got != k.replayPerStep {
		t.Errorf("FULL_AND_PIECEWISE decode: drafter host %v, want one replay %v", got,
			k.replayPerStep)
	}
	got, k := draftShare("PIECEWISE", dec)
	all := d(k).Kernels + d(k).ContextKernels + 5*d(k).SequentialKernels
	if want := time.Duration(all) * k.launchEager; got != want {
		t.Errorf("PIECEWISE decode: drafter host %v, want %d eager launches, %v", got, all,
			want)
	}
	// A decode plus one prefill chunk that completes its prompt: the draft still replays,
	// and its context kernels beyond the graph's rows launch eagerly.
	mixed := dec
	mixed.Reqs = append(append([]kernel.ReqShape{}, dec.Reqs...),
		kernel.ReqShape{Scheduled: 1808, Computed: 88192, PromptLen: 90000})
	got, k = draftShare("FULL_AND_PIECEWISE", mixed)
	if want := k.replayPerStep + time.Duration(d(k).ContextKernels)*k.launchEager; got != want {
		t.Errorf("mixed step: drafter host %v, want a replay plus %d eager context "+
			"launches, %v", got, d(k).ContextKernels, want)
	}
}

// vendoredHostEntries replicates blis-registry c45c71c's cost-model-host-overheads.yaml order
// for the entries DeepSeek-V4.1 scopes (docs/COEFF_CONTRACT.md v2.1): each model-scoped entry
// follows its part-wide namesake in the file, and the eager constant is derived on h200
// (assumed) and copied to the other eight parts (copied).
func vendoredHostEntries() []coefficient.Entry {
	all := []string{"a100-80", "a100-sxm", "b200", "b300", "gb200-nvl72", "gb300", "h100",
		"h200", "l40s"}
	others := []string{"a100-80", "a100-sxm", "b200", "b300", "gb200-nvl72", "gb300", "h100",
		"l40s"}
	m := []string{"deepseek-v4.1-flash"}
	e := func(name string, v float64, units, method string, hw, model []string) coefficient.Entry {
		x := coefficient.Entry{Name: name, Value: v, Units: vocab.Unit(units),
			Method: vocab.Method(method), Scope: coefficient.Scope{Hardware: hw, Model: model}}
		switch method {
		case "measured":
			x.Sources = []coefficient.Source{{Kind: "discussion", Cite: "test", Role: "primary"}}
		default:
			x.Rationale = "test"
		}
		if method == "copied" {
			x.CopiedFrom = "hardware=h200,model=deepseek-v4.1-flash"
		}
		return x
	}
	return []coefficient.Entry{
		e("host_admission_per_request", 40000, "us_per_request", "assumed", all, nil),
		e("host_admission_per_request", 0, "us_per_request", "not_charged", all, m),
		e("host_admission_per_token", 0.35, "us_per_token", "assumed", all, nil),
		e("host_admission_per_token", 2.67, "us_per_token", "measured", all, m),
		e("host_launch_eager_per_layer", 7, "us_per_layer", "assumed", all, nil),
		e("host_launch_eager_per_kernel", 93.8, "us_per_transfer", "assumed", []string{"h200"}, m),
		e("host_launch_eager_per_kernel", 93.8, "us_per_transfer", "copied", others, m),
	}
}

// THE TWO RESOLUTION RULES AGREE ON THE VENDORED SET, AND A COPY SAYS IT IS ONE. The registry
// resolves last-in-file wins with model-scoped placed last; the kernel never lets a part-wide
// entry displace a model-scoped one. On the vendored order the two must pick the same entry
// for every name on every part, for this model and another. And Provenance must carry the
// resolved entry's method and where a copy came from, so a report can show h100's eager
// constant as copied from h200.
func TestTheVendoredHostSetResolvesTheSameUnderBothRules(t *testing.T) {
	set := &coefficient.Set{Kind: "CoefficientSet", Name: "cost-model-host-overheads",
		Coefficients: vendoredHostEntries()}
	for _, chip := range []string{"h100", "h200", "b200"} {
		for _, model := range []string{"deepseek-v4.1-flash", "kimi-k3"} {
			scope := resolve.Scope{Hardware: chip, Model: model}
			got, err := resolve.Load([]*coefficient.Set{set}, scope)
			if err != nil {
				t.Fatal(err)
			}
			last := map[string]coefficient.Entry{}
			for _, e := range set.Coefficients {
				if scope.Admits(e.Scope) {
					last[e.Name] = e
				}
			}
			for name, want := range last {
				r, ok := got.Entry(name)
				if !ok || r.Entry.Value != want.Value || r.Entry.Method != want.Method ||
					r.Entry.CopiedFrom != want.CopiedFrom {
					t.Errorf("%s/%s %s: kernel resolved %+v, last-in-file gives %+v", chip,
						model, name, r.Entry, want)
				}
			}
		}
	}
	for _, c := range []struct{ chip, method, scope string }{
		{"h200", "assumed", "hardware: h200, model: deepseek-v4.1-flash"},
		{"h100", "copied", "hardware: h100, model: deepseek-v4.1-flash; copied from " +
			"hardware=h200,model=deepseek-v4.1-flash"},
	} {
		in := fixtureInputs(t, v41)
		if c.chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, c.chip), c.chip
		}
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...), set)
		k := build(t, in)
		if want := time.Duration(93.8 * float64(time.Microsecond)); k.launchEager != want {
			t.Errorf("%s: eager launch %v, want the model's %v", c.chip, k.launchEager, want)
		}
		found := false
		for _, o := range k.Provenance() {
			if o.Name == "host_launch_eager_per_kernel" {
				found = true
				if o.Method != c.method || o.Scope != c.scope {
					t.Errorf("%s: Provenance records %s %q, want %s %q", c.chip, o.Method,
						o.Scope, c.method, c.scope)
				}
			}
		}
		if !found {
			t.Errorf("%s: host_launch_eager_per_kernel is not in Provenance", c.chip)
		}
	}
}

// hostBusy is the host's whole busy time over b: its launches, replays and dispatch gaps,
// before any eager launch hides behind device work. PerResource reports only the part the
// step pays.
func hostBusy(k *Kernel, b kernel.Batch) time.Duration {
	busy, _ := hostBusySplit(k, b)
	return busy
}

// hostBusySplit is hostBusy with its eager launches, the part that can hide.
func hostBusySplit(k *Kernel, b kernel.Batch) (busy, eager time.Duration) {
	draft := 0
	if k.speculating && len(b.Reqs) > 0 {
		_, draft = k.priceDraft(b, &stepShape{}, &[numResources]float64{})
	}
	host, eager := k.hostSplit(b, draft)
	return host + seconds(k.launchSeconds(b, draft)), eager
}

// eagerKernel is the V4.1 fixture on chip under mode with the vendored host entries, its
// model-scoped eager constant set to eager microseconds, and a 512-token capture ceiling.
func eagerKernel(t *testing.T, chip, mode string, spec bool, eager float64) *Kernel {
	t.Helper()
	in := fixtureInputs(t, v41)
	if chip != "h200" {
		in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
	}
	e := &in.Deployment.Pools[0].Engine
	e.CUDAGraphMode = mode
	if !spec {
		e.Speculative = nil
	}
	capture := 512
	e.MaxCUDAGraphCaptureSize = &capture
	entries := vendoredHostEntries()
	for i := range entries {
		if entries[i].Name == "host_launch_eager_per_kernel" {
			entries[i].Value = eager
		}
	}
	in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
		&coefficient.Set{Kind: "CoefficientSet", Name: "test-eager", Coefficients: entries})
	return build(t, in)
}

// EAGER LAUNCHES HIDE BEHIND DEVICE WORK, AND ONLY THERE. vLLM v0.31.0 enqueues an eager
// step's kernels asynchronously, so a step costs about max(launches, device). A
// single-request decode step under NONE is launch-bound: it costs its launches x (eager +
// dispatch) whatever its device time, so the eager constant backed out of the h200 TP8 decode
// anchor, 96.56 ms over 972 launches, reproduces it. A large prefill chunk past the capture
// ceiling is device-bound: its eager launches hide behind the chunk's device time and add
// almost nothing, where an additive charge would add about 94 ms.
func TestEagerLaunchesHideBehindDeviceWork(t *testing.T) {
	const anchor = 96.56e-3
	dec := decodeBatch(1, 1, 90000)
	probe := eagerKernel(t, "h200", "NONE", false, 90)
	n := probe.stepKernels(dec)
	eager := (anchor/float64(n) - probe.launchPerKernel.Seconds()) * 1e6
	k := eagerKernel(t, "h200", "NONE", false, eager)
	est := k.StepTime(dec)
	if d := est.NoOverlap.Seconds()/anchor - 1; math.Abs(d) > 1e-5 {
		t.Errorf("decode anchor: step %v at %.2f us a launch over %d launches, want %.2f ms",
			est.NoOverlap, eager, n, anchor*1e3)
	}
	if busy := hostBusy(k, dec); est.NoOverlap != busy {
		t.Errorf("decode: step %v, want the launch-bound host time %v", est.NoOverlap, busy)
	}

	k = eagerKernel(t, "h100", "FULL_AND_PIECEWISE", true, eager)
	chunk := kernel.Batch{DecodeThreshold: 6, Reqs: []kernel.ReqShape{
		{Scheduled: 8192, Computed: 81808, PromptLen: 90000}}}
	est = k.StepTime(chunk)
	busy, launches := hostBusySplit(k, chunk)
	if launches < 50*time.Millisecond {
		t.Fatalf("the chunk launches %v eagerly; it should be past the ceiling", launches)
	}
	if exposed := est.PerResource[kernel.ResourceHost]; exposed > busy-launches+time.Microsecond {
		t.Errorf("chunk: %v of host time is exposed of %v, %v of it eager launches that "+
			"the chunk's device time should hide", exposed, busy, launches)
	}
	if est.Overlap > est.NoOverlap {
		t.Errorf("chunk: Overlap %v above NoOverlap %v", est.Overlap, est.NoOverlap)
	}
}
