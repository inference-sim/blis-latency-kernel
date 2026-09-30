package resolve

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// A stub rules pack. The tests below are about the resolver's behaviour, so the engine
// facts are stated here rather than read from a real pack: a test that failed because a
// pack changed would not be testing the resolver.
type stubRules struct {
	spBackends   map[string]bool
	customWidths map[int]bool
}

func newStubRules() stubRules {
	return stubRules{
		spBackends:   map[string]bool{"allgather_reducescatter": true, "deepep_low_latency": true},
		customWidths: map[int]bool{2: true, 4: true, 6: true, 8: true, 16: true},
	}
}

func (r stubRules) SequenceParallelMoE(backend string, ep bool, tp, dp int) bool {
	return ep && tp > 1 && dp > 1 && r.spBackends[backend]
}
func (r stubRules) CustomAllReduceSupportsWidth(tp int) bool { return r.customWidths[tp] }
func (r stubRules) TritonFetchWithholdsSMs(pageBytes int) int {
	if pageBytes > 0 && pageBytes < 28*1024 {
		return 12
	}
	return 0
}
func (r stubRules) DBOEngages(enabled bool, tokens int, uniform bool) bool {
	if !enabled {
		return false
	}
	if uniform {
		return tokens >= 32
	}
	return tokens >= 512
}

func scenarioWith(nodes, gpusPerNode int, pools ...scenario.Pool) *scenario.Scenario {
	return &scenario.Scenario{
		Kind: "Scenario", Name: "t", Model: "m", Hardware: "h",
		Coefficients: []string{"c"}, EngineVersion: "0.29.0",
		Cluster: scenario.Cluster{Nodes: nodes, GPUsPerNode: gpusPerNode},
		Pools:   pools,
	}
}

func pool(pl scenario.Parallelism, e scenario.Engine) scenario.Pool {
	return scenario.Pool{Role: scenario.RoleColocated, Nodes: 1, Parallel: pl, Engine: e}
}

// TestEmitsAnswersEveryConditionTheSchemaAllows is the property that makes the enum safe.
// The schema's condition set and this switch must be the same closed set: a condition a
// graph can state and the resolver cannot answer would silently drop a node, removing a
// cost with nothing reporting it.
func TestEmitsAnswersEveryConditionTheSchemaAllows(t *testing.T) {
	all := []model.EmitCondition{
		model.EmitAlways,
		model.EmitTensorParallel,
		model.EmitExpertParallel,
		model.EmitTensorParallelUnlessSequenceParallelMoE,
	}
	for _, c := range all {
		if !c.Valid() {
			t.Errorf("%q is in the test's list but the schema rejects it", c)
		}
		if !Recognizes(c) {
			t.Errorf("the schema allows %q but the resolver does not recognize it", c)
		}
	}
	// And the reverse: a condition the resolver would answer must be one the schema
	// allows, or the enum has drifted.
	if Recognizes("sp_moe_only") {
		t.Error("the resolver recognizes a condition the schema does not define")
	}
}

// TestEmitsUnderEachLayout states the behaviour a cost model depends on: which collectives
// exist at which parallelism. Table-driven over layouts rather than over internal state,
// so the test says what a deployment gets rather than how the resolver stores it.
func TestEmitsUnderEachLayout(t *testing.T) {
	cases := []struct {
		what   string
		layout Layout
		want   map[model.EmitCondition]bool
	}{
		{
			what:   "single rank: no collective runs",
			layout: Layout{TP: 1, ExpertWidth: 1},
			want: map[model.EmitCondition]bool{
				model.EmitAlways:                                  true,
				model.EmitTensorParallel:                          false,
				model.EmitExpertParallel:                          false,
				model.EmitTensorParallelUnlessSequenceParallelMoE: false,
			},
		},
		{
			what:   "tensor parallel only: the reductions run, the dispatch does not",
			layout: Layout{TP: 8, ExpertWidth: 1},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallel:                          true,
				model.EmitExpertParallel:                          false,
				model.EmitTensorParallelUnlessSequenceParallelMoE: true,
			},
		},
		{
			what:   "expert parallel only: the dispatch runs, the reductions do not",
			layout: Layout{TP: 1, ExpertWidth: 16},
			want: map[model.EmitCondition]bool{
				model.EmitTensorParallel:                          false,
				model.EmitExpertParallel:                          true,
				model.EmitTensorParallelUnlessSequenceParallelMoE: false,
			},
		},
		{
			what:   "sequence-parallel MoE replaces the MLP reduction",
			layout: Layout{TP: 2, ExpertWidth: 16, SequenceParallelMoE: true},
			want: map[model.EmitCondition]bool{
				// The attention reduction still runs: sequence-parallel MoE changes the
				// MLP path only.
				model.EmitTensorParallel:                          true,
				model.EmitExpertParallel:                          true,
				model.EmitTensorParallelUnlessSequenceParallelMoE: false,
			},
		},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			for cond, want := range c.want {
				if got := c.layout.Emits(cond); got != want {
					t.Errorf("Emits(%q) = %v, want %v", cond, got, want)
				}
			}
		})
	}
}

// TestUnrecognizedConditionDoesNotSilentlyDropANode: returning false for an unknown
// condition would remove a cost invisibly, so the resolver must be askable about what it
// can answer.
func TestUnrecognizedConditionDoesNotSilentlyDropANode(t *testing.T) {
	l := Layout{TP: 8, ExpertWidth: 16}
	if l.Emits("invented_condition") {
		t.Error("an unknown condition should not emit a node")
	}
	if Recognizes("invented_condition") {
		t.Error("Recognizes must report that the condition is unknown, so a caller can " +
			"reject the graph rather than price it with a node missing")
	}
}

func TestNodesSpannedFollowsTheWidestGroup(t *testing.T) {
	cases := []struct {
		what        string
		gpusPerNode int
		pl          scenario.Parallelism
		rack        int
		rackDomain  bool
		want        int
	}{
		{"tp 8 on 8-GPU nodes stays on one node", 8,
			scenario.Parallelism{TP: 8, PP: 1, DP: 1}, 0, false, 1},
		{"tp 16 on 8-GPU nodes spans two", 8,
			scenario.Parallelism{TP: 16, PP: 1, DP: 1}, 0, false, 2},
		{"expert width is the widest group", 8,
			scenario.Parallelism{TP: 1, PP: 1, DP: 16, EnableExpertParallel: true}, 0, false, 2},
		{"multi-node NVLink makes a rack one domain", 4,
			scenario.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}, 72, true, 1},
		{"a rack that is not one domain still spans nodes", 4,
			scenario.Parallelism{TP: 1, PP: 1, DP: 72, EnableExpertParallel: true}, 72, false, 18},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			s := scenarioWith(1, c.gpusPerNode, pool(c.pl, scenario.Engine{}))
			s.Cluster.GPUsPerRack = c.rack
			l, err := ResolveLayout(s, s.Pools[0],
				Fabric{RackIsOneDomain: c.rackDomain}, newStubRules())
			if err != nil {
				t.Fatalf("ResolveLayout: %v", err)
			}
			if l.NodesSpanned != c.want {
				t.Errorf("NodesSpanned = %d, want %d", l.NodesSpanned, c.want)
			}
		})
	}
}

// TestCustomAllReduceRequestIsResolvedNotObeyed: the scenario states a request, and the
// layout may decline it. A prediction that assumed the request was honoured would charge
// SM occupancy that does not occur and omit NIC traffic that does.
func TestCustomAllReduceRequestIsResolvedNotObeyed(t *testing.T) {
	no := false
	cases := []struct {
		what        string
		gpusPerNode int
		tp          int
		rackDomain  bool
		wantCustom  bool
		wantReason  string
	}{
		{"tp 8 within a node: honoured", 8, 8, false, true, ""},
		{"tp 3 is an unsupported width: declined", 8, 3, false, false, "rank counts"},
		{"tp 16 spans nodes without multi-node NVLink: declined", 8, 16, false, false,
			"spans more than"},
		{"tp 16 spans nodes with multi-node NVLink: honoured", 8, 16, true, true, ""},
	}
	for _, c := range cases {
		t.Run(c.what, func(t *testing.T) {
			s := scenarioWith(1, c.gpusPerNode, pool(
				scenario.Parallelism{TP: c.tp, PP: 1, DP: 1},
				scenario.Engine{DisableCustomAllReduce: &no}))
			l, err := ResolveLayout(s, s.Pools[0],
				Fabric{RackIsOneDomain: c.rackDomain}, newStubRules())
			if err != nil {
				t.Fatalf("ResolveLayout: %v", err)
			}
			if l.CustomAllReduce != c.wantCustom {
				t.Fatalf("CustomAllReduce = %v, want %v", l.CustomAllReduce, c.wantCustom)
			}
			if c.wantReason == "" {
				if len(l.Overrides) != 0 {
					t.Errorf("an honoured request should record no override, got %+v", l.Overrides)
				}
				return
			}
			if len(l.Overrides) == 0 {
				t.Fatal("a declined request must record why")
			}
			if !strings.Contains(l.Overrides[0].Reason, c.wantReason) {
				t.Errorf("reason %q does not mention %q", l.Overrides[0].Reason, c.wantReason)
			}
		})
	}
}

func TestFabricRatio(t *testing.T) {
	cases := []struct {
		intra, inter, want float64
	}{
		{450, 50, 9},  // NVLink 4 against InfiniBand NDR
		{450, 25, 18}, // the same chip behind RoCE
		{900, 900, 1}, // multi-node NVLink: no step at the boundary
		{450, 0, 1},   // no fabric stated: single node, nothing crosses
		{0, 50, 1},    // no chip bandwidth: not a division by an unknown
	}
	for _, c := range cases {
		f := Fabric{IntraNodeBwGBps: c.intra, InterNodeBwGBps: c.inter}
		if got := f.Ratio(); got != c.want {
			t.Errorf("Ratio(%v, %v) = %v, want %v", c.intra, c.inter, got, c.want)
		}
	}
}
