package latencykernel

import (
	"fmt"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// opaqueRules is a release's rules behind the rules.EngineRules interface alone: every method
// forwards to the wrapped rules, but the value is not a *rules.Pack. It is how a caller whose
// rules are not a blis-schemas pack (another engine model's, or a pack wrapped to log or
// override one fact) reaches the kernel.
type opaqueRules struct{ resolve.EngineRules }

// A kernel reads a release's rules only through the interface, so rules that answer every
// question the same way price every committed fixture the same way, to the nanosecond and to
// the last Provenance entry, whatever their concrete type.
//
// This is the regression the four type assertions to *rules.Pack had: for any other rules
// value the kernel took the async-compatible draft list, Engram's cpu_offload default, the
// release's decode-context backend names and its mega-MoE backends as absent, so the same
// release priced differently (and recorded assumptions it need not) when it arrived wrapped.
func TestRulesPriceAlikeWhateverTheirConcreteType(t *testing.T) {
	fixtures := fixturePaths(t)
	if len(fixtures) < 9 {
		t.Fatalf("only %d fixtures; the comparison would check almost nothing", len(fixtures))
	}
	compared := 0
	for _, path := range fixtures {
		name := strings.TrimPrefix(path, "testdata/")
		t.Run(name, func(t *testing.T) {
			in := fixtureInputs(t, name)
			viaPack, packErr := New(in)
			in.Rules = opaqueRules{in.Rules}
			viaOpaque, opaqueErr := New(in)
			if (packErr == nil) != (opaqueErr == nil) {
				t.Fatalf("the pack built a kernel (err %v) and the same rules wrapped did not "+
					"(err %v), or the reverse", packErr, opaqueErr)
			}
			if packErr != nil {
				if packErr.Error() != opaqueErr.Error() {
					t.Errorf("refused differently: %v, then %v", packErr, opaqueErr)
				}
				return
			}
			compared++
			if got, want := viaOpaque.Provenance(), viaPack.Provenance(); !reflect.DeepEqual(got, want) {
				t.Errorf("Provenance differs:\nwrapped %v\npack    %v", got, want)
			}
			if got, want := viaOpaque.Resolved(), viaPack.Resolved(); !reflect.DeepEqual(got, want) {
				t.Errorf("Resolved differs:\nwrapped %+v\npack    %+v", got, want)
			}
			if got, want := viaOpaque.FixedBytes(), viaPack.FixedBytes(); got != want {
				t.Errorf("FixedBytes differs: wrapped %+v, pack %+v", got, want)
			}
			for _, b := range []struct{ reqs, query, ctx int }{{1, 1, 4096}, {64, 1, 8192}, {4, 512, 4096}} {
				batch := decodeBatch(b.reqs, b.query, b.ctx)
				if got, want := viaOpaque.StepTime(batch), viaPack.StepTime(batch); !reflect.DeepEqual(got, want) {
					t.Errorf("StepTime(%d reqs x %d over %d) differs: wrapped %+v, pack %+v",
						b.reqs, b.query, b.ctx, got, want)
				}
			}
		})
	}
	if compared == 0 {
		t.Fatal("no fixture built a kernel, so nothing was compared")
	}
}

// Each of the four facts the kernel used to read by type-asserting to *rules.Pack moves an
// observable when it is skipped: a stated true async scheduling with a draft method the
// release refuses is refused, a DCP backend the release cannot check is disclosed as an
// assumption, and mega-MoE without expert parallelism is refused. Each case is priced through
// a pack and through the same pack wrapped, and the observable must be the one the pack gives.
func TestEachRulesFactIsReadThroughTheInterface(t *testing.T) {
	refused := func(err error) string {
		if err == nil {
			return "built"
		}
		return "refused: " + err.Error()
	}
	cases := []struct {
		name string
		// observe builds the case and returns what the fact decides.
		observe func(t *testing.T, wrap bool) string
		// want is what a *rules.Pack decides; a skipped fact gives something else.
		want string
	}{
		{"async-compatible draft methods", func(t *testing.T, wrap bool) string {
			in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
			p := &in.Deployment.Pools[in.PoolIndex]
			yes := true
			p.Engine.AsyncScheduling = &yes
			p.Engine.Speculative = &deployment.Speculative{Method: "medusa", NumSpecTokens: 1}
			if wrap {
				in.Rules = opaqueRules{in.Rules}
			}
			_, err := New(in)
			return fmt.Sprint(err != nil && strings.Contains(err.Error(), "async_scheduling is requested"))
		}, "true"},
		{"decode-context backend names", func(t *testing.T, wrap bool) string {
			k, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, func(in *Inputs) {
				in.Deployment.Pools[0].Engine.DCPCommBackend = "a2a"
				if wrap {
					in.Rules = opaqueRules{in.Rules}
				}
			})
			if err != nil {
				return refused(err)
			}
			return fmt.Sprint(hasAssumption(k, "dcp_comm_backend"))
		}, "false"},
		{"a mega-MoE backend", func(t *testing.T, wrap bool) string {
			in := fixtureInputs(t, "deepseek-v41-flash-h200-tp8.yaml")
			p := &in.Deployment.Pools[in.PoolIndex]
			p.Engine.MoEBackend = "deep_gemm_mega_moe"
			p.Parallel.EnableExpertParallel = false
			if wrap {
				in.Rules = opaqueRules{in.Rules}
			}
			_, err := New(in)
			return fmt.Sprint(err != nil && strings.Contains(err.Error(), "needs expert parallelism"))
		}, "true"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.observe(t, false); got != c.want {
				t.Fatalf("through the pack: %s, want %s; the case no longer reaches the fact", got, c.want)
			}
			if got := c.observe(t, true); got != c.want {
				t.Errorf("through the same pack wrapped: %s, want %s, as the pack gives", got, c.want)
			}
		})
	}
}

// hbmEngramRules are a release's rules whose Engram tables default to device memory, the
// value no registered release has, so a kernel that skipped the fact (taking host memory, the
// kernel's own fallback) would be told apart from one that read it.
type hbmEngramRules struct{ resolve.EngineRules }

func (hbmEngramRules) EngramOnHostByDefault() bool { return false }

// Where an Engram table lives when the deployment does not say is the release's default,
// read through the interface: a rules value whose default is device memory puts the V4.1
// tables in HBM, exactly as stating cpu_offload: false does.
func TestEngramResidencyDefaultIsTheRules(t *testing.T) {
	in := fixtureInputs(t, v41)
	if ec := in.Deployment.Pools[0].Engine.EngramConfig; ec != nil && ec.CPUOffload != nil {
		t.Fatal("the fixture states cpu_offload; the default would not be read")
	}
	onHost := build(t, in).FixedBytes().Weights
	in.Rules = hbmEngramRules{in.Rules}
	byDefault := build(t, in).FixedBytes().Weights

	stated := fixtureInputs(t, v41)
	off := false
	stated.Deployment.Pools[0].Engine.EngramConfig = &deployment.EngramConfig{CPUOffload: &off}
	inHBM := build(t, stated).FixedBytes().Weights
	if byDefault != inHBM || byDefault <= onHost {
		t.Errorf("rules defaulting Engram to device memory hold %d bytes; stating cpu_offload: "+
			"false holds %d and host-memory tables %d. The default was not read from the rules",
			byDefault, inHBM, onHost)
	}
}

// Open takes its rules from the caller and looks none up itself: a Repos with no rules source
// is refused, naming what is missing, rather than priced under rules from anywhere else.
func TestOpenRefusesARulesSourceItWasNotGiven(t *testing.T) {
	r := repos()
	r.Rules, r.RulesVersions = nil, nil
	_, err := Open("minimax-m25-h200-ep8.yaml", r)
	if err == nil {
		t.Fatal("Open built a kernel with no rules source; it must not find rules on its own")
	}
	for _, want := range []string{"Repos.Rules", "no engine rules source"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// The rules Open builds with are the caller's: a source that answers for the scenario's
// release with something other than a blis-schemas pack is used, not replaced.
func TestOpenUsesTheCallersRules(t *testing.T) {
	r := repos()
	inner := r.Rules
	var asked []string
	r.Rules = func(version string) rules.EngineRules {
		asked = append(asked, version)
		return opaqueRules{inner(version)}
	}
	in, err := OpenInputs("minimax-m25-h200-ep8.yaml", r, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := in.Rules.(opaqueRules); !ok {
		t.Errorf("Inputs.Rules is a %T, not the caller's rules", in.Rules)
	}
	if !reflect.DeepEqual(asked, []string{in.Scenario.EngineVersion}) {
		t.Errorf("the rules source was asked for %v, want exactly the scenario's %q", asked, in.Scenario.EngineVersion)
	}
}

// RulesFrom keeps "no rules for this release" as no rules: a nil *rules.Pack must not become
// a non-nil interface that reads as rules that exist.
func TestRulesFromKeepsAMissingReleaseMissing(t *testing.T) {
	lookup := RulesFrom(func(v string) *rules.Pack {
		if v == "known" {
			return &rules.Pack{Version: "known"}
		}
		return nil
	})
	if r := lookup("unknown"); r != nil {
		t.Errorf("an unknown release gave non-nil rules %#v", r)
	}
	if r := lookup("known"); r == nil || r.EngineVersion() != "known" {
		t.Errorf("a known release gave %v", r)
	}
}

// The library takes its rules from its caller and imports no engine model: only this
// repository's commands, its documentation generator and its tests may reach the vLLM packs
// (through internal/harness). A library import would put vLLM's rules back inside the cost
// model, which is what taking them from the caller removes.
func TestTheLibraryImportsNoEngineModel(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", ".", "./internal/price", "./internal/resolve").Output()
	if err != nil {
		t.Fatalf("go list -deps: %v", err)
	}
	for _, dep := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(dep, "github.com/inference-sim/blis-vllm") {
			t.Errorf("the library depends on %s; it must take engine rules from its caller", dep)
		}
	}
}
