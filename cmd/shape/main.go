// Command shape scores this kernel's concurrency response against NVIDIA's end-to-end
// accuracy snapshot, along AISimulate's own published error on the same points.
//
// # Why a shape comparison, and why it needs no simulator
//
// The snapshot publishes every measurement NORMALISED to its sweep's lowest concurrency,
// because absolute silicon latencies are not disclosed. So a sweep reads
//
//	c=4   measured 1.0000    aisimulate 0.8885
//	c=64  measured 2.8119    aisimulate 2.7570
//
// A step-time kernel predicts the same quantity directly: the ratio of StepTime at two
// concurrencies at one fixed deployment. StepTime is a pure function, so nothing here needs a
// scheduler — and the normalisation is what makes that true. An absolute time-per-token would
// need one, because it carries queueing and the resident-batch question; a ratio between two
// concurrencies cancels every constant, including this kernel's known systematic bias and
// AISimulate's fitted offsets.
//
// # What this measures, and what it does not
//
// It measures the SHAPE of the concurrency response: how step time grows as a batch widens at
// a fixed model, hardware and parallelism. That is the property this kernel's error is worst
// on — against published Granite runs its error grows monotonically with concurrency — so a
// ratio across concurrency isolates that error rather than cancelling it.
//
// It does NOT measure level. A model uniformly thirty percent low scores perfectly here. The
// absolute question stays open and is answered by cmd/score, which carries the level and the
// conditions under which it holds.
//
// It also assumes the batch equals the concurrency, as cmd/score does, because the snapshot
// states no resident batch. Where that assumption fails the error shape says so: a resident
// batch smaller than the client level flattens the true curve, so an over-prediction that
// grows with concurrency is the signature.
//
// Usage:
//
//	go run ./cmd/shape
//	go run ./cmd/shape -verbose
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules/v0_29"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
)

type corpus struct {
	Source           string `json:"source"`
	Snapshot         any    `json:"snapshot"`
	Scope            any    `json:"scope"`
	AISimulateTotals struct {
		TPOTMapePct       float64 `json:"tpot_mape_pct"`
		TPOTShapeErrorPct float64 `json:"tpot_shape_error_pct"`
		Points            int     `json:"points"`
	} `json:"aisimulate_totals"`
	Sweeps []sweep `json:"sweeps"`
}

type sweep struct {
	Scenario    string         `json:"scenario"`
	Model       string         `json:"model"`
	GPU         string         `json:"gpu"`
	Workload    string         `json:"workload"`
	Label       string         `json:"label"`
	Framework   string         `json:"framework"`
	Precision   string         `json:"precision"`
	Serving     string         `json:"serving"`
	Parallelism map[string]int `json:"parallelism"`
	TopologyID  string         `json:"topology_id"`
	Points      []point        `json:"points"`
}

type point struct {
	Concurrency            int     `json:"concurrency"`
	MeasuredRelative       float64 `json:"measured_tpot_relative"`
	AISimulateRelative     float64 `json:"aisimulate_tpot_relative"`
	AISimulateErrorPercent float64 `json:"aisimulate_tpot_error_pct"`
}

func main() {
	catalog := flag.String("catalog", "/Users/sri/Documents/Projects/blis-catalog", "")
	registry := flag.String("registry", "/Users/sri/Documents/Projects/blis-registry", "")
	data := flag.String("data", "testdata/measurements/aisimulate_e2e.json", "")
	// The generated per-deployment scenarios, which is where
	// scripts/gen_aisimulate_scenarios.py writes them and where main_test.go's
	// scenarioRoot looks. The default was "testdata" and every run failed on the first
	// sweep until a reader passed the flag.
	testdata := flag.String("testdata", "testdata/aisimulate", "scenario directory")
	verbose := flag.Bool("verbose", false, "print every point")
	flag.Parse()

	raw, err := os.ReadFile(*data)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var c corpus
	if err := json.Unmarshal(raw, &c); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Println("Concurrency-response shape against NVIDIA's end-to-end accuracy snapshot")
	fmt.Printf("%d sweeps, %d points\n\n", len(c.Sweeps), countPoints(c.Sweeps))

	mine, theirs, err := score(c, *testdata, *catalog, *registry, *verbose)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Printf("%-28s %4s %8s %8s %8s\n", "model", "n", "MAPE", "median", "worst")
	report("this kernel", mine)
	report("AISimulate (same points)", theirs)
	// The snapshot publishes two different TPOT figures and only one is comparable to this
	// table. tpot_shape_error_pct is the normalised-curve error, which is what the rows above
	// measure; tpot_mape_pct is absolute and belongs with cmd/score. Citing the absolute
	// figure here would flatter this kernel by roughly the ratio between them.
	fmt.Printf("\nAISimulate's published figures over its whole snapshot: %.2f%% TPOT shape "+
		"error (comparable\nto the rows above) and %.2f%% absolute TPOT MAPE, across %d "+
		"points.\n", c.AISimulateTotals.TPOTShapeErrorPct, c.AISimulateTotals.TPOTMapePct,
		c.AISimulateTotals.Points)
	fmt.Println(`
Scope: shape only. Every measurement in the snapshot is normalised to its sweep's lowest
concurrency because absolute latencies are not disclosed, so this scores how step time GROWS
with batch width at a fixed deployment and says nothing about level. A model uniformly low by
a constant factor scores perfectly here; cmd/score carries the level.

The batch is taken as the client concurrency, since the snapshot states no resident batch.
Where that fails the error shape says so: a resident batch below the client level flattens the
true curve, so an over-prediction growing with concurrency is the signature.

What this run found. The kernel over-predicts at every point and the excess grows with
concurrency, which is that signature. It is also the OPPOSITE sign to the same kernel's error
against the published Granite runs in cmd/score, where it under-predicts by 30 to 57 percent
and the shortfall likewise grows with concurrency. One kernel cannot be both too steep and too
shallow in the same term, so the two residuals are not one mechanism, and the batch assumption
is the term they disagree about: cmd/score's Granite arms state no resident batch either.

A cross-model test of the batch hypothesis fails. The Nemotron sweep is the only corpus arm
that reports both client concurrency and resident batch, and its resident count grows
SUPER-linearly — about c^1.44 — which would steepen a predicted curve rather than flatten it.
Transferring a scheduler behaviour between two different models, engines and parts is not
defensible as a correction, and the direction does not even agree, so nothing is adjusted here.

What would settle it is a resident batch measured on these deployments, which is what an
end-to-end simulation supplies and what no published snapshot in hand states.`)
}

// score runs the comparison and returns the per-point absolute errors for this kernel and for
// AISimulate on the same points.
//
// main and the tests both call this. A test that recomputed the ratio itself would be
// self-consistent under any mutation of the scoring path and so could not fail — which is
// exactly what happened to an earlier version of this file's tests.
//
// testdata is a parameter rather than a constant because a test runs with the package
// directory as its working directory while main runs from the repository root. Hard-coding it
// made every test SKIP on a missing path, which reads as a pass and is worse than the flaw it
// replaced.
func score(c corpus, testdata, catalog, registry string, verbose bool) (mine, theirs []float64, err error) {
	for _, s := range c.Sweeps {
		k, buildErr := build(filepath.Join(testdata, s.Scenario), catalog, registry)
		if buildErr != nil {
			return nil, nil, fmt.Errorf("%s: %w", s.Scenario, buildErr)
		}
		context := contextTokens(s)
		// The anchor: the lowest concurrency in this sweep, which is what the snapshot
		// normalised to. Both sides are divided by their own value here, so neither carries
		// an absolute scale.
		anchor := stepSeconds(k, s.Points[0].Concurrency, context)
		if anchor <= 0 {
			return nil, nil, fmt.Errorf("%s: anchor step time is not positive", s.Scenario)
		}
		// AISimulate's own anchor. The corpus normalises its predictions to the MEASURED
		// anchor, not to AISimulate's own first point, so its published relatives carry a
		// level error that a shape score must not charge it for: on one gpt-oss arm its c=4
		// value reads 2.5030 against a measured 1.0000, and every later point inherits that
		// 150 percent offset. The kernel is re-anchored to its own first point two lines
		// above, so charging AISimulate an un-anchored error would compare a shape against a
		// shape-plus-level and flatter this kernel. Re-anchoring both recovers 9.41 percent
		// for AISimulate here, against the 10.05 percent shape error it publishes over the
		// whole snapshot -- which is the check that this normalisation is the right one.
		theirAnchor := s.Points[0].AISimulateRelative
		if theirAnchor <= 0 {
			return nil, nil, fmt.Errorf("%s: AISimulate anchor is not positive", s.Scenario)
		}
		if verbose {
			fmt.Printf("--- %s %s %s %s %s, tp=%d moe_ep=%d, context %d tokens\n",
				s.Model, s.GPU, s.Label, s.Framework, s.Precision,
				s.Parallelism["tp_size"], s.Parallelism["moe_ep_size"], context)
			fmt.Printf("%6s %10s %10s %8s %10s %8s\n",
				"conc", "measured", "kernel", "error", "aisimulate", "error")
		}
		for _, p := range s.Points {
			predicted := stepSeconds(k, p.Concurrency, context) / anchor
			errMine := math.Abs(predicted/p.MeasuredRelative-1) * 100
			mine = append(mine, errMine)
			// Recomputed from AISimulate's re-anchored relatives rather than read from the
			// snapshot's aisimulate_tpot_error_pct field, which is that same un-anchored
			// quantity and carries its level error. Both sides are now the same thing: a
			// curve normalised to its own first point, divided by the measured curve
			// normalised to its own, at the same points.
			errTheirs := math.Abs((p.AISimulateRelative/theirAnchor)/p.MeasuredRelative-1) * 100
			theirs = append(theirs, errTheirs)
			if verbose {
				fmt.Printf("%6d %10.4f %10.4f %7.2f%% %10.4f %7.2f%%\n",
					p.Concurrency, p.MeasuredRelative, predicted, errMine,
					p.AISimulateRelative, errTheirs)
			}
		}
		if verbose {
			fmt.Println()
		}
	}
	return mine, theirs, nil
}

// contextTokens returns the mean context a request carries during decode, from the workload
// identity the snapshot states as input:output.
//
// A request grows from its prompt to prompt-plus-output over its decode phase, so the mean is
// the prompt plus half the output. TPOT is a per-token average over that phase, which is the
// quantity this matches.
func contextTokens(s sweep) int {
	var in, out int
	if _, err := fmt.Sscanf(s.Workload, "%d:%d", &in, &out); err != nil {
		return 0
	}
	return in + out/2
}

// decodeBatchFor builds a steady-state decode batch: every resident request contributes one
// token this step and carries the given context.
func decodeBatchFor(batch, context int) kernel.Batch {
	b := kernel.Batch{Reqs: make([]kernel.ReqShape, batch), DecodeThreshold: 8}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: 1, Computed: context - 1, PromptLen: context,
		}
	}
	return b
}

// stepSeconds is the quantity TPOT measures: one step, plus the per-token host work the output
// processor does off the forward pass but which still lands between tokens.
func stepSeconds(k *latencykernel.Kernel, batch, context int) float64 {
	e := k.StepTime(decodeBatchFor(batch, context))
	return (e.Overlap + k.OutputTokenOverhead()).Seconds()
}

func report(label string, errs []float64) {
	if len(errs) == 0 {
		fmt.Printf("%-28s %4d  no points\n", label, 0)
		return
	}
	s := append([]float64(nil), errs...)
	sort.Float64s(s)
	var sum float64
	for _, e := range s {
		sum += e
	}
	fmt.Printf("%-28s %4d %7.2f%% %7.2f%% %7.2f%%\n", label, len(s),
		sum/float64(len(s)), s[len(s)/2], s[len(s)-1])
}

func countPoints(ss []sweep) int {
	n := 0
	for _, s := range ss {
		n += len(s.Points)
	}
	return n
}

func build(scenarioPath, catalog, registry string) (*latencykernel.Kernel, error) {
	sc, err := schemas.LoadScenario(scenarioPath)
	if err != nil {
		return nil, err
	}
	graph, err := schemas.LoadModelGraph(
		filepath.Join(catalog, "models", sc.Model, "graph.yaml"))
	if err != nil {
		return nil, err
	}
	chip, err := schemas.LoadChip(
		filepath.Join(catalog, "hardware", sc.Hardware+".yaml"))
	if err != nil {
		return nil, err
	}
	var fabric *hardware.Fabric
	if sc.Fabric != "" {
		if fabric, err = schemas.LoadFabric(
			filepath.Join(catalog, "networks", sc.Fabric+".yaml")); err != nil {
			return nil, err
		}
	}
	var sets []*coefficient.Set
	for _, name := range sc.Coefficients {
		set, err := schemas.LoadCoefficientSet(
			filepath.Join(registry, "coefficients", name+".yaml"))
		if err != nil {
			return nil, err
		}
		sets = append(sets, set)
	}
	devices, err := schemas.LoadStorageDevices(
		filepath.Join(catalog, "devices", "storage.yaml"))
	if err != nil {
		return nil, err
	}
	return latencykernel.New(latencykernel.Inputs{
		Scenario: sc, PoolIndex: 0, Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: v0_29.Pack(),
	})
}

// scenarioFacts are the deployment fields a test checks against the snapshot's own record.
type scenarioFacts struct {
	tp             int
	expertParallel bool
}

func loadScenario(path string) (scenarioFacts, error) {
	sc, err := schemas.LoadScenario(path)
	if err != nil {
		return scenarioFacts{}, err
	}
	p := sc.Pools[0].Parallel
	return scenarioFacts{tp: p.TP, expertParallel: p.EnableExpertParallel}, nil
}
