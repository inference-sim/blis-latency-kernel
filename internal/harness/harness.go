// Package harness holds what this repository's scoring and reporting commands share: opening a
// kernel from a scenario file, and building the batch shapes a comparison prices.
//
// It is internal and it is not part of the cost model. The kernel package implements the
// interface blis-schemas defines and takes already-parsed documents; where those documents live
// on disk, and what batch a particular comparison wants, are properties of a harness rather
// than of a cost model. Putting them here keeps that boundary where the schema draws it.
//
// The reason it exists at all is that four commands had four copies of the same resolution.
// That is four places for a loading rule to drift: one copy read the model graph and omitted
// the fabric, so a multi-node scenario would have priced its cross-node collectives at the
// on-node rate with nothing reporting it.
package harness

import (
	"fmt"
	"path/filepath"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
)

// Repos locates the sibling repositories a scenario's names resolve against.
//
// A scenario names its model, hardware, fabric and coefficient sets without saying where they
// live — it describes a deployment, not a filesystem — so a caller supplies the roots.
type Repos struct {
	Scenarios string // directory holding scenario files
	Catalog   string // blis-catalog checkout
	Registry  string // blis-registry checkout
}

// Open resolves one scenario into a kernel.
//
// Every artifact the scenario names is loaded through blis-schemas' own loaders, and the engine
// rules come from that package's version registry rather than from a caller, so a scenario
// pinned to an older release cannot silently get current behaviour.
func Open(scenario string, r Repos) (*latencykernel.Kernel, error) {
	sc, err := schemas.LoadScenario(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return nil, err
	}
	graph, err := schemas.LoadModelGraph(
		filepath.Join(r.Catalog, "models", sc.Model, "graph.yaml"))
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", sc.Model, err)
	}
	chip, err := schemas.LoadChip(
		filepath.Join(r.Catalog, "hardware", sc.Hardware+".yaml"))
	if err != nil {
		return nil, fmt.Errorf("hardware %q: %w", sc.Hardware, err)
	}
	var fabric *hardware.Fabric
	if sc.Fabric != "" {
		if fabric, err = schemas.LoadFabric(
			filepath.Join(r.Catalog, "networks", sc.Fabric+".yaml")); err != nil {
			return nil, fmt.Errorf("fabric %q: %w", sc.Fabric, err)
		}
	}
	sets := make([]*coefficient.Set, 0, len(sc.Coefficients))
	for _, name := range sc.Coefficients {
		set, err := schemas.LoadCoefficientSet(
			filepath.Join(r.Registry, "coefficients", name+".yaml"))
		if err != nil {
			return nil, fmt.Errorf("coefficient set %q: %w", name, err)
		}
		sets = append(sets, set)
	}
	devices, err := schemas.LoadStorageDevices(
		filepath.Join(r.Catalog, "devices", "storage.yaml"))
	if err != nil {
		return nil, err
	}
	pack := rules.Lookup(sc.EngineVersion)
	if pack == nil {
		return nil, fmt.Errorf(
			"no engine rules for version %q; known versions are %v. A layout cannot be "+
				"resolved without them, and a nearby version would misprice whatever "+
				"changed between the two", sc.EngineVersion, rules.Versions())
	}
	return latencykernel.New(latencykernel.Inputs{
		Scenario: sc, PoolIndex: 0, Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: pack,
	})
}

// DecodeThreshold is the scheduled-token count above which the kernel prices a request as
// prefill. It mirrors the engine's own classifier.
const DecodeThreshold = 8

// DecodeBatch builds a steady-state decode batch: every resident request contributes one token
// this step and carries the given context.
//
// This is the shape a TPOT or ITL comparison prices, and every command that scores a decode
// sweep wants exactly it.
func DecodeBatch(requests, context int) kernel.Batch {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, requests),
		DecodeThreshold: DecodeThreshold,
	}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: 1, Computed: context - 1, PromptLen: context,
		}
	}
	return b
}

// TimePerOutputToken is the interval a client observes between tokens: one step, plus the
// per-token host work the output processor does off the forward pass but which still lands
// between tokens.
//
// A composition of two interface methods rather than a method on the kernel, because it is what
// a published TPOT figure measures rather than a property of the hardware.
func TimePerOutputToken(k *latencykernel.Kernel, b kernel.Batch) float64 {
	return (k.StepTime(b).Overlap + k.OutputTokenOverhead()).Seconds()
}

// Replicas returns how many data-parallel engine instances a scenario runs.
//
// A published metric summed across replicas divides by this to give a per-step figure. It is a
// property of the scenario rather than of the kernel — the kernel prices one rank of one
// instance and has no reason to know how many instances a deployment runs — so it is read here
// rather than added to the interface.
func Replicas(scenario string, r Repos) (int, error) {
	sc, err := schemas.LoadScenario(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return 0, err
	}
	if n := sc.Pools[0].Parallel.DP; n > 0 {
		return n, nil
	}
	return 1, nil
}
