package latencykernel

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// Repos locates the roots a scenario's names resolve against.
//
// A scenario names its model, hardware, fabric and coefficient sets without saying where
// they live — it describes a deployment, not a filesystem — so a caller supplies the roots.
type Repos struct {
	Scenarios string // directory holding scenario files
	Catalog   string // a blis-catalog checkout, or a pinned copy of one
	Registry  string // a blis-registry checkout, or a pinned copy of one
}

// Open resolves one scenario file into a kernel.
//
// The path-based half of the constructor pair. New takes documents a caller already holds;
// Open takes a filename and the roots its names resolve against, which is what a simulator
// or a scoring command has. Both end in New, so both inherit its validation and neither can
// price a deployment the other would reject.
//
// It is exported because the alternative is every consumer reimplementing it. Go forbids
// importing another module's internal packages, so while this lived in internal/harness,
// inference-sim carried a copy of the sequence and said so in a comment, and
// blis-config-search and every future backend repo would have needed their own. The
// duplication was tolerable only because it was confined to loading — but "which files a
// scenario implies" is this package's answer to give, not each caller's to guess.
//
// Every artifact is loaded through blis-schemas' own loaders, and the engine rules come from
// that package's version registry rather than from a caller, so a scenario pinned to an
// older release cannot silently get current behaviour.
//
// It prices the deployment's FIRST pool, which is what a colocated deployment has. Use
// OpenPool to price one pool of a disaggregated one.
func Open(scenario string, r Repos) (*Kernel, error) {
	return OpenPool(scenario, r, 0)
}

// OpenPool resolves ONE pool of a scenario file into a kernel.
//
// A disaggregated deployment states a prefill pool and a decode pool, and the two differ in
// the quantities that set step time: tensor-parallel width, whether expert parallelism is
// on, the engine's token budget. One kernel prices one pool — each runs its own engine with
// its own settings — so a caller serving the roles separately opens one kernel per pool.
// Pricing both from pool 0 would charge the decode pool the prefill pool's parallelism, and
// the simulation would still run.
//
// Open is this function at pool 0. Both exist because the common case should not have to
// name an index, and the disaggregated case cannot be served without one.
func OpenPool(scenario string, r Repos, poolIndex int) (*Kernel, error) {
	in, err := OpenInputs(scenario, r, poolIndex)
	if err != nil {
		return nil, err
	}
	return New(in)
}

// OpenInputs resolves one pool of a scenario file into the documents New prices it from,
// without building the kernel.
//
// It is OpenPool's loading half, exported for a caller that needs the documents as well as
// the kernel. A scoring harness that configures a rival backend for the same deployment
// needs the model's name, the chip's memory and the model's expert geometry -- deployment
// identity, which the kernel deliberately does not re-export (blis-schemas#36 draws that
// line: identity for comparison is the harness's concern, not the cost model's). Reading
// them from these Inputs, and building the kernel with New(in), means the harness holds the
// SAME chip and graph the kernel priced, not a second load of files that could resolve
// differently.
//
// What it must not be used for is reaching configuration the kernel resolved: widths and
// backends are Resolved's to report, and the pool's settings are Deployment's. Reading
// those from Inputs instead would rebuild the second answer to "which pool is this" that
// the interface exists to remove.
func OpenInputs(scenario string, r Repos, poolIndex int) (Inputs, error) {
	sc, dep, err := LoadBundle(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return Inputs{}, err
	}
	// Bounds-checked here as well as in New, so the error can name the FILE. New sees
	// documents and reports "outside the deployment's N pool(s)"; a caller that passed a
	// filename needs to know which of several scenarios lacks the pool it asked for.
	if poolIndex < 0 || poolIndex >= len(dep.Pools) {
		return Inputs{}, fmt.Errorf("%s states %d pool(s); pool %d was requested",
			scenario, len(dep.Pools), poolIndex)
	}
	graph, err := schemas.LoadModelGraph(
		filepath.Join(r.Catalog, "models", sc.Model, "graph.yaml"))
	if err != nil {
		return Inputs{}, fmt.Errorf("model %q: %w", sc.Model, err)
	}
	chip, err := schemas.LoadChip(
		filepath.Join(r.Catalog, "hardware", sc.Cluster.Hardware+".yaml"))
	if err != nil {
		return Inputs{}, fmt.Errorf("hardware %q: %w", sc.Cluster.Hardware, err)
	}
	var fabric *hardware.Fabric
	if sc.Cluster.Fabric != "" {
		if fabric, err = schemas.LoadFabric(
			filepath.Join(r.Catalog, "networks", sc.Cluster.Fabric+".yaml")); err != nil {
			return Inputs{}, fmt.Errorf("fabric %q: %w", sc.Cluster.Fabric, err)
		}
	}
	sets := make([]*coefficient.Set, 0, len(sc.Coefficients))
	for _, name := range sc.Coefficients {
		set, err := schemas.LoadCoefficientSet(
			filepath.Join(r.Registry, "coefficients", name+".yaml"))
		if err != nil {
			return Inputs{}, fmt.Errorf("coefficient set %q: %w", name, err)
		}
		sets = append(sets, set)
	}
	devices, err := schemas.LoadStorageDevices(
		filepath.Join(r.Catalog, "devices", "storage.yaml"))
	if err != nil {
		return Inputs{}, err
	}
	pack := rules.Lookup(sc.EngineVersion)
	if pack == nil {
		return Inputs{}, fmt.Errorf(
			"no engine rules for version %q; known versions are %v. A layout cannot be "+
				"resolved without them, and a nearby version would misprice whatever "+
				"changed between the two", sc.EngineVersion, rules.Versions())
	}
	return Inputs{
		Scenario: sc, Deployment: dep, PoolIndex: poolIndex,
		Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: pack,
	}, nil
}

// LoadBundle reads a scenario and the deployment applied to it from one file, as two YAML
// documents separated by `---`.
//
// blis-schemas keeps the two apart: a Scenario is the immutable problem (model, cluster
// inventory, coefficient and engine-version references) and a Deployment is the tunable
// configuration chosen against it (pools, each a parallelism layout with its own engine
// settings). One `blis run` is one of each.
//
// They share a file here, rather than sitting in a scenario and a sibling deployment file,
// because a measurement row addresses a deployment by a single filename —
// `{"scenario": "glm-5-h200-fp8-sglang-tp8.yaml", ...}` — and testdata/measurements holds
// over three thousand such rows. A sibling-file layout would rewrite every one of them to
// say nothing it does not already say.
//
// blis-schemas' own LoadScenario and LoadDeployment each open a path and read one document,
// so neither can reach the second half of a pair; this reproduces their strict decoding
// over a single stream instead. Strictness is the point: a misspelled key that parsed
// silently would leave a document that validates while omitting the setting its author
// intended, which is the failure mode hardest to notice.
func LoadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)

	var sc scenario.Scenario
	if err := dec.Decode(&sc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("%s is empty: a kernel needs a scenario and the "+
				"deployment applied to it", path)
		}
		return nil, nil, fmt.Errorf("decoding the scenario in %s: %w", path, err)
	}

	var dep deployment.Deployment
	if err := dec.Decode(&dep); err != nil {
		// A scenario with no deployment cannot build a kernel: there is no pool to price.
		// Saying so here names the file, where failing later would surface as an
		// out-of-range pool index far from the document that lacks one.
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("%s holds a scenario but no deployment: the pools "+
				"a kernel prices live in a second YAML document, separated by `---`", path)
		}
		return nil, nil, fmt.Errorf("decoding the deployment in %s: %w", path, err)
	}

	// Every caller reads at least one pool, several as Pools[0] without a length check of
	// their own. Rejecting an empty list here turns what would be an index panic far from
	// the file into an error that names it. A pool-less deployment describes no engine, so
	// there is nothing a caller could do with one anyway.
	if len(dep.Pools) == 0 {
		return nil, nil, fmt.Errorf(
			"the deployment in %s declares no pools: there is nothing to price", path)
	}

	return &sc, &dep, nil
}
