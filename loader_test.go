package latencykernel

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/rules/v0_29"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// The catalog and registry are pinned copies committed under testdata, so these tests
// read real artifacts without depending on a checkout outside this repository. See
// testdata/VENDORED.md for what is vendored, from which upstream commit, and why.
//
// They are relative paths because a Go test runs with its own package directory as the
// working directory, and this package is the repository root. Overridable so a working
// copy can be scored against a live upstream checkout:
//
//	BLIS_CATALOG=../blis-catalog go test ./...
var (
	catalogRoot  = envOr("BLIS_CATALOG", filepath.Join("testdata", "catalog"))
	registryRoot = envOr("BLIS_REGISTRY", filepath.Join("testdata", "registry"))
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// fixture builds a kernel from a committed scenario and the real catalog and registry.
//
// Deliberately not a hand-built Inputs struct. A kernel assembled from literals in a
// test proves the arithmetic and nothing about whether the committed artifacts can
// drive it, and the second is where the failures have been.
//
// A MISSING sibling checkout skips, because not every working copy has one. A checkout
// that is PRESENT but unreadable fails, because that is a real incompatibility and
// skipping it is how one hid: under the pseudo-version this repo was pinned to,
// blis-catalog's storage.yaml failed to decode (it already carried v0.2.0's field names),
// so every test here skipped on "catalog unavailable" and 35 of them were passing by not
// running. absentCheckout below draws that line.
func fixture(t testing.TB, scenario string) *Kernel {
	t.Helper()
	// The fixture itself is committed beside this test, so it is never merely absent:
	// a failure to read one is a malformed fixture and always a real failure.
	sc, dep, err := loadBundle(filepath.Join("testdata", scenario))
	if err != nil {
		t.Fatalf("committed fixture %s: %v", scenario, err)
	}
	graph, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", sc.Model, "graph.yaml"))
	requireArtifact(t, catalogRoot, "catalog", err)
	chip, err := schemas.LoadChip(
		filepath.Join(catalogRoot, "hardware", sc.Cluster.Hardware+".yaml"))
	requireArtifact(t, catalogRoot, "catalog", err)
	var fabric *hardware.Fabric
	if sc.Cluster.Fabric != "" {
		fabric, err = schemas.LoadFabric(
			filepath.Join(catalogRoot, "networks", sc.Cluster.Fabric+".yaml"))
		requireArtifact(t, catalogRoot, "catalog", err)
	}
	var sets []*coefficient.Set
	for _, name := range sc.Coefficients {
		set, err := schemas.LoadCoefficientSet(
			filepath.Join(registryRoot, "coefficients", name+".yaml"))
		requireArtifact(t, registryRoot, "registry", err)
		sets = append(sets, set)
	}
	devices, err := schemas.LoadStorageDevices(
		filepath.Join(catalogRoot, "devices", "storage.yaml"))
	requireArtifact(t, catalogRoot, "catalog", err)
	k, err := New(Inputs{
		Scenario: sc, Deployment: dep, PoolIndex: 0,
		Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: v0_29.Pack(),
	})
	if err != nil {
		t.Fatalf("New from committed artifacts: %v", err)
	}
	return k
}

// loadBundle reads a fixture's scenario and deployment, the two documents of one file.
//
// internal/harness has the same reader, but this package cannot call it: harness imports
// this one, so a test here importing harness would close an import cycle. The duplication
// is deliberate and small, and it keeps the direction of dependency right — a cost model
// does not depend on a harness.
func loadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	var sc scenario.Scenario
	if err := dec.Decode(&sc); err != nil {
		return nil, nil, fmt.Errorf("decoding the scenario in %s: %w", path, err)
	}
	var dep deployment.Deployment
	if err := dec.Decode(&dep); err != nil {
		return nil, nil, fmt.Errorf("decoding the deployment in %s: %w", path, err)
	}
	if len(dep.Pools) == 0 {
		return nil, nil, fmt.Errorf(
			"the deployment in %s declares no pools: there is nothing to price", path)
	}
	return &sc, &dep, nil
}

// requireArtifact reports a sibling-repository load failure as a skip or a failure,
// depending on which of the two it actually is.
//
// A sibling checkout that is not there skips: not every working copy has blis-catalog and
// blis-registry beside it, and a test cannot read what is absent. A checkout that IS there
// and still failed to load is an incompatibility between this repo and that one — the
// exact condition worth failing on, and the one a blanket skip hid for the whole life of
// the pseudo-version pin.
func requireArtifact(t testing.TB, root, what string, err error) {
	t.Helper()
	if err == nil {
		return
	}
	if _, statErr := os.Stat(root); os.IsNotExist(statErr) {
		t.Skipf("%s checkout absent at %s", what, root)
	}
	t.Fatalf("%s at %s is present but unreadable, which is an incompatibility rather "+
		"than a missing checkout: %v", what, root, err)
}
