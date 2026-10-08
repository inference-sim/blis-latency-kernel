package latencykernel

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-latency-kernel/internal/artifacttest"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/rules/v0_29"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
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
// so every test here skipped on "catalog unavailable" and 41 of them were passing by not
// running — measured by `go test -v .` at 3cf1ef1: 41 top-level skips, 46 counting
// subtests, 0 failures, suite reporting ok. artifacttest.RequireArtifact draws that line.
func fixtureInputs(t testing.TB, scenario string) Inputs {
	t.Helper()
	// The fixture itself is committed beside this test, so it is never merely absent:
	// a failure to read one is a malformed fixture and always a real failure.
	sc, dep, err := LoadBundle(filepath.Join("testdata", scenario))
	if err != nil {
		t.Fatalf("committed fixture %s: %v", scenario, err)
	}
	graphPath := filepath.Join(catalogRoot, "models", sc.Model, "graph.yaml")
	graph, err := schemas.LoadModelGraph(graphPath)
	artifacttest.RequireArtifact(t, catalogRoot, graphPath, "catalog", err)
	chipPath := filepath.Join(catalogRoot, "hardware", sc.Cluster.Hardware+".yaml")
	chip, err := schemas.LoadChip(chipPath)
	artifacttest.RequireArtifact(t, catalogRoot, chipPath, "catalog", err)
	var fabric *hardware.Fabric
	if sc.Cluster.Fabric != "" {
		fabricPath := filepath.Join(catalogRoot, "networks", sc.Cluster.Fabric+".yaml")
		fabric, err = schemas.LoadFabric(fabricPath)
		artifacttest.RequireArtifact(t, catalogRoot, fabricPath, "catalog", err)
	}
	var sets []*coefficient.Set
	for _, name := range sc.Coefficients {
		setPath := filepath.Join(registryRoot, "coefficients", name+".yaml")
		set, err := schemas.LoadCoefficientSet(setPath)
		artifacttest.RequireArtifact(t, registryRoot, setPath, "registry", err)
		sets = append(sets, set)
	}
	devicesPath := filepath.Join(catalogRoot, "devices", "storage.yaml")
	devices, err := schemas.LoadStorageDevices(devicesPath)
	artifacttest.RequireArtifact(t, catalogRoot, devicesPath, "catalog", err)
	return Inputs{
		Scenario: sc, Deployment: dep, PoolIndex: 0,
		Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: v0_29.Pack(),
	}
}

// fixture builds a kernel from the Inputs a committed scenario implies.
//
// Split from fixtureInputs so a test that needs to PERTURB one field before construction --
// to watch New refuse a deployment that does not fit its cluster, say -- can take the
// documents and assemble them itself, rather than reimplementing the loading.
func fixture(t testing.TB, scenario string) *Kernel {
	t.Helper()
	k, err := New(fixtureInputs(t, scenario))
	if err != nil {
		t.Fatalf("New from committed artifacts: %v", err)
	}
	return k
}
