package latencykernel

import (
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/rules/v0_29"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
)

// The repositories are siblings rather than vendored, so a test that needs a real
// catalog reads it from disk and skips when it is absent.
const (
	catalogRoot  = "/Users/sri/Documents/Projects/blis-catalog"
	registryRoot = "/Users/sri/Documents/Projects/blis-registry"
)

// fixture builds a kernel from a committed scenario and the real catalog and registry.
//
// Deliberately not a hand-built Inputs struct. A kernel assembled from literals in a
// test proves the arithmetic and nothing about whether the committed artifacts can
// drive it, and the second is where the failures have been.
func fixture(t testing.TB, scenario string) *Kernel {
	t.Helper()
	sc, err := schemas.LoadScenario(filepath.Join("testdata", scenario))
	if err != nil {
		t.Skipf("scenario unavailable: %v", err)
	}
	graph, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", sc.Model, "graph.yaml"))
	if err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	chip, err := schemas.LoadChip(
		filepath.Join(catalogRoot, "hardware", sc.Hardware+".yaml"))
	if err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	var fabric *hardware.Fabric
	if sc.Fabric != "" {
		fabric, err = schemas.LoadFabric(
			filepath.Join(catalogRoot, "networks", sc.Fabric+".yaml"))
		if err != nil {
			t.Skipf("catalog unavailable: %v", err)
		}
	}
	var sets []*coefficient.Set
	for _, name := range sc.Coefficients {
		set, err := schemas.LoadCoefficientSet(
			filepath.Join(registryRoot, "coefficients", name+".yaml"))
		if err != nil {
			t.Skipf("registry unavailable: %v", err)
		}
		sets = append(sets, set)
	}
	devices, err := schemas.LoadStorageDevices(
		filepath.Join(catalogRoot, "devices", "storage.yaml"))
	if err != nil {
		t.Skipf("catalog unavailable: %v", err)
	}
	k, err := New(Inputs{
		Scenario: sc, PoolIndex: 0, Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: v0_29.Pack(),
	})
	if err != nil {
		t.Fatalf("New from committed artifacts: %v", err)
	}
	return k
}
