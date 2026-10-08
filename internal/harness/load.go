package harness

import (
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/scenario"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
)

// LoadBundle forwards to the root package's LoadBundle, which is where the reader now
// lives.
//
// It moved out with Open, for the same reason: a consumer in another module could not reach
// it here, and reading the scenario+deployment pair is part of answering "which files does
// this scenario imply" rather than part of this harness. The forward keeps every existing
// caller compiling.
func LoadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	return latencykernel.LoadBundle(path)
}
