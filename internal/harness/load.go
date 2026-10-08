package harness

import (
	"errors"
	"fmt"
	"io"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/scenario"
)

// LoadBundle reads a scenario and the deployment applied to it from one file, as two
// YAML documents separated by `---`.
//
// blis-schemas keeps the two apart: a Scenario is the immutable problem (model, cluster
// inventory, coefficient and engine-version references) and a Deployment is the tunable
// configuration chosen against it (pools, each a parallelism layout with its own engine
// settings). One `blis run` is one of each.
//
// They share a file here, rather than sitting in a scenario and a sibling deployment
// file, because a measurement row addresses a deployment by a single filename —
// `{"scenario": "glm-5-h200-fp8-sglang-tp8.yaml", ...}` — and testdata/measurements
// holds over three thousand such rows. A sibling-file layout would rewrite every one of
// them to say nothing it does not already say.
//
// blis-schemas' own LoadScenario and LoadDeployment each open a path and read one
// document, so neither can reach the second half of a pair; this reproduces their
// strict decoding over a single stream instead. Strictness is the point: a misspelled
// key that parsed silently would leave a document that validates while omitting the
// setting its author intended, which is the failure mode hardest to notice.
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

	// Every caller reads at least one pool, several as Pools[0] without a length check
	// of their own. Rejecting an empty list here turns what would be an index panic far
	// from the file into an error that names it. A pool-less deployment describes no
	// engine, so there is nothing a caller could do with one anyway.
	if len(dep.Pools) == 0 {
		return nil, nil, fmt.Errorf(
			"the deployment in %s declares no pools: there is nothing to price", path)
	}

	return &sc, &dep, nil
}
