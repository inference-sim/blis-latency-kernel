module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

// The schemas repository is developed alongside this one; a release tag replaces this
// once both are published.
replace github.com/inference-sim/blis-schemas => ../blis-schemas

require (
	github.com/inference-sim/blis-schemas v0.0.0
	github.com/inference-sim/inference-sim v0.9.2
)

require (
	github.com/sirupsen/logrus v1.9.3 // indirect
	golang.org/x/sys v0.0.0-20220715151400-c0bba94af5f8 // indirect
	gopkg.in/yaml.v3 v3.0.1
)

// cmd/baseline compares this kernel against BLIS's earlier roofline and trained-physics models,
// so it imports the simulator. It is behind the `baseline` build tag and excluded from a default
// build, because the simulator is a large dependency that nothing else here needs. It resolves
// against the published simulator module, so no local checkout or replace directive is needed:
//
//	go test -tags baseline ./cmd/baseline
//	go run  -tags baseline ./cmd/baseline
