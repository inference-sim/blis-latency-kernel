module github.com/inference-sim/blis-latency-kernel

go 1.24

// The schemas repository is developed alongside this one; a release tag replaces this
// once both are published.
replace github.com/inference-sim/blis-schemas => ../blis-schemas

require (
	github.com/inference-sim/blis-schemas v0.0.0
	github.com/inference-sim/inference-sim v0.0.0-00010101000000-000000000000
)

require (
	github.com/sirupsen/logrus v1.9.3 // indirect
	golang.org/x/sys v0.0.0-20220715151400-c0bba94af5f8 // indirect
	gopkg.in/yaml.v3 v3.0.1
)

replace github.com/inference-sim/inference-sim => /Users/sri/Documents/Projects/mechanismdesign/learningmaterials/inference-sim
