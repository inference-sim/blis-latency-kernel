module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

// The schemas repository is developed alongside this one; a release tag replaces this
// once both are published.
replace github.com/inference-sim/blis-schemas => ../blis-schemas

require github.com/inference-sim/blis-schemas v0.0.0

require gopkg.in/yaml.v3 v3.0.1 // indirect
