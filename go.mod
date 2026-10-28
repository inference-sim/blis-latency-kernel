module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

require (
	// TEMPORARY: inference-sim/blis-schemas#106 at bd0c5d0 (https://github.com/inference-sim/blis-schemas/pull/106,
	// stacked on #102): #102 holds a pool's engine block and an offload's connector options unopened
	// (deployment.EngineBlock) and takes the engine's rules from the caller, and #106 adds the engine settings
	// the kernel prices from (kernel/enginesettings). Moves to a blis-schemas release tag before this merges: https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-schemas v0.7.1-0.20261028162459-bd0c5d015b01
	gopkg.in/yaml.v3 v3.0.1
)
