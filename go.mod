module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

require (
	// TEMPORARY: blis-schemas#113 at 4a20f18 (adds LoadBundle; https://github.com/inference-sim/blis-schemas/pull/113).
	// Moves to the blis-schemas release tag before this merges.
	github.com/inference-sim/blis-schemas v0.9.1-0.20261030160652-4a20f189ceb4
	gopkg.in/yaml.v3 v3.0.1
)
