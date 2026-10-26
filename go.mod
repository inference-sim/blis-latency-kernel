module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

require (
	// TEMPORARY: inference-sim/blis-schemas#102 at 630937e (https://github.com/inference-sim/blis-schemas/pull/102),
	// which adds rules.EngineRules; #102's later commits change only comments. Moves to a blis-schemas release tag before this merges:
	// https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261025121631-630937e6f21f
	// TEMPORARY: blis-vllm main at f43fad16b584 (the vLLM rules packs, moved from blis-schemas
	// by inference-sim/inference-sim#2021). Used only by cmd/, internal/docgen, internal/harness
	// and tests. Moves to a blis-vllm release tag before this merges:
	// https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261025013633-f43fad16b584
	gopkg.in/yaml.v3 v3.0.1
)
