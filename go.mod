module github.com/inference-sim/blis-latency-kernel

go 1.27.2

require (
	// TEMPORARY: head of inference-sim/blis-schemas#102 (https://github.com/inference-sim/blis-schemas/pull/102),
	// which adds rules.EngineRules. Moves to a blis-schemas release tag before this merges.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261024062559-1571a115227a
	// TEMPORARY: blis-vllm main at 0f264b2c9eba (the vLLM rules packs, moved from blis-schemas
	// by inference-sim/inference-sim#2021). Used only by cmd/, internal/docgen, internal/harness
	// and tests. Moves to a blis-vllm release tag before this merges.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261024062740-0f264b2c9eba
	gopkg.in/yaml.v3 v3.0.1
)
