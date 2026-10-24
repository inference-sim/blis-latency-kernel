module github.com/inference-sim/blis-latency-kernel

go 1.27.2

require (
	// TEMPORARY: head of inference-sim/blis-schemas#102 (https://github.com/inference-sim/blis-schemas/pull/102),
	// which adds rules.EngineRules. Moves to a blis-schemas release tag before this merges.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261024105647-9a71eeb47a9e
	// TEMPORARY: blis-vllm main at 8db6c9d9e238 (the vLLM rules packs, moved from blis-schemas
	// by inference-sim/inference-sim#2021). Used only by cmd/, internal/docgen, internal/harness
	// and tests. Moves to a blis-vllm release tag before this merges.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261024105808-8db6c9d9e238
	gopkg.in/yaml.v3 v3.0.1
)
