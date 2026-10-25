module github.com/inference-sim/blis-latency-kernel

go 1.27.2

require (
	// TEMPORARY: head of inference-sim/blis-schemas#102 (https://github.com/inference-sim/blis-schemas/pull/102),
	// which adds rules.EngineRules. Moves to a blis-schemas release tag before this merges.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261025005659-c09250ca92d9
	// TEMPORARY: blis-vllm main at 0287b2f75ef4 (the vLLM rules packs, moved from blis-schemas
	// by inference-sim/inference-sim#2021). Used only by cmd/, internal/docgen, internal/harness
	// and tests. Moves to a blis-vllm release tag before this merges.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261025005745-0287b2f75ef4
	gopkg.in/yaml.v3 v3.0.1
)
