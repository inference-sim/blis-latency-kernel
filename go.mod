module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

require (
	// TEMPORARY: inference-sim/blis-schemas#102 at 6c207b4 (https://github.com/inference-sim/blis-schemas/pull/102),
	// which holds a pool's engine block and an offload's connector options unopened (deployment.EngineBlock)
	// and takes the engine's rules from the caller. Moves to a blis-schemas release tag before this merges: https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-schemas v0.7.1-0.20261027144933-6c207b46a5e7
	// TEMPORARY: blis-vllm main at 9b5ec9da1999 (the vLLM rules packs, rules/pack and rules/engineconfig,
	// the engine block's decoding, moved from blis-schemas by inference-sim/inference-sim#2021). Used only
	// by vllmsettings, cmd/, internal/docgen, internal/harness and tests. Moves to a blis-vllm release
	// tag before this merges: https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261027143841-9b5ec9da1999
	gopkg.in/yaml.v3 v3.0.1
)
