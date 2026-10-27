module github.com/inference-sim/blis-latency-kernel

go 1.24.0

toolchain go1.24.2

require (
	// TEMPORARY: inference-sim/blis-schemas#102 at ddfd45b (https://github.com/inference-sim/blis-schemas/pull/102),
	// which adds rules.EngineRules, rules.CUDAGraphCaptureInput and deployment.EngineBlock. Moves to a blis-schemas
	// release tag before this merges: https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-schemas v0.6.1-0.20261027005726-ddfd45b9e944
	// TEMPORARY: blis-vllm main at 480d4340c353 (the vLLM rules packs and rules/engineconfig, the
	// engine block's decoding, moved from blis-schemas by inference-sim/inference-sim#2021). Used only
	// by vllmsettings, cmd/, internal/docgen, internal/harness and tests. Moves to a blis-vllm release
	// tag before this merges: https://github.com/inference-sim/blis-latency-kernel/issues/88.
	github.com/inference-sim/blis-vllm/rules v0.0.0-20261027010410-480d4340c353
	gopkg.in/yaml.v3 v3.0.1
)
