package latencykernel

import "github.com/inference-sim/blis-schemas/spec/deployment"

// graphCaptureRequest is what the engine's rules resolve CUDA graph capture sizes from: the
// cudagraph mode the kernel resolved, the stated maximum or sizes, and the scheduler's token
// budget (rules.EngineRules.CUDAGraphCaptureSizes reads nothing else). The kernel fills it from
// its own engine settings, edited as the engine resolves them, so the rules see the mode and the
// budget the engine runs with.
type graphCaptureRequest struct {
	Mode                string
	MaxCaptureSize      *int
	CaptureSizes        []int
	MaxNumBatchedTokens int
}

// engine is the request in the shape the rules take today.
func (r graphCaptureRequest) engine() deployment.Engine {
	return deployment.Engine{CUDAGraphMode: r.Mode, MaxCUDAGraphCaptureSize: r.MaxCaptureSize,
		CUDAGraphCaptureSizes: r.CaptureSizes, MaxNumBatchedTokens: r.MaxNumBatchedTokens}
}
