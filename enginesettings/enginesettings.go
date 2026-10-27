// Package enginesettings is the engine block of one pool as the kernel prices it: the settings
// that change a step's time or the pool's occupancy, and nothing else.
//
// The kernel does not open the engine block itself. A scenario's engine: block is the engine
// model's to decode (blis-schemas holds it opaque; blis-vllm owns vLLM's), so the kernel's
// caller decodes it and hands the kernel these settings, as it hands the kernel the engine's
// rules (Repos.Rules). A vLLM caller converts with blis-latency-kernel/vllmsettings, which
// fills every field here from blis-vllm's engineconfig.Engine.
//
// The types are plain values: an unset pointer is "the engine decides", as in the block it was
// decoded from, and the kernel resolves and records what the engine decided.
package enginesettings

import "slices"

// Settings is one pool's engine settings, in the engine block's own terms.
type Settings struct {
	Quantization    string
	CacheDType      string
	MambaCacheDType string
	MambaCacheMode  string
	MoEBackend      string

	BlockSize           int
	MaxNumSeqs          int
	MaxNumBatchedTokens int
	// MaxModelLen is the context bound as stated: a count, auto, or unstated (MaxModelLen).
	MaxModelLen MaxModelLen
	// EnablePrefixCaching is nil when unstated (the engine's default, on).
	EnablePrefixCaching *bool
	// AsyncScheduling is nil when unstated (the engine decides).
	AsyncScheduling *bool
	// DisableCascadeAttn is nil when unstated.
	DisableCascadeAttn *bool

	All2AllBackend         string
	AllReduceBackend       string
	DisableCustomAllReduce *bool

	CUDAGraphMode           string
	MaxCUDAGraphCaptureSize *int
	CUDAGraphCaptureSizes   []int

	DCPCommBackend          string
	DCPQReplicate           *bool
	CPKVCacheInterleaveSize int

	// Frontend is the API server: "" or "python" (the default), or "rust".
	Frontend string

	DBO          *DBO
	EPLB         *EPLB
	Speculative  *Speculative
	EngramConfig *EngramConfig
}

// FrontendRust is the Rust API-server frontend.
const FrontendRust = "rust"

// RustFrontend reports whether the pool serves through the Rust frontend.
func (s Settings) RustFrontend() bool { return s.Frontend == FrontendRust }

// Speculating reports whether the pool drafts tokens.
func (s Settings) Speculating() bool { return s.Speculative != nil && s.Speculative.NumSpecTokens > 0 }

// DBO is dual-batch overlap.
type DBO struct{ Enabled bool }

// EPLB is expert-parallel load balancing.
type EPLB struct {
	Enabled             bool
	NumRedundantExperts int
}

// Speculative is the drafter: its method and how many tokens it drafts a step.
type Speculative struct {
	Method        string
	NumSpecTokens int
}

// EngramConfig is where an Engram table lives. Each nil field is the engine's default.
type EngramConfig struct {
	CPUOffload     *bool
	DPSharedMemory *bool
	UseTHP         *bool
}

// MaxModelLen is a context bound as an engine block states it: a positive count, auto
// (MaxModelLenAuto, fitted to the KV memory) or zero (unstated, derived from the model).
type MaxModelLen int

// MaxModelLenAuto is a bound fitted to the KV memory at startup.
const MaxModelLenAuto MaxModelLen = -1

// Auto reports whether the bound is auto.
func (m MaxModelLen) Auto() bool { return m == MaxModelLenAuto }

// Stated returns the bound and true when it is a count.
func (m MaxModelLen) Stated() (int, bool) {
	if m > 0 {
		return int(m), true
	}
	return 0, false
}

// Clone is a deep copy: a kernel keeps its own, so a caller editing the settings it passed
// does not change a kernel already built from them.
func (s Settings) Clone() Settings {
	for _, b := range []**bool{&s.EnablePrefixCaching, &s.AsyncScheduling, &s.DisableCascadeAttn,
		&s.DisableCustomAllReduce, &s.DCPQReplicate} {
		if *b != nil {
			v := **b
			*b = &v
		}
	}
	if s.MaxCUDAGraphCaptureSize != nil {
		v := *s.MaxCUDAGraphCaptureSize
		s.MaxCUDAGraphCaptureSize = &v
	}
	s.CUDAGraphCaptureSizes = slices.Clone(s.CUDAGraphCaptureSizes)
	if s.DBO != nil {
		v := *s.DBO
		s.DBO = &v
	}
	if s.EPLB != nil {
		v := *s.EPLB
		s.EPLB = &v
	}
	if s.Speculative != nil {
		v := *s.Speculative
		s.Speculative = &v
	}
	if s.EngramConfig != nil {
		v := *s.EngramConfig
		for _, b := range []**bool{&v.CPUOffload, &v.DPSharedMemory, &v.UseTHP} {
			if *b != nil {
				c := **b
				*b = &c
			}
		}
		s.EngramConfig = &v
	}
	return s
}
