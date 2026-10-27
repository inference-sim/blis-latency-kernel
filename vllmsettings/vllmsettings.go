// Package vllmsettings converts a vLLM engine block into the engine settings the kernel prices
// (blis-latency-kernel/enginesettings). It is the one copy of that conversion: the kernel's own
// commands, harness and tests use it, and so does blis-vllm when it opens a kernel.
//
// It sits outside the kernel library on purpose. The library takes its engine settings from its
// caller, as it takes the engine's rules, so it never imports an engine model; this package is
// where a vLLM caller does the decoding.
package vllmsettings

import (
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-latency-kernel/enginesettings"
)

// Of is the kernel's engine settings for a vLLM engine block.
func Of(e deployment.Engine) enginesettings.Settings {
	s := enginesettings.Settings{
		Quantization:            e.Quantization,
		CacheDType:              e.CacheDType,
		MambaCacheDType:         e.MambaCacheDType,
		MambaCacheMode:          e.MambaCacheMode,
		MoEBackend:              e.MoEBackend,
		BlockSize:               e.BlockSize,
		MaxNumSeqs:              e.MaxNumSeqs,
		MaxNumBatchedTokens:     e.MaxNumBatchedTokens,
		MaxModelLen:             enginesettings.MaxModelLen(e.MaxModelLen),
		EnablePrefixCaching:     e.EnablePrefixCaching,
		AsyncScheduling:         e.AsyncScheduling,
		DisableCascadeAttn:      e.DisableCascadeAttn,
		All2AllBackend:          e.All2AllBackend,
		AllReduceBackend:        e.AllReduceBackend,
		DisableCustomAllReduce:  e.DisableCustomAllReduce,
		CUDAGraphMode:           e.CUDAGraphMode,
		MaxCUDAGraphCaptureSize: e.MaxCUDAGraphCaptureSize,
		CUDAGraphCaptureSizes:   e.CUDAGraphCaptureSizes,
		DCPCommBackend:          e.DCPCommBackend,
		DCPQReplicate:           e.DCPQReplicate,
		CPKVCacheInterleaveSize: e.CPKVCacheInterleaveSize,
		Frontend:                string(e.Frontend),
	}
	if e.DBO != nil {
		s.DBO = &enginesettings.DBO{Enabled: e.DBO.Enabled}
	}
	if e.EPLB != nil {
		s.EPLB = &enginesettings.EPLB{Enabled: e.EPLB.Enabled, NumRedundantExperts: e.EPLB.NumRedundantExperts}
	}
	if e.Speculative != nil {
		s.Speculative = &enginesettings.Speculative{Method: e.Speculative.Method,
			NumSpecTokens: e.Speculative.NumSpecTokens}
	}
	if e.EngramConfig != nil {
		s.EngramConfig = &enginesettings.EngramConfig{CPUOffload: e.EngramConfig.CPUOffload,
			DPSharedMemory: e.EngramConfig.DPSharedMemory, UseTHP: e.EngramConfig.UseTHP}
	}
	return s.Clone()
}

// OfPool is Of for one pool of a deployment, the shape the kernel's Repos.EngineSettings takes.
func OfPool(dep *deployment.Deployment, pool int) (enginesettings.Settings, error) {
	return Of(dep.Pools[pool].Engine), nil
}
