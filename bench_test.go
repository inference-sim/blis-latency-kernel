package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
)

// StepTime is called once per simulated step, millions of times over a run, so its cost
// and its allocation count both matter. The per-stage composition walks one entry per
// distinct layer KIND rather than per layer — three entries for a 72-layer model — and
// accumulates into a fixed array, so the only allocation is the published result map.
func BenchmarkStepTimeDecode(b *testing.B) {
	k := benchKernel(b)
	batch := decodeBatch(256, 2, 8192)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTime(batch)
	}
}

func BenchmarkStepTimeSingleRequest(b *testing.B) {
	k := benchKernel(b)
	batch := decodeBatch(1, 1, 8192)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTime(batch)
	}
}

func BenchmarkStepTimeLargeBatch(b *testing.B) {
	// The batch loop is the one term that walks the requests, so a wide batch shows its
	// cost against the per-kind loop's fixed size.
	k := benchKernel(b)
	batch := decodeBatch(1024, 1, 32768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTime(batch)
	}
}

func BenchmarkMemoryQueries(b *testing.B) {
	k := benchKernel(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.SequenceVariableBytes(8192)
		_ = k.FixedBytes()
	}
}

// benchKernel builds a kernel once, outside the timed loop.
func benchKernel(b *testing.B) *Kernel {
	b.Helper()
	return fixture(b, "granite5-h200-ep16.yaml")
}

// BenchmarkStepTimeIntoDecode is the form a simulator's inner loop uses: one reused map,
// no allocation per step.
func BenchmarkStepTimeIntoDecode(b *testing.B) {
	k := benchKernel(b)
	batch := decodeBatch(256, 2, 8192)
	per := make(map[kernel.Resource]time.Duration, 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.StepTimeInto(batch, per)
	}
}
