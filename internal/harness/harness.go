// Package harness holds what this repository's scoring and reporting commands share: opening a
// kernel from a scenario file, and building the batch shapes a comparison prices.
//
// It is internal and it is not part of the cost model. The kernel package implements the
// interface blis-schemas defines and takes already-parsed documents; where those documents live
// on disk, and what batch a particular comparison wants, are properties of a harness rather
// than of a cost model. Putting them here keeps that boundary where the schema draws it.
//
// The reason it exists at all is that four commands had four copies of the same resolution.
// That is four places for a loading rule to drift: one copy read the model graph and omitted
// the fabric, so a multi-node scenario would have priced its cross-node collectives at the
// on-node rate with nothing reporting it.
package harness

import (
	"path/filepath"

	"github.com/inference-sim/blis-schemas/kernel"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
)

// Repos is the root package's Repos. An alias rather than a second type, so that a
// harness.Repos{...} literal and a latencykernel.Repos{...} literal are the same value and
// no conversion is needed at the boundary.
type Repos = latencykernel.Repos

// Open forwards to the root package's Open, which is where this sequence now lives.
//
// It stayed here as a function rather than being deleted because every command in this
// repository calls harness.Open, and the move was about making the sequence REACHABLE from
// other modules rather than about relocating callers. Keeping the name means the promotion
// is a visibility change and nothing else.
func Open(scenario string, r Repos) (*latencykernel.Kernel, error) {
	return latencykernel.Open(scenario, r)
}

// DecodeThreshold is the scheduled-token count above which the kernel prices a request as
// prefill, for the batches the scoring commands build. It is this harness's cutoff, not a
// value read from the engine: vLLM's is per backend -- 1 by default, raised to
// 1 + num_speculative_tokens under speculative decoding, and 128 or 512 on the FlashMLA
// and FlashAttention-MLA backends (vllm/v1/attention/backends/utils.py, backend.py, and
// mla/flashmla.py:113, mla/flashattn_mla.py:110 at v0.31.0). cmd/shape and cmd/score price
// single-token decode batches, which every one of those thresholds classifies as decode;
// a prefill chunk of at most 512 tokens, which cmd/bandprobe and cmd/worked-table can pass,
// is one the FlashMLA-family backends would run on their decode path instead.
const DecodeThreshold = 8

// DecodeBatch builds a steady-state decode batch: every resident request contributes one token
// this step and carries the given context.
//
// This is the shape a TPOT or ITL comparison prices, and every command that scores a decode
// sweep wants exactly it.
func DecodeBatch(requests, context int) kernel.Batch {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, requests),
		DecodeThreshold: DecodeThreshold,
	}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: 1, Computed: context - 1, PromptLen: context,
		}
	}
	return b
}

// TimePerOutputToken is the interval a client observes between tokens: one step, plus the
// per-token host work the output processor does off the forward pass but which still lands
// between tokens.
//
// A composition of two interface methods rather than a method on the kernel, because it is what
// a published TPOT figure measures rather than a property of the hardware.
//
// NoOverlap, not Overlap, and the choice is measured. StepTime's own comment states it:
// "A caller that wants one figure should read NoOverlap." Over 219 points of NVIDIA's FPM
// dataset spanning two models, two parts and five parallelism topologies, Overlap's signed
// error is -13.45% against NoOverlap's -3.44%, and NoOverlap is closer on 158 of them; the
// physical reason is PIECEWISE cudagraph mode, where attention runs eagerly between captured
// segments so per-layer overlap is structurally limited. blis-registry's
// docs/band-selection.md records the evidence.
//
// This read Overlap until schemas v0.2.0 removed StepEstimate.Expected, which the kernel had
// set to NoOverlap from exactly that evidence. Dropping the named edge left every caller to
// re-decide, and the three in this repository all picked the optimistic one -- so the
// migration silently under-priced every reported TPOT by about ten points.
func TimePerOutputToken(k *latencykernel.Kernel, b kernel.Batch) float64 {
	return (k.StepTime(b).NoOverlap + k.OutputTokenOverhead()).Seconds()
}

// Replicas returns how many data-parallel engine instances a deployment runs.
//
// A published metric summed across replicas divides by this to give a per-step figure. It is a
// property of the deployment rather than of the kernel — the kernel prices one rank of one
// instance and has no reason to know how many instances a deployment runs — so it is read here
// rather than added to the interface.
func Replicas(scenario string, r Repos) (int, error) {
	_, dep, err := LoadBundle(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return 0, err
	}
	if n := dep.Pools[0].Parallel.DP; n > 0 {
		return n, nil
	}
	return 1, nil
}
