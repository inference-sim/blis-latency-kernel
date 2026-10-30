package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// A piecewise-captured stage replays its own layers' segments: one per layer plus one per
// sparse-MLA indexer it holds, plus one. Two stages of one model, one holding the sparse
// layers, replay different counts, and neither its bare layer count.
func TestAPiecewiseStageReplaysItsOwnSegments(t *testing.T) {
	k := &Kernel{graphDecode: graphModePiecewise, graphOther: graphModePiecewise,
		replayPerStep: time.Microsecond, captureTokens: 512}
	dense := price.PlannedLayer{Count: 4, AttnKind: model.AttentionGQA, AttnQHeads: 8,
		Attentions: []price.PlannedAttention{{}}}
	sparse := price.PlannedLayer{Count: 3, AttnKind: model.AttentionSparseMLA, AttnQHeads: 8,
		Attentions: []price.PlannedAttention{{}, {Role: "block_index_scores"}}}
	b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 9, PromptLen: 10}}}
	for _, c := range []struct {
		what string
		st   pipelineStage
		want time.Duration
	}{
		{"dense stage", pipelineStage{layers: []price.PlannedLayer{dense}, totalLayers: 4}, 5 * time.Microsecond},
		{"sparse stage", pipelineStage{layers: []price.PlannedLayer{sparse}, totalLayers: 3}, 7 * time.Microsecond},
	} {
		if got := k.stageHost(b, &c.st, 0); got != c.want {
			t.Errorf("%s: host %v, want %v", c.what, got, c.want)
		}
	}
	// Under a breakable capture the stage replays one piece per break it holds, plus one,
	// even where the whole model's count is larger.
	k.breakableCapture, k.captureBreaks = true, 99
	st := pipelineStage{layers: []price.PlannedLayer{sparse}, totalLayers: 3}
	if got := k.stageHost(b, &st, 0); got != 4*time.Microsecond {
		t.Errorf("breakable stage: host %v, want 4us", got)
	}
}
