package price

import "math"

// Memory laws: how many bytes a deployment holds, and how that divides across ranks.
//
// These are the terms a feasibility check needs. They are separate from the timing laws
// because a caller asks whether a batch fits before asking how long it takes, and the fit
// question has no timing content.

// KVBytesPerToken returns per-rank KV bytes for one token.
//
//	2 * max(1, nKV/tp) * headDim * bytes(dtype) * layers
//
// Three details decide the answer, and getting any of them wrong changes a capacity
// verdict by a large factor:
//
// The head count floors at one. An engine divides KV heads by the tensor-parallel width
// and replicates rather than splitting a head, so a model with eight KV heads stops
// dividing at tp 8: at tp 16 each rank still holds one head, and per-rank bytes are the
// same as at tp 8 rather than half.
//
// Latent attention holds one vector per token with no separate value tensor, so its head
// count is one whatever the config says and no tensor-parallel width reduces it. Callers
// pass nKV = 1 for those kinds, which the model graph already records.
//
// The cache dtype is independent of the weight dtype. A bf16 model with an fp8 cache
// halves this, which is the difference between a deployment holding one long sequence and
// two.
func KVBytesPerToken(nKV, tp, headDim, layers int, dtypeBytes float64) float64 {
	if nKV <= 0 || headDim <= 0 || layers <= 0 || dtypeBytes <= 0 {
		return 0
	}
	perRankHeads := nKV
	if tp > 1 {
		perRankHeads = nKV / tp
		if perRankHeads < 1 {
			// The replication floor: a head is never split across ranks.
			perRankHeads = 1
		}
	}
	return 2 * float64(perRankHeads) * float64(headDim) * dtypeBytes * float64(layers)
}

// PagedBytes rounds a token count up to whole pages before charging for it.
//
// KV is allocated in blocks, so a 17-token sequence at a block size of 16 occupies two
// blocks rather than 17/16 of one. The rounding matters most where it is least expected:
// at a large block size a short sequence wastes most of a page, and a capacity estimate
// that ignored it would overcount how many sequences fit.
func PagedBytes(tokens, blockSize int, bytesPerToken float64) int64 {
	if tokens <= 0 || bytesPerToken <= 0 {
		return 0
	}
	if blockSize <= 1 {
		return int64(float64(tokens) * bytesPerToken)
	}
	pages := (tokens + blockSize - 1) / blockSize
	return int64(float64(pages) * float64(blockSize) * bytesPerToken)
}

// RecurrentStateBytes returns per-sequence bytes for one recurrent layer.
//
// The mode decides whether this is a fixed cost or a context-proportional one, which is a
// difference of three orders of magnitude rather than a detail:
//
//   - RecurrentCacheNone: one page plus the speculative blocks. Prefix caching off.
//   - RecurrentCacheAlign: two pages plus speculation and prefill checkpoints. The
//     default when prefix caching is on.
//   - RecurrentCacheAll: one page per block position up to the context bound, which makes
//     the state proportional to maxModelLen and so a variable cost rather than a fixed one.
//
// A memory model that treated the state as fixed in every mode would understate the third
// by the block count — 8192x at a 131072 context and 16-token blocks.
func RecurrentStateBytes(pageBytes float64, mode RecurrentCacheMode,
	maxModelLen, blockSize, specBlocks, prefillCheckpointBlocks int) float64 {
	if pageBytes <= 0 {
		return 0
	}
	switch mode {
	case RecurrentCacheAll:
		if blockSize <= 0 {
			return pageBytes
		}
		pages := (maxModelLen + blockSize - 1) / blockSize
		return pageBytes * float64(pages+specBlocks)
	case RecurrentCacheAlign:
		return pageBytes * float64(2+specBlocks+prefillCheckpointBlocks)
	default:
		return pageBytes * float64(1+specBlocks)
	}
}

// RecurrentCacheMode is the engine's cache strategy for recurrent layers.
type RecurrentCacheMode string

const (
	RecurrentCacheNone  RecurrentCacheMode = "none"
	RecurrentCacheAlign RecurrentCacheMode = "align"
	RecurrentCacheAll   RecurrentCacheMode = "all"
)

// ProportionalToContext reports whether a mode makes the state grow with the context
// bound. Callers use it to decide which memory term the state belongs in: a fixed
// per-sequence cost, or one that varies with prompt length.
func (m RecurrentCacheMode) ProportionalToContext() bool {
	return m == RecurrentCacheAll
}

// ExpertsPerRank returns how many experts each rank of an expert-parallel group holds,
// and the static imbalance the placement carries.
//
// An engine distributes experts as evenly as it can: with a remainder, the first ranks
// take one extra. So an indivisible split is legal and uneven, and the imbalance is a
// step-time term rather than an error — the grouped GEMM on a rank with one extra expert
// is larger, and the collective waits for the slowest rank.
//
// Under expert-parallel load balancing the physical count must divide the width, and an
// indivisible layout does not start at all. That check belongs to the resolver, which has
// the scenario; this function reports what the placement would be.
func ExpertsPerRank(experts, ep int) (base, withExtra, ranksWithExtra int, imbalance float64) {
	if ep <= 0 || experts <= 0 {
		return 0, 0, 0, 1
	}
	base = experts / ep
	ranksWithExtra = experts % ep
	withExtra = base
	if ranksWithExtra > 0 {
		withExtra = base + 1
	}
	if base == 0 {
		// More ranks than experts: some ranks hold none, and the layer cannot run as a
		// balanced collective. Reported rather than smoothed over.
		return 0, withExtra, ranksWithExtra, 0
	}
	return base, withExtra, ranksWithExtra, float64(withExtra) / float64(base)
}

// ExpertsTouched returns the expected number of DISTINCT local experts a step reads.
//
// The cost model's decode assumption is that every local expert is touched, so a rank
// reads its whole expert shard however small the batch. That holds at the batch sizes the
// assumption was written for and fails badly below them: with 224 experts and top-8
// routing, a single token can touch at most 8 of them and a rank holding 14 reads one.
// Charging the whole shard at one token overstates the memory term by fourteen times, and
// the error is largest exactly where a latency-sensitive deployment operates.
//
// A local expert goes untouched when every one of the step's tokens routes elsewhere.
// Treating each token's choice of k from E as independent across tokens,
//
//	P(one local expert untouched) = ((E - k) / E) ^ tokens
//
// so the expected count is localExperts times one minus that. The independence assumption
// is the router's own behaviour at the granularity that matters: a token's top-k is
// chosen without replacement WITHIN the token, which this respects by using k/E as the
// per-token hit probability, and across tokens the router is not coordinated.
//
// The result saturates quickly — at 224 experts and top-8 it is within one percent of the
// full shard by 128 tokens — so this changes decode pricing at small batch and nothing
// above it. It is deliberately a continuous expectation rather than a rounded count: a
// step reads whole experts, but a cost model averaging over steps wants the mean.
func ExpertsTouched(tokens, totalExperts, topK int, localExperts float64) float64 {
	if localExperts <= 0 || tokens <= 0 {
		return 0
	}
	if totalExperts <= 0 || topK <= 0 || topK >= totalExperts {
		// Every expert is reachable by any token, so the whole shard is read.
		return localExperts
	}
	missOnce := float64(totalExperts-topK) / float64(totalExperts)
	// math.Pow with a large exponent underflows to zero, which is the right limit: at
	// many tokens nothing goes untouched.
	untouched := math.Pow(missOnce, float64(tokens))
	return localExperts * (1 - untouched)
}
