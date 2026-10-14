package latencykernel

import (
	"math"

	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// The per-layer and head terms the step composes that are more than a line each.

// decodeReadBytes is what one layer's primary attention reads for these decode requests.
//
// A layer that states its caches reads their records (statedReadBytes). Any other reads
// its inferred cache, at its own per-token width, over the tokens its bound and the
// decode-context shard leave it (dcpDecodeTokens).
func (k *Kernel) decodeReadBytes(l *price.PlannedLayer, kv []layerKV, i int,
	contexts []int) float64 {
	if kv[i].stated {
		var read float64
		for _, ctx := range contexts {
			if ctx > 0 {
				read += k.statedReadBytes(l.AttnCaches, l.AttnIndexTopK, ctx)
			}
		}
		return read
	}
	var tokens float64
	if bounded := k.dcpDecodeTokens(l, l.AttnKind, contexts); bounded >= 0 {
		tokens = bounded
	} else {
		for _, ctx := range contexts {
			if ctx > 0 {
				tokens += float64(ctx)
			}
		}
	}
	return tokens * kv[i].perToken
}

// prefillAttention prices one attention kernel's prefill work, given in the units
// causalFLOPs carries times its per-rank heads and head width: the measured floor and the
// work on the ramp at the measured scale, charged to SM. A part with no measured prefill
// form returns the work as FLOPs instead, for the layer's unmodified ramp, which the sweeps
// say understates by over ten times.
func (k *Kernel) prefillAttention(work float64, s *stepShape) (smSeconds, flops float64) {
	if k.attentionPrefillScale <= 0 {
		return 0, work
	}
	scale := price.Efficiency(s.tokensF, k.epsMax, k.mHalf) * k.attentionPrefillScale
	return k.attentionPrefillFloor.Seconds() +
		work/(k.computeFLOPsPerSecond*scale)*s.smDerate, 0
}

// attentionHeads is the query heads one rank's attention kernel runs over: its share of
// the heads under tensor parallelism, or all of them for a kernel every rank runs whole.
//
// DeepSeek-V4-family FlashMLA on SM90 pads them. Its FP8 kernels take only 64 or 128 query
// heads, so each rank pads its local heads up -- "return 64 if num_heads <= 64 else 128"
// (vllm/models/deepseek_v41/nvidia/flashmla.py:66-74 at v0.31.0) -- and computes the
// padded ones. That happens only where tensor parallelism splits the heads
// (n_local_heads = n_heads // tp_size, attention.py:276-284): at TP8 a 64-head V4.1 rank
// holds 8 heads and computes 64, and a 128-head V4-Pro rank holds 16 and computes 64. Under
// attention data parallelism a rank keeps every head and pads nothing.
func (k *Kernel) attentionHeads(q int, replicated bool, kind model.AttentionKind) float64 {
	h := float64(q)
	if replicated {
		return h
	}
	h /= k.tp
	if k.padFlashMLA && kind == model.AttentionSparseMLA && k.tp > 1 {
		if h <= 64 {
			return 64
		}
		return 128
	}
	return h
}

// contextBytes is one layer's context-proportional Elementwise traffic this step (blis-
// schemas model.ContextWork): work that scales with the attended context rather than with
// the scheduled tokens, such as DeepSeek-V4.1's prefill dequantize of a request's whole
// compressed stream into a BF16 workspace (vllm/models/deepseek_v41/nvidia/flashmla.py:
// 305-313 at v0.31.0) and its indexer's logits and candidate masks, which run one row per
// query token over the whole logits width.
func (k *Kernel) contextBytes(l *price.PlannedLayer, s *stepShape) float64 {
	var total float64
	for _, c := range l.Context {
		tpe := float64(c.TokensPerEntry)
		var entries float64
		switch c.Per {
		case model.PerQueryToken:
			// A prefill query covers its own prefix: the chunk's causal pair count.
			for _, ch := range s.prefillChunks {
				sched := float64(ch.sched)
				entries += (sched*float64(ch.prefix) + sched*sched*0.5) / tpe
			}
			if !c.PrefillOnly {
				if c.MaxModelLenExtent && k.maxModelLen > 0 {
					// The decode kernel runs over a buffer padded to max_model_len.
					entries += float64(s.decodeTokens) * float64(k.maxModelLen) / tpe
				} else {
					entries += s.decodeQueryCtx / tpe
				}
			}
		case model.PerRequest:
			for _, p := range s.prefills {
				entries += float64(p.computed+p.scheduled) / tpe
			}
			if !c.PrefillOnly {
				entries += s.decodeKVTokens / tpe
			}
		}
		b := entries * c.BytesPerEntry
		if !c.Replicated {
			b /= k.tp
		}
		total += b
	}
	return total
}

// gatherTraffic is one layer's table lookups this step: the device bytes they move, and
// the seconds a host-resident table's rows take to cross the host link.
//
// A rank reads the rows it holds: under a row split each token's row lives on one rank,
// so a rank reads 1/tp of them; under a head split each rank reads one row from each of
// its own ceil(rows_per_token / shards) heads (common/engram.py part_n_hash_cols); a
// replicated table is read whole. Each rank writes the gathered output.
func (k *Kernel) gatherTraffic(l *price.PlannedLayer, s *stepShape) (bytes, hostSeconds float64) {
	for _, g := range l.Gathers {
		rows := s.tokensF
		if g.TokenSharded && k.layout.ModelSequenceParallel {
			rows = s.spRows
		}
		perToken := float64(g.RowsPerToken)
		switch g.Shard {
		case model.ShardRows:
			perToken /= k.tp
		case model.ShardHeads:
			perToken = math.Ceil(perToken / float64(k.gatherShards(g)))
		}
		read := rows * perToken * float64(g.RowBytes)
		bytes += rows * float64(g.OutBytesPerToken)
		if k.gatherOnHost(g) && k.hostGatherBytesPerSecond > 0 {
			hostSeconds += read / k.hostGatherBytesPerSecond
			continue
		}
		bytes += read
	}
	return bytes, hostSeconds
}

// gatherOnHost reports whether a table resolves to host memory: its stated residency,
// unless the engine setting it names moves it.
func (k *Kernel) gatherOnHost(g price.PlannedGather) bool {
	if g.Control == model.ResidencyEngramCPUOffload {
		return k.engramOnHost
	}
	return g.Residency == model.ResidencyHost
}

// gatherShards is how many ways a head-split table divides.
func (k *Kernel) gatherShards(g price.PlannedGather) int {
	if g.Control == model.ResidencyEngramCPUOffload && k.engramShards > 0 {
		return k.engramShards
	}
	return max(k.layout.TP, 1)
}

// recurrentSeconds is one recurrent layer's mixer time this step, decode rows and prefill
// tokens each at their own phase's form.
//
// Decode rows take the measured decode pair. Prefill tokens take the measured prefill pair
// where the registry carries one; where it does not, they are priced as the chunked scan
// the engine runs, from its FLOPs on the BF16 ramp, and the kernel records that it did
// (liftStepTerms). A chunked delta-rule scan over chunks of C tokens does, per token and
// head of key width dk and value width dv, its intra-chunk products against the chunk
// (2C(dk + dv) multiply-adds with the key-key term of the WY transform) and three state
// products of dk x dv (the output read, the update and the WY correction): 2(2C(dk + dv))
// + 6 dk dv FLOPs, at C = 64, the chunk size of vLLM's chunked KDA and GDN kernels. A
// step's heads are its rank's tp-th share.
func (k *Kernel) recurrentSeconds(l *price.PlannedLayer, s *stepShape) float64 {
	var t float64
	if s.decodeTokens > 0 {
		if fr, ok := k.recurrent[l.RecurrentKind]; ok {
			t += fr.floor.Seconds() + float64(s.decodeTokens)/fr.rate
		}
	}
	if s.prefillTokens <= 0 {
		return t
	}
	if fr, ok := k.recurrentPrefill[l.RecurrentKind]; ok {
		return t + fr.floor.Seconds() + float64(s.prefillTokens)/fr.rate
	}
	const chunk = 64
	dk := float64(l.RecurrentStateSize)
	dv := float64(l.RecurrentHeadDim)
	if dv <= 0 {
		dv = dk
	}
	heads := math.Max(float64(l.RecurrentHeads)/k.tp, 1)
	perToken := heads * (4*chunk*(dk+dv) + 6*dk*dv)
	tokens := float64(s.prefillTokens)
	fit := k.prefillFit
	if eff := price.Efficiency(tokens, fit.epsMax, fit.mHalf); eff > 0 && fit.peak > 0 {
		t += tokens * perToken / (fit.peak * eff) * s.smDerate
	}
	return t
}

// expertPadding is how much a tensor-parallel slice of the routed experts grows when the
// kernel that serves them pads it. On a part without a four-bit peak, vLLM v0.31.0 serves
// MXFP4 (and NVFP4) experts with Marlin W4A16, which pads each partition's intermediate width up to a
// multiple of 128 (vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:752-757): at TP8
// with expert parallelism off, DeepSeek-V4.1's 2,304 / 8 = 288 pads to 384, a third more
// expert weight and work a rank runs. With expert parallelism a rank holds whole experts
// and nothing is padded.
func (k *Kernel) expertPadding(l *price.PlannedLayer) float64 {
	d := l.RoutedFormat.DType
	if k.expertTensorShards <= 1 || l.ExpertInner <= 0 || k.chip.NVFP4Peak > 0 ||
		(d != model.DTypeMXFP4 && d != model.DTypeNVFP4) {
		return 1
	}
	slice := float64(l.ExpertInner) / k.expertTensorShards
	return math.Ceil(slice/128) * 128 / slice
}

// sparseCausalFLOPs is the prefill attention work of a layer whose query reads a bounded
// set of keys, in causalFLOPs' units: for a query at position p,
//
//	min(p + 1, window) + min((p + 1) / ratio, topk)
//
// keys, the window term absent where the layer states no window and the top-k cap absent
// where it selects nothing. A layer that selects a top-k reads it flat in context; one that
// compresses and selects nothing reads the whole compressed stream.
func sparseCausalFLOPs(chunks []chunk, window, topk, ratio int) float64 {
	pairs := 0.0
	if window > 0 {
		pairs = windowedCausalFLOPs(chunks, window) / 4
	}
	r := float64(max(ratio, 1))
	cap := math.Inf(1)
	if topk > 0 {
		cap = float64(topk)
	}
	for _, c := range chunks {
		// Positions p+1 in [lo, hi]: the queries read min(x / r, cap) keys each.
		lo, hi := float64(c.prefix+1), float64(c.prefix+c.sched)
		if hi < lo {
			continue
		}
		cross := math.Min(hi, cap*r) // last position still below the cap
		if cross >= lo {
			// Summed over the positions, less the half-token the continuum convention of
			// the unbounded term (s*c + s^2/2) does not count, so a cap that never binds
			// reproduces it.
			n := cross - lo + 1
			pairs += ((lo+cross)/2 - 0.5) * n / r
			lo = cross + 1
		}
		if hi >= lo && !math.IsInf(cap, 1) {
			pairs += (hi - lo + 1) * cap
		}
	}
	return 4 * pairs
}

// priceHead prices the head once per step, sharded over the vocabulary: its GEMMs on their
// formats' token ramps and their weight reads, the traffic of the elementwise nodes the
// graph states a width for, and its collectives. It returns the stage time.
//
// A head norm the graph states no width for is not charged, as it never was: it is one
// hidden-width read and write a step, small beside the LM head. A node that states one --
// DeepSeek-V4.1's mHC collapse, tens of kilobytes a token -- is.
func (k *Kernel) priceHead(head price.PlannedLayer, s *stepShape,
	perResource *[numResources]float64) float64 {
	var headSM, weightBytes float64
	elementwise := s.tokensF * head.ElementwiseReplicatedBytesPerToken
	if k.layout.ModelSequenceParallel {
		elementwise += s.spRows * head.ElementwiseTokenShardedBytesPerToken
	} else {
		elementwise += s.tokensF * head.ElementwiseTokenShardedBytesPerToken
	}
	for _, g := range head.DenseGEMMs {
		div := k.tp
		if g.Replicated {
			div = 1
		}
		weightBytes += g.WeightReadBytes / div
		fit := k.fit(g.Format, false)
		if eff := price.Efficiency(s.tokensF, fit.epsMax, fit.mHalf); eff > 0 {
			headSM += s.tokensF * g.FLOPsPerToken / div / (fit.peak * eff) * s.smDerate
		}
	}
	headHBM := (weightBytes + elementwise) / k.hbmBytesPerSecond
	var onNode, crossNode float64
	for _, c := range head.Collectives {
		key := collKey{Op: c.Op, Group: c.Group}
		if k.crossesNodes(key) {
			crossNode += k.collectiveSeconds(key, s.tokensF*c.BytesPerToken*k.spanFor(key))
		} else {
			onNode += k.collectiveSeconds(key, s.tokensF*c.BytesPerToken)
		}
	}
	perResource[idxSM] += headSM
	perResource[idxHBM] += headHBM
	perResource[idxNVLink] += onNode
	perResource[idxNIC] += crossNode
	return math.Max(math.Max(headSM, headHBM), math.Max(onNode, crossNode))
}
