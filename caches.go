package latencykernel

import (
	"math"

	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// Per-layer KV caches, sized from what each layer states it owns.
//
// A graph that states an attention node's caches (blis-schemas v0.3.0 model.Cache) says
// which layer allocates each cache, how large one stored record is, how many context
// tokens a record covers, and whether a window or a fixed entry count bounds it. That is
// what residency, P/D transfer volume and offload volume follow, and none of it is
// recoverable from a read shape. For DeepSeek-V4.1-Flash on vLLM v0.31.0 the difference is
// an order of magnitude: one per-model figure times every attention layer charged 20,480
// B a context token, where only four layers own a compressed cache, a compressed record
// covers two tokens on three of them, and every layer's 128-token window is a bounded
// ring rather than a cache that grows with context -- about 1,790 B a token plus a fixed
// ring term per sequence.
//
// A layer that states no caches keeps the inference it always had (Kernel.kvBytesPerToken),
// so a graph opts in per node and no other graph moves.

// layerKV is one planned layer's KV, as the pricer reads it.
type layerKV struct {
	// stated is a layer whose attention states its caches; perToken is otherwise its
	// inferred cache's bytes per context token, per rank.
	stated   bool
	perToken float64
}

// cacheTerm is one cache some layers own, with its paging resolved.
type cacheTerm struct {
	cache model.Cache
	// entryBytes is the record under the cache dtype that runs, and pageBytes one page of
	// it: block tokens hold block / tokens_per_entry records, rounded up to the page
	// alignment.
	entryBytes int
	block      int
	pageBytes  float64
	// count is how many layers own it.
	count int
	// draft is a speculator layer's cache, resident only while the deployment speculates.
	draft bool
}

// layerCaches returns every cache a layer's attention kernels state, owned or read.
func layerCaches(l price.PlannedLayer) []price.PlannedCache {
	var out []price.PlannedCache
	for _, a := range l.Attentions {
		out = append(out, a.Caches...)
	}
	return out
}

// cacheTerms resolves the caches a layer owns, count times over.
func (k *Kernel) cacheTerms(stated []price.PlannedCache, count int) []cacheTerm {
	var out []cacheTerm
	for _, c := range stated {
		if !c.Owned {
			// A reader adds read traffic and nothing resident: the cache is its owner's.
			continue
		}
		block := c.BlockSize
		if block < 1 {
			block = max(k.blockSize, 1)
		}
		entries := (block + c.EntryTokens() - 1) / c.EntryTokens()
		page := float64(entries * c.EntryBytes)
		if a := c.PageAlignment; a > 0 {
			page = math.Ceil(page/float64(a)) * float64(a)
		}
		out = append(out, cacheTerm{cache: c.Cache, entryBytes: c.EntryBytes, block: block,
			pageBytes: page, count: count})
	}
	return out
}

// pages is how many pages a cache holds for tokens context tokens.
func (t cacheTerm) pages(tokens int) int {
	if tokens <= 0 {
		return 0
	}
	return (tokens + t.block - 1) / t.block
}

// growing reports whether the cache grows with context: a kv or index cache.
func (t cacheTerm) growing() bool {
	return t.cache.Kind == model.CacheKV || t.cache.Kind == model.CacheIndex
}

// variableBytes is what the context-growing caches hold for a sequence of tokens, after
// decode-context parallelism's shard of the token count.
func (k *Kernel) cacheVariableBytes(tokens int) float64 {
	var total float64
	for _, t := range k.caches {
		if !t.growing() || (t.draft && !k.speculating) {
			continue
		}
		local := tokens
		if k.layout.DCP > 1 {
			// An entry is not divided across tensor-parallel ranks, and decode-context
			// parallelism gives each rank 1/dcp of the entries (blis-schemas model.Cache,
			// vllm/v1/kv_cache_interface.py max_num_blocks_per_req).
			local = (tokens + k.layout.DCP - 1) / k.layout.DCP
		}
		total += float64(t.pages(local)) * t.pageBytes * float64(t.count)
	}
	return total
}

// cacheFixedBytes is what the bounded caches hold per sequence: each window ring at the
// pages the engine lets a request hold, and each state cache's fixed entries.
//
// The ring is cdiv(window - 1 + in-flight tokens, block) + 1 pages
// (max_admission_blocks_per_request, vllm/v1/kv_cache_interface.py at v0.31.0), which the
// rules pack applies (WindowPagesPerRequest). A request has 1 + num_spec_tokens tokens in
// flight in a decode step, so DSpark's five draft tokens make a 128-token ring of 32-token
// pages six pages rather than five.
func (k *Kernel) cacheFixedBytes() float64 {
	inFlight := 1 + k.numSpecTokens()
	var total float64
	for _, t := range k.caches {
		if t.draft && !k.speculating {
			continue
		}
		switch t.cache.Kind {
		case model.CacheWindow:
			p := k.rules.WindowPagesPerRequest(t.cache, k.blockSize, inFlight, 0, k.maxModelLen)
			total += float64(p) * t.pageBytes * float64(t.count)
		case model.CacheState:
			total += float64(t.cache.Entries*t.entryBytes) * float64(t.count)
		}
	}
	return total
}

// cacheTransferBytes is what a P/D handoff moves of the stated caches for a sequence of
// tokens: the target's owned caches only, as NIXL moves them -- the growing caches in full,
// each window cache clipped to cdiv(window, block) + 1 pages, and the state caches whole
// (vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:134-146 at v0.31.0).
// A reader's cache is its owner's and crosses once; a speculator's caches do not cross,
// since the decode pool's drafter writes its own.
func (k *Kernel) cacheTransferBytes(tokens int) float64 {
	var total float64
	for _, t := range k.caches {
		if t.draft {
			continue
		}
		var pages int
		switch t.cache.Kind {
		case model.CacheKV, model.CacheIndex:
			pages = t.pages(tokens)
		case model.CacheWindow:
			pages = min(t.pages(tokens), (t.cache.Window+t.block-1)/t.block+1)
		case model.CacheState:
			total += float64(t.cache.Entries*t.entryBytes) * float64(t.count)
			continue
		}
		total += float64(pages) * t.pageBytes * float64(t.count)
	}
	return total
}

// statedReadBytes is what one decode request at context ctx reads from a layer's stated
// caches through attention a: each window cache's most recent min(ctx, window) records,
// and each growing cache's records -- ceil(ctx / tokens_per_entry) of them, capped at the
// top-k the layer selects where it selects one. Only the caches the kernel reads count: the
// indexer reads the index cache, attention the others.
//
// The shard of decode-context parallelism applies to the growing part only, the part
// stored by token position.
func (k *Kernel) statedReadBytes(caches []price.PlannedCache, topk, ctx int) float64 {
	var total float64
	for _, c := range caches {
		switch c.Kind {
		case model.CacheWindow:
			w := ctx
			if c.Window > 0 {
				w = min(ctx, c.Window)
			}
			total += float64(w) * float64(c.EntryBytes)
		case model.CacheKV, model.CacheIndex:
			entries := (ctx + c.EntryTokens() - 1) / c.EntryTokens()
			if topk > 0 {
				entries = min(entries, topk)
			}
			if k.layout.DCP > 1 && c.Kind == model.CacheKV {
				entries = int(dcpLocalTokens([]int{entries}, k.layout.DCP,
					k.decodeContext.Interleave))
			}
			total += float64(entries) * float64(c.EntryBytes)
		}
	}
	return total
}

// numSpecTokens is the draft tokens per request per step the deployment runs, zero when it
// does not speculate.
func (k *Kernel) numSpecTokens() int {
	if s := k.pool.Engine.Speculative; s != nil && s.NumSpecTokens > 0 {
		return s.NumSpecTokens
	}
	return 0
}
