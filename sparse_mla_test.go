package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// A DSA MODEL RUNS 64-TOKEN BLOCKS. Its indexer's backend takes exactly 64
// (vllm/v1/attention/backends/mla/indexer.py:203-204 at v0.31.0), so the engine picks 64 when
// no size is stated and refuses a stated size that is not a multiple of 64. Asserted on glm5:
// 16 and 32 refused, 128 accepted, and an unstated size charged as 64 -- 1 and 64 tokens
// occupy one page, 65 occupy two -- and disclosed as an assumption. A non-DSA model keeps the
// stock 16-token page.
func TestADSAModelRunsSixtyFourTokenBlocks(t *testing.T) {
	build := func(fixtureName string, block int) (*Kernel, error) {
		t.Helper()
		in := fixtureInputs(t, fixtureName)
		in.Deployment.Pools[0].Engine.BlockSize = block
		return New(in)
	}
	for _, block := range []int{16, 32} {
		if _, err := build(dcpSparseFixture, block); err == nil {
			t.Errorf("block_size %d on glm5 was admitted; the engine refuses it", block)
		}
	}
	if _, err := build(dcpSparseFixture, 128); err != nil {
		t.Errorf("block_size 128 on glm5 was refused: %v", err)
	}
	pages := func(k *Kernel, tokens ...int) []int64 {
		out := make([]int64, len(tokens))
		for i, n := range tokens {
			out[i] = k.SequenceVariableBytes(n)
		}
		return out
	}
	glm, err := build(dcpSparseFixture, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p := pages(glm, 1, 64, 65); p[0] != p[1] || p[2] != 2*p[1] {
		t.Errorf("unstated block_size on glm5: 1, 64 and 65 tokens charged %v bytes; want "+
			"one 64-token page, one, and two", p)
	}
	disclosed := false
	for _, a := range glm.assumptions {
		disclosed = disclosed || a.Name == "block_size"
	}
	if !disclosed {
		t.Error("the unstated block size on glm5 is not disclosed as an assumption")
	}
	mla, err := build(dcpMLAFixture, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p := pages(mla, 1, 16, 17); p[0] != p[1] || p[2] != 2*p[1] {
		t.Errorf("unstated block_size on deepseek-v3 (not DSA): 1, 16 and 17 tokens charged "+
			"%v bytes; want the stock 16-token page", p)
	}
}

// THE BACKEND DECIDES WHICH CONTEXT-PARALLEL LAYOUTS A DSA MODEL STARTS. Each case is a
// layout and whether vLLM v0.31.0 starts it, on glm5 (64 query heads):
//
//   - Hopper under DCP leaves only FlashMLA-sparse, which needs a quantized cache it lists
//     ("fp8", not fp8_e4m3) and fewer than 32 heads a rank (flashmla_sparse.py:409-460,
//     :745-751). Without DCP, fp8_e4m3 starts on FlashInfer's SM90 backend.
//   - nvfp4_ds_mla is SM100-only (:218-222).
//   - Blackwell leads with FlashInfer for a quantized cache and for a 16-bit one at 16 heads
//     a rank or fewer; past that FlashMLA-sparse leads, and a 16-bit cache under DCP fails it
//     (vllm/platforms/cuda.py:95-131). FlashInfer refuses PCP with DCP, which FlashMLA-sparse
//     serves only on fp8_ds_mla (flashinfer_mla_sparse.py:117-125), so nvfp4_ds_mla, which
//     it keeps as stated, does not start there.
//   - With PCP, the gathered head count is heads x tp only where dcp spans tp x pcp, and the
//     heads otherwise: at tp2/pcp2/dcp4, 32 local heads gather to 64 and pad alike.
//
// Every DCP layout states the ag_rs combine without a replicated query. GLM-5's model hook
// would otherwise choose a2a with q_replicate (vllm/model_executor/models/config.py:43-50),
// which FlashMLA-sparse refuses (flashmla_sparse.py:416-422); a GLM-5 deployment on that
// backend has to state ag_rs to start at all.
func TestTheBackendDecidesWhichContextParallelLayoutsADSAModelStarts(t *testing.T) {
	variant := func(tp, pcp, dcp int, cache string, blackwell bool) (*Kernel, error) {
		t.Helper()
		return dcpVariant(t, dcpSparseFixture, tp, pcp, dcp, func(in *Inputs) {
			e := &in.Deployment.Pools[0].Engine
			if dcp > 1 {
				off := false
				e.DCPCommBackend, e.DCPQReplicate = "ag_rs", &off
			}
			in.Deployment.Pools[0].Engine.CacheDType = cache
			if blackwell {
				chip := *in.Chip
				chip.NVFP4Peak = 1
				in.Chip = &chip
			}
		})
	}
	for _, c := range []struct {
		name         string
		tp, pcp, dcp int
		cache        string
		blackwell    bool
		starts       bool
	}{
		{"hopper dcp, fp8, 16 heads a rank", 4, 1, 2, "fp8", false, true},
		{"hopper dcp, fp8, 32 heads a rank", 2, 1, 2, "fp8", false, false},
		{"hopper dcp, fp8_e4m3", 8, 1, 2, "fp8_e4m3", false, false},
		{"hopper dcp, auto", 4, 1, 2, "auto", false, false},
		{"hopper dcp, bfloat16", 4, 1, 2, "bfloat16", false, false},
		{"hopper no dcp, fp8_e4m3", 8, 1, 1, "fp8_e4m3", false, true},
		{"hopper no dcp, auto", 8, 1, 1, "auto", false, true},
		{"hopper, nvfp4_ds_mla", 8, 1, 1, "nvfp4_ds_mla", false, false},
		{"blackwell, nvfp4_ds_mla", 8, 1, 1, "nvfp4_ds_mla", true, true},
		{"blackwell pcp+dcp, fp8", 2, 4, 4, "fp8", true, true},
		{"blackwell pcp+dcp, bfloat16", 2, 4, 4, "bfloat16", true, false},
		{"blackwell pcp+dcp, nvfp4_ds_mla", 2, 4, 4, "nvfp4_ds_mla", true, false},
		{"hopper pcp+dcp spanning pcp, 32 heads a rank", 2, 2, 2, "fp8", false, true},
		{"hopper pcp+dcp spanning tp x pcp, 32 heads gather to 64", 2, 2, 4, "fp8", false,
			true},
		{"blackwell dcp, fp8, 32 heads a rank", 2, 1, 2, "fp8", true, true},
		{"blackwell dcp, auto, 16 heads a rank", 4, 1, 2, "auto", true, true},
		{"blackwell dcp, auto, 32 heads a rank", 2, 1, 2, "auto", true, false},
	} {
		_, err := variant(c.tp, c.pcp, c.dcp, c.cache, c.blackwell)
		if c.starts && err != nil {
			t.Errorf("%s: refused, but the engine starts it: %v", c.name, err)
		}
		if !c.starts && err == nil {
			t.Errorf("%s: admitted, but the engine refuses it", c.name)
		}
	}
	// What a started layout stores. Where FlashMLA-sparse serves Blackwell's PCP+DCP the "fp8"
	// cache is the packed 656-byte cell; with PCP alone FlashInfer serves it as stated, 576.
	cell := func(pcp, dcp int) int64 {
		t.Helper()
		k, err := variant(2, pcp, dcp, "fp8", true)
		if err != nil {
			t.Fatal(err)
		}
		return int64(k.kvBytesPerToken / float64(k.kvLayers))
	}
	if got := cell(4, 4); got != 656 {
		t.Errorf("blackwell pcp+dcp fp8: %d bytes a token per layer, want fp8_ds_mla's 656", got)
	}
	if got := cell(4, 1); got != 576 {
		t.Errorf("blackwell pcp fp8: %d bytes a token per layer, want plain fp8's 576", got)
	}
}

// Layouts no catalog fixture reaches, asserted on the selection directly. A 128-head DSA
// model (DeepSeek-V3.2) at tp=8 holds 16 heads a rank: at dcp=8 they gather to 128, which pads
// to a different fp8 decode size from the local 16, and FlashMLA-sparse refuses it; at dcp=4
// they gather to 64 and it starts (flashmla_sparse.py:436-460, :684-687). With PCP the gathered
// count is heads x tp when dcp spans tp x pcp: at tp4/pcp2/dcp8, 32 local heads gather to
// 128 and are refused, where pcp2/dcp2 leaves them at 32 (vLLM admits dcp only as 1, pcp or
// tp x pcp under PCP, vllm/config/parallel.py:571-577). The a2a combine is refused on
// FlashMLA-sparse (:416-422), and a dtype no backend lists has none.
func TestSparseMLABackendRefusesWhatNoBackendServes(t *testing.T) {
	starts := func(r sparseMLABackendRequest) bool {
		if r.dcpComm == "" {
			r.dcpComm = resolve.DCPAllGatherReduceScatter
		}
		_, _, err := sparseMLABackend(r)
		return err == nil
	}
	for _, c := range []struct {
		name string
		r    sparseMLABackendRequest
		want bool
	}{
		{"128 heads, tp8 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 8, pcp: 1}, false},
		{"128 heads, tp8 dcp4", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 4, pcp: 1}, true},
		{"128 heads, tp4 pcp2 dcp8", sparseMLABackendRequest{cache: "fp8", headsPerRank: 32,
			tp: 4, dcp: 8, pcp: 2}, false},
		{"128 heads, tp8 pcp2 dcp2", sparseMLABackendRequest{cache: "fp8", headsPerRank: 16,
			tp: 8, dcp: 2, pcp: 2}, true},
		{"a2a combine", sparseMLABackendRequest{cache: "fp8", headsPerRank: 8, tp: 8, dcp: 2,
			pcp: 1, dcpComm: resolve.DCPAllToAll}, false},
		{"fp8_e5m2 on hopper", sparseMLABackendRequest{cache: "fp8_e5m2", headsPerRank: 8,
			tp: 8, dcp: 1, pcp: 1}, false},
	} {
		if got := starts(c.r); got != c.want {
			t.Errorf("%s: starts %v, want %v", c.name, got, c.want)
		}
	}
}

// Where a DSA model on Hopper lands on FlashInfer's SM90 sparse MLA, which needs a FlashInfer
// release the kernel cannot see, the choice is disclosed; where it lands elsewhere, it is not.
func TestTheFlashInferSM90ChoiceIsDisclosed(t *testing.T) {
	disclosed := func(cache string) bool {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		in.Deployment.Pools[0].Engine.CacheDType = cache
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		for _, a := range k.assumptions {
			if a.Name == "sparse_mla_backend" {
				return true
			}
		}
		return false
	}
	if !disclosed("fp8_e4m3") {
		t.Error("fp8_e4m3 on h200 selects FlashInfer's SM90 sparse MLA undisclosed")
	}
	if disclosed("fp8") {
		t.Error("fp8 on h200 selects FlashMLA-sparse, but a FlashInfer assumption was disclosed")
	}
}

// A REPACKED CACHE IS REPORTED AS RESOLVED. Where the backend serves a stated "fp8" cache as
// fp8_ds_mla (h200), Resolved says so, since the priced layout is not the one requested;
// where it serves it as stated (the same chip marked data-center Blackwell), nothing is
// reported.
func TestARepackedCacheIsReportedAsResolved(t *testing.T) {
	repack := func(blackwell bool) string {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		in.Deployment.Pools[0].Engine.CacheDType = "fp8"
		if blackwell {
			chip := *in.Chip
			chip.NVFP4Peak = 1
			in.Chip = &chip
		}
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range k.Resolved().Overrides {
			if o.Field == "cache_dtype" {
				return o.Resolved
			}
		}
		return ""
	}
	if got := repack(false); got != "fp8_ds_mla" {
		t.Errorf("fp8 on h200: Resolved reports the cache as %q, want fp8_ds_mla", got)
	}
	if got := repack(true); got != "" {
		t.Errorf("fp8 on Blackwell: Resolved reports a repack to %q; FlashInfer serves it "+
			"as stated", got)
	}
}
