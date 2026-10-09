package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// A DSA MODEL RUNS 64-TOKEN BLOCKS. Its indexer's backend takes exactly 64
// (vllm/v1/attention/backends/mla/indexer.py:203-204 at v0.31.0), so the engine picks 64 when
// no size is stated and refuses a stated size 64 does not divide (select_common_block_size,
// vllm/v1/worker/utils.py:330-391). Asserted on glm5: 16 and 32 refused, 128 accepted, and an
// unstated size charged as 64 -- one token occupies a whole 64-token page, and the choice is
// disclosed as an assumption. A non-DSA model keeps the stock default of 16.
func TestADSAModelRunsSixtyFourTokenBlocks(t *testing.T) {
	build := func(fixtureName string, block int) (*Kernel, error) {
		t.Helper()
		in := fixtureInputs(t, fixtureName)
		in.Deployment.Pools[0].Engine.BlockSize = block
		return New(in)
	}
	for _, block := range []int{16, 32} {
		if _, err := build(dcpSparseFixture, block); err == nil ||
			!strings.Contains(err.Error(), "refuses") {
			t.Errorf("block_size %d on glm5 was not refused (err %v)", block, err)
		}
	}
	if _, err := build(dcpSparseFixture, 128); err != nil {
		t.Errorf("block_size 128 on glm5 was refused: %v", err)
	}
	pageOfOne := func(fixtureName string) (int64, int64) {
		t.Helper()
		k, err := build(fixtureName, 0)
		if err != nil {
			t.Fatal(err)
		}
		return k.SequenceVariableBytes(1), k.SequenceVariableBytes(16)
	}
	one, sixteen := pageOfOne(dcpSparseFixture)
	if one != sixteen {
		t.Errorf("unstated block_size on glm5: one token charged %d bytes and sixteen %d; "+
			"both occupy one 64-token page", one, sixteen)
	}
	k, _ := build(dcpSparseFixture, 0)
	found := false
	for _, a := range k.assumptions {
		if a.Name == "block_size" && strings.HasPrefix(a.Scope, "64:") {
			found = true
		}
	}
	if !found {
		t.Error("the 64-token block on glm5 is not disclosed as an assumption")
	}
	one, sixteen = pageOfOne(dcpMLAFixture)
	if seventeen := func() int64 {
		k, _ := build(dcpMLAFixture, 0)
		return k.SequenceVariableBytes(17)
	}(); one != sixteen || seventeen == sixteen {
		t.Errorf("unstated block_size on deepseek-v3 (not DSA): pages of 1, 16 and 17 tokens "+
			"charged %d, %d and %d; want the stock 16-token page", one, sixteen, seventeen)
	}
}

// THE BACKEND DECIDES WHICH CONTEXT-PARALLEL LAYOUTS A DSA MODEL STARTS. Under DCP on Hopper
// only FlashMLA-sparse remains (FlashAttention's and FlashInfer's SM90 sparse MLA do not
// support DCP), and FlashMLA-sparse refuses DCP with 32 or more heads a rank, and does not
// list fp8_e4m3. On data-center Blackwell, FlashInfer refuses PCP with DCP, so FlashMLA-sparse
// serves it -- repacking an "fp8" cache to fp8_ds_mla and refusing a 16-bit one
// (vllm/v1/attention/backends/mla/flashmla_sparse.py:409-435, flashinfer_mla_sparse.py:117-125
// at v0.31.0). glm5 has 64 query heads.
func TestTheBackendDecidesWhichContextParallelLayoutsADSAModelStarts(t *testing.T) {
	variant := func(tp, pcp, dcp int, cache string, blackwell bool) (*Kernel, error) {
		t.Helper()
		return dcpVariant(t, dcpSparseFixture, tp, pcp, dcp, func(in *Inputs) {
			in.Deployment.Pools[0].Engine.CacheDType = cache
			if blackwell {
				chip := *in.Chip
				chip.NVFP4Peak = 1
				in.Chip = &chip
			}
		})
	}
	for _, c := range []struct {
		name          string
		tp, pcp, dcp  int
		cache         string
		blackwell, ok bool
	}{
		{"hopper dcp, 16 heads a rank", 4, 1, 2, "fp8", false, true},
		{"hopper dcp, 32 heads a rank", 2, 1, 2, "fp8", false, false},
		{"hopper dcp, fp8_e4m3", 8, 1, 2, "fp8_e4m3", false, false},
		{"hopper no dcp, fp8_e4m3", 8, 1, 1, "fp8_e4m3", false, true},
		{"blackwell pcp+dcp, fp8", 2, 4, 4, "fp8", true, true},
		{"blackwell pcp+dcp, bf16", 2, 4, 4, "bfloat16", true, false},
		{"blackwell dcp, 32 heads a rank, fp8", 2, 1, 2, "fp8", true, true},
	} {
		_, err := variant(c.tp, c.pcp, c.dcp, c.cache, c.blackwell)
		if c.ok && err != nil {
			t.Errorf("%s: refused, but the engine starts it: %v", c.name, err)
		}
		if !c.ok && (err == nil || !strings.Contains(err.Error(), "refuses")) {
			t.Errorf("%s: not refused (err %v), but the engine does not start it", c.name, err)
		}
	}
	// Where FlashMLA-sparse serves Blackwell's PCP+DCP, the "fp8" cache is the packed cell;
	// at dcp=1 FlashInfer serves it as stated.
	page := func(pcp, dcp int) int64 {
		t.Helper()
		k, err := variant(2, pcp, dcp, "fp8", true)
		if err != nil {
			t.Fatal(err)
		}
		return int64(k.kvBytesPerToken / float64(k.kvLayers))
	}
	if got := page(4, 4); got != 656 {
		t.Errorf("blackwell pcp+dcp fp8: %d bytes a token per layer, want fp8_ds_mla's 656", got)
	}
	if got := page(4, 1); got != 576 {
		t.Errorf("blackwell pcp fp8: %d bytes a token per layer, want plain fp8's 576", got)
	}
}

// The selection itself, on the combinations a fixture cannot reach: the a2a combine is
// refused on FlashMLA-sparse, and an unlisted dtype has no backend at all.
func TestSparseMLABackendRefusesWhatNoBackendServes(t *testing.T) {
	_, _, err := sparseMLABackend(sparseMLABackendRequest{cache: "fp8", headsPerRank: 8,
		dcp: 2, pcp: 1, dcpComm: resolve.DCPAllToAll})
	if err == nil {
		t.Error("FlashMLA-sparse under DCP with the a2a combine was not refused")
	}
	if _, _, err := sparseMLABackend(sparseMLABackendRequest{cache: "fp8_e5m2",
		headsPerRank: 8, dcp: 1, pcp: 1}); err == nil {
		t.Error("an fp8_e5m2 sparse cache on Hopper found a backend; none lists it")
	}
	b, layout, err := sparseMLABackend(sparseMLABackendRequest{cache: "auto",
		headsPerRank: 8, dcp: 1, pcp: 1})
	if err != nil || b != backendFlashAttnMLASparse || layout != "auto" {
		t.Errorf("a 16-bit cache on Hopper chose %s with layout %q (err %v); want "+
			"FlashAttention's sparse MLA, as stated", b, layout, err)
	}
}
