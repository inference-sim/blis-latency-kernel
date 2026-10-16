package latencykernel

import (
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// contractEntries are blis-registry PR #47's DeepSeek-V4.1-Flash h200 entries as
// docs/COEFF_CONTRACT.md v2.1 states them, model-scoped as the registry scopes them.
func contractEntries(families ...string) []coefficient.Entry {
	all := map[string][]coefficient.Entry{
		"sparse_mla": {
			{Name: "attention_prefill_floor_sparse_mla", Value: 14.8, Units: "us_per_transfer"},
			{Name: "attention_prefill_per_request_sparse_mla", Value: 1.323, Units: "us_per_request"},
			{Name: "attention_prefill_per_token_sparse_mla", Value: 0.049, Units: "us_per_token"},
			{Name: "attention_prefill_rate_sparse_mla", Value: 3159685, Units: "bytes_per_us"},
		},
		"block_index_scores": {
			{Name: "attention_prefill_floor_block_index_scores", Value: 4.1, Units: "us_per_transfer"},
			{Name: "attention_prefill_per_token_block_index_scores", Value: 0.00466, Units: "us_per_token"},
			{Name: "attention_prefill_rate_block_index_scores", Value: 7298637, Units: "bytes_per_us"},
		},
		"mhc": {
			{Name: "mhc_floor_pre", Value: 8.9, Units: "us_per_transfer"},
			{Name: "mhc_rate_pre", Value: 1799384, Units: "bytes_per_us"},
			{Name: "mhc_floor_post", Value: 3.0, Units: "us_per_transfer"},
			{Name: "mhc_rate_post", Value: 1850611, Units: "bytes_per_us"},
		},
		"host": {
			{Name: "host_admission_per_token", Value: 2.67, Units: "us_per_token", Method: "measured"},
			{Name: "host_admission_per_request", Value: 0, Units: "us_per_request", Method: "not_charged"},
		},
	}
	var out []coefficient.Entry
	for _, f := range families {
		for _, e := range all[f] {
			if e.Method == "" {
				e.Method = "assumed"
			}
			if e.Method != "measured" {
				e.Rationale = "test"
			} else {
				e.Sources = []coefficient.Source{{Kind: "discussion", Cite: "test", Role: "primary"}}
			}
			e.Scope = coefficient.Scope{Hardware: []string{"h200"},
				Model: []string{"deepseek-v4.1-flash"}}
			out = append(out, e)
		}
	}
	return out
}

// withEntries is the V4.1 fixture with entries appended as their own set, less any entry
// named in drop.
func withEntries(t *testing.T, entries []coefficient.Entry, drop ...string) *Kernel {
	t.Helper()
	in := fixtureInputs(t, v41)
	var kept []coefficient.Entry
	for _, e := range entries {
		dropped := false
		for _, d := range drop {
			dropped = dropped || e.Name == d
		}
		if !dropped {
			kept = append(kept, e)
		}
	}
	if len(kept) > 0 {
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...),
			&coefficient.Set{Kind: "CoefficientSet", Name: "test-v41", Coefficients: kept})
	}
	return build(t, in)
}

// priceOne prices one of k's target layer kinds, once, over b, returning its per-resource
// time. layer, when given, replaces the planned layer, so a test can price an edited copy.
func priceOne(t *testing.T, k *Kernel, id string, b kernel.Batch,
	layer *price.PlannedLayer) [numResources]float64 {
	t.Helper()
	for i, l := range k.plan.Layers {
		if l.ID != id {
			continue
		}
		if layer != nil {
			l = *layer
		}
		l.Count = 1
		s := k.shapeOf(b)
		var per [numResources]float64
		k.priceLayers([]price.PlannedLayer{l}, [][]gemmFit{k.layerFits[i]},
			[]layerKV{k.layerKV[i]}, &s, &per)
		return per
	}
	t.Fatalf("no layer kind %q", id)
	return [numResources]float64{}
}

func layerOf(t *testing.T, k *Kernel, id string) price.PlannedLayer {
	t.Helper()
	for _, l := range k.plan.Layers {
		if l.ID == id {
			return l
		}
	}
	t.Fatalf("no layer kind %q", id)
	return price.PlannedLayer{}
}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

// chunkOn is one prefill chunk of sched tokens on prefix computed tokens of a longer prompt.
func chunkOn(sched, prefix int) kernel.Batch {
	return kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: sched, Computed: prefix, PromptLen: prefix + sched + 10000}}}
}

// THE SPARSE-MLA PREFILL FORM IS THE CONTRACT'S, AND REPLACES THE GENERIC PAIR. One 8,192-token
// chunk on an 81,808-token prefix reads K = 640 keys a query (128 window + 512 selected, in
// 64-token blocks) at 584 B over 64 padded heads, so the form is
// 14.8 + 1.323 + 0.049 x 8192 + 8192 x 640 x 584 / 3,159,685 us = 1.3865 ms a layer, the
// contract's "1.39 ms / layer". On csa1_moe (no scorer, no mHC entries in this set) the
// layer's SM time moves by exactly that less the generic pair's charge.
func TestTheSparseMLAPrefillFormReplacesTheGenericPair(t *testing.T) {
	b := chunkOn(8192, 81808)
	with := withEntries(t, contractEntries("sparse_mla"))
	without := withEntries(t, nil)
	l := layerOf(t, without, "csa1_moe")
	s := without.shapeOf(b)
	heads := without.attentionHeads(l.AttnQHeads, l.AttnReplicated, l.AttnKind)
	generic, _ := without.prefillAttention(sparseCausalFLOPs(s.prefillChunks, l.AttnWindow,
		l.AttnIndexTopK, l.AttnCompressRatio)*heads*float64(l.AttnHeadDim), &s)
	want := (14.8 + 1.323 + 0.049*8192 + 8192.0*640*584/3159685) * 1e-6
	got := priceOne(t, with, "csa1_moe", b, nil)[idxSM] -
		priceOne(t, without, "csa1_moe", b, nil)[idxSM] + generic
	if !near(got, want) {
		t.Errorf("sparse-MLA prefill form %.4f ms, want the contract's %.4f ms", got*1e3,
			want*1e3)
	}
	// Decode is untouched by a prefill form.
	d := decodeBatch(4, 6, 90000)
	if a, b := with.StepTime(d), without.StepTime(d); a.NoOverlap != b.NoOverlap {
		t.Errorf("a decode step moved from %v to %v under a prefill form", b.NoOverlap,
			a.NoOverlap)
	}
}

// THE INDEXER'S PREFILL FORM IS THE CONTRACT'S. csa1_idx_mask_moe's scorer has ratio 1 and 32
// query heads over 132 B keys, so one 8,192-token chunk on 81,808 reads
// sum_{p=81809}^{90000} p x 132 x 32 / 64 bytes, and the scorer costs
// 4.1 + 0.00466 x 8192 + that / 7,298,637 us. Its SM time moves by exactly that less the
// generic charge (half attention's FLOPs a pair over its heads).
func TestTheIndexerPrefillFormIsTheContracts(t *testing.T) {
	b := chunkOn(8192, 81808)
	with := withEntries(t, contractEntries("block_index_scores"))
	without := withEntries(t, nil)
	l := layerOf(t, without, "csa1_idx_mask_moe")
	s := without.shapeOf(b)
	var a price.PlannedAttention
	for _, x := range l.Attentions {
		if x.Role == "block_index_scores" {
			a = x
		}
	}
	if a.CompressRatio != 1 || a.QHeads != 32 {
		t.Fatalf("scorer ratio %d heads %d; the golden assumes 1 and 32", a.CompressRatio,
			a.QHeads)
	}
	pairs := sparseCausalFLOPs(s.prefillChunks, 0, 0, 1) / 2
	generic, _ := without.prefillAttention(pairs*without.attentionHeads(a.QHeads,
		a.Replicated, a.Kind)*float64(a.HeadDim), &s)
	keys := 8192.0 * (81809 + 90000) / 2
	want := (4.1 + 0.00466*8192 + keys*132*32/64/7298637) * 1e-6
	got := priceOne(t, with, "csa1_idx_mask_moe", b, nil)[idxSM] -
		priceOne(t, without, "csa1_idx_mask_moe", b, nil)[idxSM] + generic
	if !near(got, want) {
		t.Errorf("indexer prefill form %.4f ms, want the contract's %.4f ms", got*1e3,
			want*1e3)
	}
}

// EACH mHC OP IS CHARGED EXACTLY ONCE. With the mHC entries, a layer prices as the same layer
// with its mHC nodes and hc_*fn GEMMs deleted, plus the measured form: nothing of the
// nodes' stated bytes and nothing of the GEMMs' compute or weight read survives beside it.
// On csa1_moe at 4 x 6 decode tokens (sequence parallelism off at this TP8 fixture), its
// two post+pre sites cost 2 x (8.9 + 3.0 + 24 x 40,960 x (1/1,799,384 + 1/1,850,611)) us.
func TestEachMHCOpIsChargedExactlyOnce(t *testing.T) {
	with := withEntries(t, contractEntries("mhc"))
	without := withEntries(t, nil)
	for _, b := range []kernel.Batch{decodeBatch(4, 6, 90000), chunkOn(1808, 88192)} {
		tokens := float64(b.Tokens())
		l := layerOf(t, without, "csa1_moe")
		if len(l.MHC) != 2 {
			t.Fatalf("csa1_moe has %d mHC nodes; the golden assumes 2", len(l.MHC))
		}
		stripped := l
		stripped.DenseGEMMs = nil
		gemms := 0
		for _, g := range l.DenseGEMMs {
			if g.MHCFn {
				gemms++
				continue
			}
			stripped.DenseGEMMs = append(stripped.DenseGEMMs, g)
		}
		if gemms != 2 {
			t.Fatalf("csa1_moe has %d hc_*fn GEMMs; the golden assumes 2", gemms)
		}
		for _, m := range l.MHC {
			stripped.ElementwiseTokenShardedBytesPerToken -= m.Bytes
		}
		// The fit is the fixture's per-GEMM fit list, so drop the stripped GEMMs' fits too.
		i := 0
		for j, ll := range without.plan.Layers {
			if ll.ID == "csa1_moe" {
				i = j
			}
		}
		fits := without.layerFits[i]
		var kept []gemmFit
		for gi, g := range l.DenseGEMMs {
			if !g.MHCFn {
				kept = append(kept, fits[gi])
			}
		}
		s := without.shapeOf(b)
		var bare [numResources]float64
		without.priceLayers([]price.PlannedLayer{withCount(stripped, 1)}, [][]gemmFit{kept},
			[]layerKV{without.layerKV[i]}, &s, &bare)
		form := 2 * (8.9 + 3.0 + tokens*40960*(1/1799384.0+1/1850611.0)) * 1e-6
		got := priceOne(t, with, "csa1_moe", b, nil)
		if !near(got[idxSM], bare[idxSM]+form) {
			t.Errorf("%v tokens: SM %.6f ms, want the layer without its mHC ops %.6f ms "+
				"plus the form %.6f ms", tokens, got[idxSM]*1e3, bare[idxSM]*1e3, form*1e3)
		}
		if !near(got[idxHBM], bare[idxHBM]) {
			t.Errorf("%v tokens: HBM %.6f ms, want the layer without its mHC ops %.6f ms",
				tokens, got[idxHBM]*1e3, bare[idxHBM]*1e3)
		}
	}
}

// The head's final mHC post is covered the same way: priced at the post form, its stated
// bytes no longer charged.
func TestTheHeadsFinalMHCPostIsChargedOnce(t *testing.T) {
	with := withEntries(t, contractEntries("mhc"))
	without := withEntries(t, nil)
	b := decodeBatch(4, 6, 90000)
	head := without.plan.Head
	if len(head.MHC) != 1 || head.MHC[0].Pre || !head.MHC[0].Post {
		t.Fatalf("head mHC nodes %+v; the golden assumes one post", head.MHC)
	}
	priced := func(k *Kernel) [numResources]float64 {
		s := k.shapeOf(b)
		var per [numResources]float64
		k.priceHead(k.plan.Head, &s, &per)
		return per
	}
	a, z := priced(with), priced(without)
	tokens := float64(b.Tokens())
	if got, want := a[idxSM]-z[idxSM], (3.0+tokens*40960/1850611)*1e-6; !near(got, want) {
		t.Errorf("head mHC post adds %v s of SM, want %v", got, want)
	}
	if got, want := z[idxHBM]-a[idxHBM],
		tokens*head.MHC[0].Bytes/without.hbmBytesPerSecond; !near(got, want) {
		t.Errorf("head HBM falls by %v s, want the post node's stated bytes, %v", got, want)
	}
}

func withCount(l price.PlannedLayer, n int) price.PlannedLayer { l.Count = n; return l }

// A FAMILY IS ALL OR NONE, AND ITS ABSENCE PRICES AS BEFORE. Each family with one member
// dropped prices every step exactly as with no entries at all, and Provenance records the
// fallback; with every entry absent, the rate's absence is recorded as optional records it.
func TestAPartialFamilyFallsBackAndSaysSo(t *testing.T) {
	none := withEntries(t, nil)
	steps := []kernel.Batch{decodeBatch(4, 6, 90000), chunkOn(8192, 81808),
		chunkOn(1808, 88192)}
	for _, c := range []struct{ family, drop, record string }{
		{"sparse_mla", "attention_prefill_per_request_sparse_mla", "attention_prefill_sparse_mla"},
		{"block_index_scores", "attention_prefill_floor_block_index_scores",
			"attention_prefill_block_index_scores"},
		{"mhc", "mhc_floor_pre", "mhc_pre"},
		{"mhc", "mhc_rate_post", "mhc"},
	} {
		k := withEntries(t, contractEntries(c.family), c.drop)
		for _, b := range steps {
			if a, z := k.StepTime(b), none.StepTime(b); a.NoOverlap != z.NoOverlap ||
				a.Overlap != z.Overlap {
				t.Errorf("%s without %s: a %d-token step prices %v, want the absent "+
					"family's %v", c.family, c.drop, b.Tokens(), a.NoOverlap, z.NoOverlap)
			}
		}
		if !hasFallback(k, c.record) {
			t.Errorf("%s without %s: no fallback recorded under %s", c.family, c.drop,
				c.record)
		}
	}
	for _, name := range []string{"attention_prefill_rate_sparse_mla",
		"attention_prefill_rate_block_index_scores", "mhc_rate_pre", "mhc_rate_post"} {
		if !hasAssumption(none, name) {
			t.Errorf("%s is absent and its absence is not recorded", name)
		}
	}
}

func hasFallback(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name && strings.HasPrefix(o.Scope, "fallback") {
			return true
		}
	}
	return false
}

// THE CONTRACT'S HOST ENTRIES RESOLVE. A per-request admission of zero is accepted where it
// is not_charged, and the model's per-token cost replaces the part-wide one: admission is
// 2.67 us a prompt token and nothing a request.
func TestTheContractsAdmissionEntriesResolve(t *testing.T) {
	k := withEntries(t, contractEntries("host"))
	if got, want := k.AdmissionOverhead(90000),
		time.Duration(2.67*90000*float64(time.Microsecond)); got != want {
		t.Errorf("admission of a 90,000-token prompt %v, want %v", got, want)
	}
}

// The block sums are exact: against a per-position loop over random ranges, ratios, caps and
// blocks.
func TestTheBlockSumsMatchAPerPositionLoop(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		lo := 1 + r.Intn(5000)
		hi := lo + r.Intn(3000) - 50
		ratio := 1 + r.Intn(130)
		cap := r.Intn(700)
		block := []int{1, 64}[r.Intn(2)]
		var want, wantCeil float64
		for p := lo; p <= hi; p++ {
			m := p / ratio
			if cap > 0 {
				m = min(m, cap)
			}
			want += float64((m + block - 1) / block)
			wantCeil += float64((p + ratio - 1) / ratio)
		}
		if got := sumBlocks(lo, hi, ratio, cap, block); got != want {
			t.Fatalf("sumBlocks(%d, %d, %d, %d, %d) = %v, want %v", lo, hi, ratio, cap,
				block, got, want)
		}
		if got := sumCeil(lo, hi, ratio); hi >= lo && got != wantCeil {
			t.Fatalf("sumCeil(%d, %d, %d) = %v, want %v", lo, hi, ratio, got, wantCeil)
		}
	}
}
