package latencykernel

import (
	"math"
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// Decode-context parallelism shards the KV cache by token, and these pin that
// behaviourally.
//
// THE DEFECT. Layout.DCP was assigned once and never read: over the whole repository
// `grep -rnE '\.DCP' --include='*.go'` returned the one assignment in the resolver. So two
// deployments differing only in dcp priced byte-identically -- on h200, 4 nodes, tp8/dp4
// with expert parallelism, dcp 1 and dcp 8 both gave 1104.4547515127522 tokens/sec,
// 186.05310625 ms TTFT and 6.880028693656094 ms ITL.
//
// WHY IT MATTERS MOST FOR LATENT ATTENTION. price.KVBytesPerToken divides KV heads by tp
// and floors at one head, so a model the catalog declares `n_kv: 1, d_h: 576` gets ZERO KV
// reduction from tensor parallelism: per-rank heads are 1 at tp=1 and still 1 at tp=8. DCP
// is the only axis that can shard such a cache, because it shards by token rather than by
// head. Nine catalog models declare an mla or sparse_mla layer.
//
// WHAT THE ENGINE DOES, which these tests encode (vllm at v0.31.0):
//
//	tokens divide, bytes per token do not   kv_cache_interface.py:545-550, :578-583
//	a sliding window is refused outright    kv_cache_interface.py:868-872
//	recurrent state is never sharded        kv_cache_interface.py:1097-1098
//	the per-rank length, with striping      attention/backends/utils.py:1143-1156
//	the ag_rs combine, three collectives    attention/ops/dcp.py:458, :493, :1593
//
// The fixtures are chosen for their layer kinds, which a probe confirmed: glm5-h200-tp8 is
// 78 sparse_mla layers, deepseek-v3 is 61 mla, and gpt-oss-120b is the hybrid -- 18 gqa
// against 18 swa -- which is the case a deployment-wide divisor would get wrong.
const (
	dcpMLAFixture    = "aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml"
	dcpSparseFixture = "glm5-h200-tp8.yaml"
	dcpHybridFixture = "aisimulate/gpt-oss-120b-h200-fp4-vllm-tp8.yaml"
)

// dcpKernel builds a kernel from a committed fixture with one parallelism field changed.
// Mutating the documents before New is what lets a test vary a layout the fixtures do not
// carry: no committed scenario sets dcp, which is also why every existing test stayed
// green while the field was write-only.
func dcpKernel(t testing.TB, fixture string, dcp int) *Kernel {
	t.Helper()
	in := fixtureInputs(t, fixture)
	in.Deployment.Pools[0].Parallel.DCP = dcp
	k, err := New(in)
	if err != nil {
		t.Fatalf("%s at dcp=%d: %v", fixture, dcp, err)
	}
	return k
}

// ---------------------------------------------------------------------------
// The arithmetic, pinned exactly against the engine's own per-rank form.
// ---------------------------------------------------------------------------

// vllmLocalSeqLen is get_dcp_local_seq_lens transcribed for one rank
// (vllm/v1/attention/backends/utils.py:1143-1156 at v0.31.0). It is the ORACLE: the
// kernel's own form must equal the maximum of this across ranks, because the decode
// combine is a collective and every rank waits for the one holding the most tokens.
func vllmLocalSeqLen(seqLen, dcp, interleave, rank int) int {
	base := seqLen / interleave / dcp * interleave
	rem := seqLen - base*dcp - rank*interleave
	if rem < 0 {
		rem = 0
	}
	if rem > interleave {
		rem = interleave
	}
	return base + rem
}

// The headline arithmetic property. Charging the mean would price a step cheaper than any
// rank can deliver, and dropping the interleave remainder would price it cheaper still.
//
// Mutation-checked: replacing the body with ctx/dcp (floor), with ceil over the batch
// total, or with base alone each fails here.
func TestDCPLocalTokensMatchesTheSlowestRankNotTheMean(t *testing.T) {
	for _, dcp := range []int{2, 4, 8} {
		for _, interleave := range []int{1, 16, 32} {
			for seqLen := 0; seqLen < 600; seqLen++ {
				want, sum := 0, 0
				for rank := 0; rank < dcp; rank++ {
					v := vllmLocalSeqLen(seqLen, dcp, interleave, rank)
					if v > want {
						want = v
					}
					sum += v
				}
				// The engine's own form must conserve the context. If this ever fails the
				// oracle is wrong, not the kernel.
				if sum != seqLen {
					t.Fatalf("oracle does not conserve: dcp=%d interleave=%d seqLen=%d "+
						"shards sum to %d", dcp, interleave, seqLen, sum)
				}
				if seqLen > 0 && want < 1 {
					want = 1
				}
				got := dcpLocalTokens([]int{seqLen}, dcp, interleave)
				if got != float64(want) {
					t.Fatalf("dcp=%d interleave=%d seqLen=%d: got %v, want %d "+
						"(the slowest rank's share)", dcp, interleave, seqLen, got, want)
				}
			}
		}
	}
}

// At the engine's default interleave of 1 -- what an unstated size runs at unless NIXL pins
// it -- the form must reduce to ceil(context/dcp). Stated separately because it is the
// common case, and because a reader checking this file against vLLM will look for it.
func TestDCPLocalTokensReducesToCeilingAtTheDefaultInterleave(t *testing.T) {
	for _, seqLen := range []int{1, 7, 100, 999, 8192, 131072} {
		for _, dcp := range []int{2, 3, 8} {
			want := float64((seqLen + dcp - 1) / dcp)
			if got := dcpLocalTokens([]int{seqLen}, dcp, 1); got != want {
				t.Errorf("seqLen=%d dcp=%d: got %v, want %v", seqLen, dcp, got, want)
			}
		}
	}
	// The documented worked example: striping contributes a bounded additive term, not a
	// multiplicative one. 1000 tokens over 8 ranks in runs of 32 gives the slowest rank
	// 128 against a nominal 125 -- under 3%. Bounded is not the same as small, though: the
	// term is at most one run per request, so on a context short against the run it
	// dominates, and 100 tokens over 8 ranks in runs of 64 put 64 on the slowest rank
	// against a nominal 13.
	if got := dcpLocalTokens([]int{1000}, 8, 32); got != 128 {
		t.Errorf("interleave 32 at 1000 tokens over 8 ranks: got %v, want 128", got)
	}
	if got := dcpLocalTokens([]int{100}, 8, 64); got != 64 {
		t.Errorf("interleave 64 at 100 tokens over 8 ranks: got %v, want 64", got)
	}
}

// The shards must sum to the whole context: tokens divide freely, with NO replication
// floor. This is where DCP differs from the KV-head division next door, which floors at
// one head because a head is never split -- and copying that floor here would be the
// natural mistake, since the two functions sit feet apart.
func TestDCPLocalTokensShardsSumToTheWholeContextWithNoReplication(t *testing.T) {
	for _, dcp := range []int{2, 4, 8} {
		for _, seqLen := range []int{16, 17, 100, 1024, 32768} {
			var sum int
			for rank := 0; rank < dcp; rank++ {
				sum += vllmLocalSeqLen(seqLen, dcp, 1, rank)
			}
			if sum != seqLen {
				t.Errorf("dcp=%d seqLen=%d: shards sum to %d", dcp, seqLen, sum)
			}
			// A floored form would exceed the context once summed. The kernel's figure is
			// the maximum rather than the sum, so the check is that it never exceeds the
			// context and never falls below the fair share.
			got := dcpLocalTokens([]int{seqLen}, dcp, 1)
			if got > float64(seqLen) {
				t.Errorf("dcp=%d seqLen=%d: a shard of %v exceeds the whole context",
					dcp, seqLen, got)
			}
			if got < float64(seqLen)/float64(dcp) {
				t.Errorf("dcp=%d seqLen=%d: a shard of %v is below the fair share %v",
					dcp, seqLen, got, float64(seqLen)/float64(dcp))
			}
		}
	}
}

// New refuses exactly the context-parallel layouts the engine refuses, and admits their
// neighbours.
//
// vLLM v0.31.0 checks two things at startup (vllm/config/parallel.py:563-578):
//
//	pcp == 1:  tp % dcp == 0                  "DCP reuses the TP ranks"
//	pcp  > 1:  dcp in {1, pcp, tp*pcp}         "disabled, span the PCP axis, or span the
//	                                            full TP x PCP axis"
//
// and its world size is pp * tp * pcp (:899-903), so PCP adds ranks where DCP adds none.
// blis-schemas v0.2.2 enforces all three, and New runs its field validation, so a kernel is
// never built for a layout that does not start. Asserted here because it is the contract a
// caller sweeping layouts relies on: a refused layout is an error, never a price.
//
// glm5, a DSA sparse-MLA stack, because PCP with DCP runs only on DSA layers; the plain-MLA
// refusal is the last row.
func TestNewRefusesTheContextParallelLayoutsTheEngineRefuses(t *testing.T) {
	build := func(fixture string, tp, pcp, dcp, nodes int) error {
		t.Helper()
		in := fixtureInputs(t, fixture)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.TP, pool.Parallel.PCP, pool.Parallel.DCP = tp, pcp, dcp
		pool.Nodes, in.Scenario.Cluster.Nodes = nodes, nodes
		if nodes > 1 {
			in.Scenario.Cluster.Fabric = "ib-400g"
			f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", "ib-400g.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			in.Fabric = f
		}
		_, err := New(in)
		return err
	}
	for _, c := range []struct {
		name                string
		fixture             string
		tp, pcp, dcp, nodes int
		admitted            bool
	}{
		{"dcp divides tp", dcpSparseFixture, 8, 1, 4, 1, true},
		{"dcp does not divide tp", dcpSparseFixture, 8, 1, 3, 1, false},
		{"dcp wider than tp", dcpSparseFixture, 4, 1, 8, 1, false},
		{"dcp spans the pcp axis", dcpSparseFixture, 8, 2, 2, 2, true},
		{"dcp spans tp x pcp", dcpSparseFixture, 4, 2, 8, 1, true},
		{"dcp is neither", dcpSparseFixture, 8, 2, 4, 2, false},
		{"pcp fits the GPUs", dcpSparseFixture, 4, 2, 1, 1, true},
		{"pcp needs more GPUs than the pool has", dcpSparseFixture, 8, 2, 1, 1, false},
		// MLAAttention.supports_pcp_dcp is False (mla_attention.py:440-442, raised at
		// :684-687); only DeepseekV32Attention opts in.
		{"plain mla runs pcp alone", dcpMLAFixture, 4, 2, 1, 1, true},
		{"plain mla refuses pcp with dcp", dcpMLAFixture, 4, 2, 8, 1, false},
	} {
		err := build(c.fixture, c.tp, c.pcp, c.dcp, c.nodes)
		if c.admitted && err != nil {
			t.Errorf("%s (tp=%d pcp=%d dcp=%d on %d node(s)): refused, but the engine "+
				"starts it: %v", c.name, c.tp, c.pcp, c.dcp, c.nodes, err)
		}
		if !c.admitted && err == nil {
			t.Errorf("%s (tp=%d pcp=%d dcp=%d on %d node(s)): priced, but the engine "+
				"refuses it at startup", c.name, c.tp, c.pcp, c.dcp, c.nodes)
		}
	}
}

// A context shorter than the group still occupies a token somewhere. Zero here would make
// a decode read free, which is the silent-and-cheap direction this repository refuses.
func TestDCPLocalTokensFloorsAtOneTokenPerRank(t *testing.T) {
	for _, dcp := range []int{2, 8, 64} {
		if got := dcpLocalTokens([]int{1}, dcp, 1); got != 1 {
			t.Errorf("one token over %d ranks: got %v, want 1", dcp, got)
		}
	}
	// Defensive arithmetic, since a library cannot assume its caller validated: a
	// non-positive width is absent, not a division by zero. blis-schemas owns the
	// admissibility rules (tp % dcp == 0, and dcp in {1, pcp, tp*pcp} when pcp is on);
	// this only refuses to divide by zero, following the max(TP, 1) pattern elsewhere.
	for _, dcp := range []int{0, -4} {
		if got := dcpLocalTokens([]int{4096}, dcp, 1); got != -1 {
			t.Errorf("dcp=%d should read as absent: got %v, want -1", dcp, got)
		}
	}
	if got := dcpLocalTokens([]int{4096}, 4, 0); got != 1024 {
		t.Errorf("a zero interleave should fall back to the engine default of 1: "+
			"got %v, want 1024", got)
	}
}

// Per request, not over the batch total. The remainder term is per sequence, so
// sum-then-shard and shard-then-sum disagree whenever a context is not a multiple of
// interleave*dcp -- which in a real batch is the normal case.
func TestDCPLocalTokensBoundsEachRequestNotTheBatchTotal(t *testing.T) {
	// Four requests of 1000 at dcp=3: each pays its own ceiling, 4*334 = 1336. Sharding
	// the 4000-token total once would give 1334.
	perRequest := dcpLocalTokens([]int{1000, 1000, 1000, 1000}, 3, 1)
	batchTotal := dcpLocalTokens([]int{4000}, 3, 1)
	if perRequest <= batchTotal {
		t.Errorf("per-request sharding gave %v and batch-total sharding %v; the "+
			"per-request figure must be the larger, since each sequence pays its own "+
			"remainder", perRequest, batchTotal)
	}
	if perRequest != 1336 || batchTotal != 1334 {
		t.Errorf("got per-request %v and batch-total %v; want 1336 and 1334",
			perRequest, batchTotal)
	}
}

// Which kinds shard, and which the engine refuses to shard at all.
func TestDCPShardsOnlyTheKindsTheEngineShards(t *testing.T) {
	for _, c := range []struct {
		kind model.AttentionKind
		want bool
		why  string
	}{
		{model.AttentionGQA, true, "full attention: AttentionSpec.dcp_sharded is True"},
		{model.AttentionMLA, true, "latent: the only axis that can shard a one-head cache"},
		{model.AttentionSparseMLA, true, "latent and selective: still sharded by token"},
		{model.AttentionSWA, false, "refused: \"DCP not support sliding window\""},
		{model.AttentionKind("something-new"), false, "unrecognized over-prices rather than under-prices"},
	} {
		if got := dcpShardsKV(c.kind); got != c.want {
			t.Errorf("%s: got %v, want %v (%s)", c.kind, got, c.want, c.why)
		}
	}
}

// ---------------------------------------------------------------------------
// Step time: the regression the issue opens with.
// ---------------------------------------------------------------------------

// THE ACCEPTANCE CRITERION. Two deployments differing only in dcp must no longer price
// identically, and widening the shard must keep lowering what a rank READS.
//
// The read, not the whole step. DCP trades decode time for capacity: each width shards
// the read further and adds a combine whose payload and floors grow with the group, so the
// step need not fall monotonically, and at 32 requests on a 32k context it does not -- on
// deepseek-v3 the step rises from dcp=4 to dcp=8 once the log-sum-exp all-gather, the
// third collective vLLM launches per layer (vllm/v1/attention/ops/dcp.py:458 at v0.31.0),
// is priced. An earlier form of this test required the step to fall at every width, which
// held only because that collective was left out.
//
// Four widths rather than two, deliberately: a mutation replacing the divisor with any
// constant passes a two-point check. The pooled-scan commit found the same thing.
func TestDCPLowersTheDecodeReadAndWideningItLowersFurther(t *testing.T) {
	for _, fixture := range []string{dcpMLAFixture, dcpSparseFixture} {
		t.Run(fixture, func(t *testing.T) {
			batch := decodeBatch(32, 1, 32768)
			var prevRead, prevStep float64
			for i, dcp := range []int{1, 2, 4, 8} {
				est := dcpKernel(t, fixture, dcp).StepTime(batch)
				read := est.PerResource[kernel.ResourceHBM].Seconds()
				step := est.NoOverlap.Seconds()
				if i > 0 && read >= prevRead {
					t.Errorf("dcp=%d read %.6f ms of HBM, not below the %.6f ms of the "+
						"previous width: a wider shard must read less",
						dcp, read*1e3, prevRead*1e3)
				}
				if i > 0 && step == prevStep {
					t.Errorf("dcp=%d priced the step identically to the previous width",
						dcp)
				}
				prevRead, prevStep = read, step
			}
		})
	}
}

// The decode READ must divide by the shard exactly, which a difference of differences
// isolates: with read(d) = R/d plus terms that do not depend on d,
//
//	(t(1) - t(2)) / (t(2) - t(4)) = (R - R/2) / (R/2 - R/4) = 2
//
// Every unsharded term -- the floor, the GEMMs, the MoE, the host -- cancels. So this
// kills a divisor that merely decreases in dcp: an off-by-one, a square root, dcp/2.
//
// Charged on the HBM resource because that is what a decode read binds on.
func TestDCPDividesTheDecodeReadByExactlyTheShard(t *testing.T) {
	for _, fixture := range []string{dcpMLAFixture, dcpSparseFixture} {
		t.Run(fixture, func(t *testing.T) {
			batch := decodeBatch(32, 1, 32768)
			read := func(dcp int) float64 {
				return dcpKernel(t, fixture, dcp).StepTime(batch).
					PerResource[kernel.ResourceHBM].Seconds()
			}
			h1, h2, h4 := read(1), read(2), read(4)
			if h2 >= h1 || h4 >= h2 {
				t.Fatalf("the HBM term is not falling with the shard: %.6f, %.6f, %.6f ms",
					h1*1e3, h2*1e3, h4*1e3)
			}
			ratio := (h1 - h2) / (h2 - h4)
			if math.Abs(ratio-2) > 1e-6 {
				t.Errorf("(t1-t2)/(t2-t4) is %.9f, want 2: the read is not dividing by "+
					"exactly the shard width", ratio)
			}
		})
	}
}

// The shard is applied PER REQUEST at the call site, not as one ratio over the batch.
//
// The engine keeps a per-request vector of local lengths and sums it in the kernel
// (dcp_local_seq_lens, vllm/v1/attention/backend.py:427-428, built per request at
// flash_attn.py:884-893 of v0.31.0). Each sequence's shard rounds on its own, so a batch
// of many short contexts reads more than one long context of the same total -- every
// request pays its own ceiling.
//
// Measured here: eight requests of 1,001 tokens at dcp=4 read 8*ceil(1001/4) = 2,008
// positions, where one request of 8,008 reads ceil(8008/4) = 2,002. Same total context,
// six positions apart, and a batch-wide ratio would report 2,002 for both.
//
// AT THE CALL SITE RATHER THAN ON THE PURE FUNCTION, which is the point: an earlier draft
// of this change applied the shard as a batch-wide fraction multiplied into the already
// bounded token count. That is exact for a non-sparse layer and not in general -- searched
// over 200,000 random (width, top-k, compression, context) draws the two disagree by up to
// 4.3% on a heavily compressed sparse layer -- and no test on the pure function could see
// it, because the pure function was right either way. This asserts the composition.
func TestDCPShardsEachRequestSeparatelyAtTheCallSite(t *testing.T) {
	// A full-attention cache, so the only thing narrowing the read is the shard. A sparse
	// layer selects a different number of positions in the two batches below -- that is
	// its own bound doing its job -- which would mask the rounding this isolates. The
	// sparse composition is asserted in the subtest that follows.
	read := func(k *Kernel, b kernel.Batch) float64 {
		return k.StepTime(b).PerResource[kernel.ResourceHBM].Seconds()
	}
	sharded := dcpKernel(t, dcpMLAFixture, 4)
	whole := dcpKernel(t, dcpMLAFixture, 1)

	many := decodeBatch(8, 1, 1001)
	one := decodeBatch(1, 1, 8008)

	// Sharding lowers both reads. The SAVING is what differs: one long request rounds
	// once, eight short ones round eight times and so keep more positions.
	savedMany := read(whole, many) - read(sharded, many)
	savedOne := read(whole, one) - read(sharded, one)
	if savedMany <= 0 || savedOne <= 0 {
		t.Fatalf("sharding should lower both reads: savings of %.9f and %.9f s",
			savedMany, savedOne)
	}
	if savedOne <= savedMany {
		t.Errorf("one 8,008-token request saved %.9f s and eight 1,001-token requests "+
			"saved %.9f s; eight separate ceilings retain more positions than one, so "+
			"the single request must save more", savedOne, savedMany)
	}

	// The sparse composition, stated directly on the quantity the pricer uses: selection
	// first, then the shard, each per request. A batch-wide ratio over the selected total
	// would report the same number for both batches.
	var sparse *price.PlannedLayer
	ks := dcpKernel(t, dcpSparseFixture, 4)
	for i := range ks.plan.Layers {
		if sparseTopK(&ks.plan.Layers[i]) > 0 {
			sparse = &ks.plan.Layers[i]
			break
		}
	}
	if sparse == nil {
		t.Fatal("the sparse fixture carries no selecting layer")
	}
	eight := make([]int, 8)
	for i := range eight {
		eight[i] = 1001
	}
	perRequest := ks.dcpDecodeTokens(sparse, sparse.AttnKind, eight)
	batchWide := ks.dcpDecodeTokens(sparse, sparse.AttnKind, []int{8008})
	if perRequest <= batchWide {
		t.Errorf("eight selecting requests of 1,001 read %v positions and one of 8,008 "+
			"reads %v; each request selects and then shards on its own", perRequest, batchWide)
	}

	// THE CASE THAT SEPARATES THE TWO COMPOSITIONS, and the reason this is not a matter of
	// taste. Where the selection SATURATES -- a context past the top-k, so the layer reads
	// exactly top-k positions whatever the context -- the shard of that selection is
	// exact: ceil(2048/4) = 512. A ratio taken from the RAW contexts instead carries their
	// remainder into a quantity the remainder has nothing to do with: ceil(3298/4)/3298 is
	// 0.25015, and 2048 times that is 512.3105.
	//
	// The error is small and it is in the wrong direction for the wrong reason, which is
	// the kind that survives review. Asserted on an exact integer so there is nothing to
	// tune: a saturated selection sharded four ways is a whole number of positions.
	topk := sparseTopK(sparse)
	if sparse.AttnCompressRatio > 0 {
		t.Skipf("this fixture's sparse layer compresses (ratio %d), so its selection does "+
			"not saturate at the top-k and the exact figure below does not apply",
			sparse.AttnCompressRatio)
	}
	saturated := ks.dcpDecodeTokens(sparse, sparse.AttnKind, []int{topk + 1250})
	if want := float64((topk + 3) / 4); saturated != want {
		t.Errorf("a context past the top-k read %v positions at dcp=4, want exactly %v "+
			"(the top-k of %d sharded four ways); a shard taken from the raw context "+
			"rather than from the selection leaks the context's remainder",
			saturated, want, topk)
	}
}

// With the interleave unstated and no NIXL connector, the pricer must shard at the engine's
// DEFAULT of one token.
//
// cp_kv_cache_interleave_size defaults to 1 (vllm/config/parallel.py:393 at v0.31.0):
// "Interleave_size=1: token-level alignment, where token `i` is stored on dcp_rank
// `i % dcp_world_size`." At that value the slowest rank holds exactly ceil(context/dcp).
//
// Pinned as an exact integer per request, because a larger run shifts the figure by up to
// one run per request and a step time would not show it: at dcp=4 a 1,001-token context
// reads 251 positions at the default and 256 at a block-aligned 16, and both look equally
// plausible. The stated and NIXL-pinned cases are in dcp_knobs_test.go.
func TestDCPShardsAtTheEngineDefaultTokenInterleave(t *testing.T) {
	k := dcpKernel(t, dcpMLAFixture, 4)
	var full *price.PlannedLayer
	for i := range k.plan.Layers {
		if k.plan.Layers[i].AttnQHeads > 0 && sparseTopK(&k.plan.Layers[i]) == 0 {
			full = &k.plan.Layers[i]
			break
		}
	}
	if full == nil {
		t.Fatal("this fixture carries no non-selecting attention layer")
	}
	for _, c := range []struct {
		contexts []int
		want     float64
	}{
		{[]int{1001}, 251},       // ceil(1001/4); at interleave 16 it would be 256
		{[]int{8008}, 2002},      // ceil(8008/4)
		{[]int{1001, 1001}, 502}, // each request rounds on its own
		{[]int{4096}, 1024},      // an exact multiple leaves no remainder
		{[]int{1, 1, 1, 1}, 4},   // one token each, one position each
		{[]int{32768, 1001}, 8192 + 251},
	} {
		if got := k.dcpDecodeTokens(full, full.AttnKind, c.contexts); got != c.want {
			t.Errorf("contexts %v at dcp=4: read %v positions, want %v",
				c.contexts, got, c.want)
		}
	}
}

// A hybrid stack must shard only the half the engine shards. gpt-oss-120b alternates 18
// gqa layers with 18 sliding-window ones, and a sliding window is a configuration vLLM
// refuses to start under DCP at all -- so a deployment-wide divisor would shard layers
// whose cache the engine never shards.
//
// Asserted as a comparison rather than a value: the hybrid's decode read must fall by
// strictly less than an all-sharding model's does. Measured here: about 15% against about
// 55%.
func TestDCPDoesNotShardASlidingWindowLayer(t *testing.T) {
	fall := func(fixture string) float64 {
		batch := decodeBatch(32, 1, 32768)
		read := func(dcp int) float64 {
			return dcpKernel(t, fixture, dcp).StepTime(batch).
				PerResource[kernel.ResourceHBM].Seconds()
		}
		at1 := read(1)
		return (at1 - read(8)) / at1
	}
	hybrid, allSharding := fall(dcpHybridFixture), fall(dcpMLAFixture)
	if hybrid <= 0 {
		t.Errorf("the hybrid's gqa half should still shard, but its read did not fall")
	}
	if hybrid >= allSharding {
		t.Errorf("the hybrid read fell %.1f%% and the all-sharding model %.1f%%; a stack "+
			"half of whose layers are sliding-window must shard strictly less",
			hybrid*100, allSharding*100)
	}
}

// Capacity: the KV a rank holds must fall with the shard, and the paging must happen AFTER
// the division rather than before it -- which is the engine's order
// (max_memory_usage_bytes divides max_model_len by dcp_world_size, then rounds to blocks;
// page_size_bytes never sees dcp).
//
// Dividing a per-token byte figure instead would make a page fractional and understate
// occupancy by up to dcp below block_size*dcp tokens. This asserts the whole-page
// behaviour that distinguishes the two.
func TestDCPShardsCapacityByTokensBeforePaging(t *testing.T) {
	k1 := dcpKernel(t, dcpMLAFixture, 1)
	k8 := dcpKernel(t, dcpMLAFixture, 8)

	// Well above block_size*dcp, where the two orderings agree and the division is exact.
	long1, long8 := k1.SequenceVariableBytes(32768), k8.SequenceVariableBytes(32768)
	if long8*8 != long1 {
		t.Errorf("at 32768 tokens: dcp=1 holds %d and dcp=8 holds %d; the shard should be "+
			"exact at a context this long", long1, long8)
	}

	// Below one page per rank. Sharding the TOKEN count and then paging gives one whole
	// page; dividing bytes per token would give an eighth of one.
	perRankPage := k8.SequenceVariableBytes(1)
	if perRankPage != k1.SequenceVariableBytes(1) {
		t.Errorf("a single token occupies %d bytes at dcp=8 against %d at dcp=1; one "+
			"token is one page on one rank either way, because a page is not divisible",
			perRankPage, k1.SequenceVariableBytes(1))
	}

	// The hybrid is left unsharded, which is the stated coverage limit rather than an
	// oversight: kvBytesPerToken is one whole-model scalar, so a stack mixing sharding
	// and non-sharding kinds cannot be expressed exactly here. Over-stating is the safe
	// direction, and the engine refuses the configuration anyway.
	h1 := dcpKernel(t, dcpHybridFixture, 1).SequenceVariableBytes(32768)
	h8 := dcpKernel(t, dcpHybridFixture, 8).SequenceVariableBytes(32768)
	if h1 != h8 {
		t.Errorf("the hybrid's capacity moved from %d to %d; a stack holding a "+
			"sliding-window cache is deliberately left unsharded here", h1, h8)
	}
}

// The capacity shard must agree with the engine's LOGICAL BLOCK, which DCP widens.
//
// vLLM scales the scheduler's block to its token span under DCP:
// resolve_dcp_kv_block_size returns `spec.block_size * dcp_world_size` for a sharded spec
// (vllm/v1/core/kv_cache_utils.py:717-719 at v0.31.0). So a sequence is allocated in units
// of block_size*dcp tokens, of which each rank stores block_size -- which is why a real
// deployment keeps the product in sync with its router's block size, the GLM-5.3-Flash
// canary running KV_BLOCK_SIZE 64 at DCP_SIZE 8 for a 512-token logical block.
//
// Sharding the token count and THEN paging at block_size is the same arithmetic -- the
// integer identity cdiv(cdiv(L, dcp), bs) == cdiv(L, bs*dcp) -- and this pins that the two
// readings agree, so a later change cannot drift from the allocator's granularity. The
// lengths below straddle a logical block boundary at 512, where a form that paged before
// sharding would disagree.
func TestDCPCapacityAgreesWithTheEnginesLogicalBlock(t *testing.T) {
	in := fixtureInputs(t, dcpMLAFixture)
	in.Deployment.Pools[0].Parallel.DCP = 8
	in.Deployment.Pools[0].Engine.BlockSize = 64
	k, err := New(in)
	if err != nil {
		t.Fatalf("dcp=8 at block size 64: %v", err)
	}
	const blockSize, dcp = 64, 8
	for _, tokens := range []int{1, 64, 100, 511, 512, 513, 1000, 4096, 32768} {
		// The engine's reading: whole logical blocks of blockSize*dcp tokens, each
		// leaving this rank blockSize tokens to store.
		logical := (tokens + blockSize*dcp - 1) / (blockSize * dcp)
		want := price.PagedBytes(logical*blockSize, blockSize, k.kvBytesPerToken)
		if got := k.SequenceVariableBytes(tokens); got != want {
			t.Errorf("%d tokens at dcp=8, block 64: holds %d bytes, want %d (%d logical "+
				"block(s) of %d tokens, %d of them on this rank)",
				tokens, got, want, logical, blockSize*dcp, blockSize)
		}
	}
}

// DCP must not shard a prefill-to-decode transfer. A prefill pool holds the WHOLE cache
// for a request -- DCP shards the decode cache -- so the bytes crossing between pools are
// the whole request's whatever the decode pool's width is.
//
// THIS IS THE TEST THAT RULES OUT THE OBVIOUS IMPLEMENTATION. Dividing kvBytesPerToken
// once at construction would reach all three decode read sites in one edit, and it would
// also divide this, making every PD transfer exactly dcp times too cheap at every token
// count. That is why the shard is applied to token counts instead.
func TestDCPDoesNotShardAPrefillToDecodeTransfer(t *testing.T) {
	from := kernel.Placement{Node: 0, Rack: 0}
	to := kernel.Placement{Node: 1, Rack: 0}
	want := dcpKernel(t, dcpMLAFixture, 1).PDTransferTime(4096, from, to)
	for _, dcp := range []int{2, 4, 8} {
		if got := dcpKernel(t, dcpMLAFixture, dcp).PDTransferTime(4096, from, to); got != want {
			t.Errorf("dcp=%d priced a PD transfer at %v against %v at dcp=1; a prefill "+
				"pool sends the whole request's KV", dcp, got, want)
		}
	}
}

// DCP must not touch prefill. It shards the DECODE cache; a prefill chunk computes every
// position it was scheduled. Prefill-context parallelism is the axis that splits a prefill,
// and conflating the two is the easiest thing to get backwards.
func TestDCPDoesNotShardPrefill(t *testing.T) {
	prefill := decodeBatch(1, 4096, 4096)
	want := dcpKernel(t, dcpMLAFixture, 1).StepTime(prefill).NoOverlap
	for _, dcp := range []int{2, 8} {
		if got := dcpKernel(t, dcpMLAFixture, dcp).StepTime(prefill).NoOverlap; got != want {
			t.Errorf("dcp=%d priced a prefill-only step at %v against %v at dcp=1",
				dcp, got, want)
		}
	}
}

// The per-call attention floor must NOT divide with the shard, and the bytes MUST.
//
// An MLA decode's floor is 51.5-89.5us against full attention's 9.5-19.5us precisely
// because its per-call setup reads a latent cache. That is a launch-and-descriptor cost,
// paid once per rank per invocation however many tokens the rank holds; sharding a
// sequence does not make a kernel launch cheaper. Dividing it by 8 would put an MLA floor
// at 6.4-11.2us -- BELOW full attention's -- for the kernel the registry measures as
// several times costlier to set up. This is the modelling judgement the issue asked to be
// made on the evidence rather than inherited from whichever behaviour falls out of the
// byte arithmetic.
//
// SEPARATING THE TWO TERMS. The decode read is affine in context: slope*context +
// intercept, where the slope is bytes-per-token over the rate and the intercept is the
// per-layer floors. Two contexts at one width recover both. The shard must then divide the
// SLOPE exactly and leave the INTERCEPT untouched.
//
// Asserting it this way rather than as a lower bound on the step is what gives the test
// teeth: a mutation scaling the floor by the shard survives any assertion that the step
// exceeds its floor total, because on this fixture the GEMM and MoE terms are five times
// the floors and swamp them. It also has to be measured at a context long enough to
// actually shard -- at a one-token context the shard is 1 by construction, so the
// mutation is inert there and the test would pass for the wrong reason.
func TestDCPDividesTheReadRateButNotThePerCallFloor(t *testing.T) {
	// The affine fit, from two contexts well above block_size*dcp.
	fit := func(dcp int) (slope, intercept float64) {
		k := dcpKernel(t, dcpMLAFixture, dcp)
		read := func(ctx int) float64 {
			return k.StepTime(decodeBatch(1, 1, ctx)).
				PerResource[kernel.ResourceHBM].Seconds()
		}
		lo, hi := 8192, 16384
		atLo, atHi := read(lo), read(hi)
		slope = (atHi - atLo) / float64(hi-lo)
		return slope, atLo - slope*float64(lo)
	}

	slope1, intercept1 := fit(1)
	if slope1 <= 0 || intercept1 <= 0 {
		t.Fatalf("the decode read is not affine and positive in context: slope %.6e, "+
			"intercept %.6f ms", slope1, intercept1*1e3)
	}
	for _, dcp := range []int{2, 8} {
		slope, intercept := fit(dcp)
		// The rate term divides by exactly the shard.
		if want := slope1 / float64(dcp); math.Abs(slope-want)/want > 1e-9 {
			t.Errorf("dcp=%d: the per-token read rate is %.6e, want %.6e (the dcp=1 "+
				"figure over %d)", dcp, slope, want, dcp)
		}
		// The floor does not move at all.
		if math.Abs(intercept-intercept1) > 1e-12 {
			t.Errorf("dcp=%d: the per-call floor moved to %.9f ms from %.9f ms; a "+
				"launch cost is paid once per rank whatever the shard",
				dcp, intercept*1e3, intercept1*1e3)
		}
	}
}

// The structural check: a shard the pricer ignores is the defect this change fixes, so
// poke the resolved layout on a built kernel and require the price to move. This is what
// catches a field that is plumbed but never read -- which is exactly the state DCP was in.
func TestDCPGeometryReachesThePricer(t *testing.T) {
	k := fixture(t, dcpMLAFixture)
	batch := decodeBatch(32, 1, 32768)
	before := k.StepTime(batch).NoOverlap
	k.layout.DCP = 8
	after := k.StepTime(batch).NoOverlap
	if after >= before {
		t.Errorf("setting the resolved DCP width to 8 left the step at %v against %v; "+
			"the layout field is not reaching the pricer", after, before)
	}
}

// ---------------------------------------------------------------------------
// The collectives, and the coefficient key they resolve against.
// ---------------------------------------------------------------------------

// Sharding a sequence means no rank holds the whole context, so each computes a partial
// attention output and the group must combine them. That combine is what DCP trades for
// the capacity it buys, and omitting it would make DCP look free in both directions.
//
// THE PAYLOAD IS PER DECODE TOKEN, one query row per token the decode regime runs (the MQA
// rows, num_mqa_tokens). A prefill token takes no part in a decode combine, so the charge
// must be independent of how many prefill tokens share the step; and a speculative decode
// verifying two tokens moves two rows.
//
// A pure decode batch of one-token requests cannot show either, because there the request
// count, the decode-token count and the step's token count are one number -- which is why
// the discriminating cases are a MIXED batch (two decode rows beside a 2,048-token prefill)
// and a SPECULATIVE one (two requests of two tokens against four requests of one).
func TestDCPAddsACombineCollectiveSizedByDecodeTokens(t *testing.T) {
	link := func(dcp int, b kernel.Batch) float64 {
		return dcpKernel(t, dcpMLAFixture, dcp).StepTime(b).
			PerResource[kernel.ResourceNVLink].Seconds()
	}
	added := func(b kernel.Batch) float64 { return link(2, b) - link(1, b) }
	// Two decode rows, and the same two beside a prefill chunk.
	pure := decodeBatch(2, 1, 32768)
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 32767, PromptLen: 32768},
		{Scheduled: 1, Computed: 32767, PromptLen: 32768},
		{Scheduled: 2048, Computed: 0, PromptLen: 2048},
	}}
	if pure.Tokens() != 2 || mixed.Tokens() != 2050 {
		t.Fatalf("the batches must differ sharply in token count to discriminate: "+
			"got %d and %d", pure.Tokens(), mixed.Tokens())
	}

	// A combine must appear at all.
	if on, off := link(2, pure), link(1, pure); on <= off {
		t.Fatalf("the on-node collective term is %.6f ms at dcp=2 against %.6f ms at "+
			"dcp=1; a DCP combine must add communication", on*1e3, off*1e3)
	}

	// And it must be the SAME charge in both batches, since both hold two decode rows.
	// Charging the step's token count would make the mixed figure 1,025 times larger.
	// Compared to 2ns: each resource total is reported as a time.Duration, so a difference
	// of two totals carries up to a nanosecond of truncation from each.
	addedPure, addedMixed := added(pure), added(mixed)
	if math.Abs(addedMixed-addedPure) > 2e-9 {
		t.Errorf("the combine cost %.6f ms beside a prefill chunk against %.6f ms "+
			"without one, on the same two decode rows; a prefill token does not join a "+
			"decode combine", addedMixed*1e3, addedPure*1e3)
	}

	// Rows are TOKENS. Two speculative requests verifying two tokens each run four query
	// rows, exactly as four one-token decodes do, and the step's token count is four in
	// both, so every other collective is the same too.
	spec := decodeBatch(2, 2, 32768)
	four := decodeBatch(4, 1, 32768)
	if a, b := added(spec), added(four); math.Abs(a-b) > 2e-9 {
		t.Errorf("two requests of two decode tokens added %.6f ms of combine against "+
			"%.6f ms for four requests of one; a combine moves one row per decode token",
			a*1e3, b*1e3)
	}

	// Doubling the decode rows must double the BYTES, which is a weaker statement than
	// doubling the charge: a collective is floor + bytes/rate, and each of the three
	// launches pays its floor however small the payload. Subtracting the floors isolates
	// the half that scales. ag_rs launches a query all-gather, a log-sum-exp all-gather and
	// an output reduce-scatter, so the floors are two all-gathers and one reduce-scatter.
	//
	// Asserting the total would have been wrong, and was: an earlier draft of this test
	// required the total to double and failed against correct code.
	at2 := dcpKernel(t, dcpMLAFixture, 2)
	floors := float64(at2.plan.TotalLayers) *
		(2*at2.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupDCP}].Seconds() +
			at2.collectiveFloors[collKey{Op: model.OpReduceScatter, Group: price.GroupDCP}].Seconds())
	bytesAtTwo := addedPure - floors
	bytesAtFour := added(four) - floors
	if bytesAtTwo <= 0 {
		t.Fatalf("the combine's byte term is %.6f ms, not positive above its floors",
			bytesAtTwo*1e3)
	}
	// Exactly 2 up to the reporting resolution: each step's resource totals are whole
	// nanoseconds, and each byte term is a difference of two of them less a floor sum, so
	// it carries a few nanoseconds of truncation.
	if diff := bytesAtFour - 2*bytesAtTwo; math.Abs(diff) > 6e-9 {
		t.Errorf("doubling the decode rows scaled the combine's byte term by %.6f, want "+
			"exactly 2: one query row crosses per decode token", bytesAtFour/bytesAtTwo)
	}

	// A prefill-only step must carry no DCP combine at all.
	prefill := decodeBatch(1, 4096, 4096)
	if sharded, bare := link(8, prefill), link(1, prefill); sharded != bare {
		t.Errorf("a prefill-only step carried %.6f ms of collective at dcp=8 against "+
			"%.6f ms at dcp=1", sharded*1e3, bare*1e3)
	}
}

// The DCP collectives must resolve coefficients at THEIR OWN width, not the
// tensor-parallel one.
//
// This is what the (op, group) coefficient key exists for. A DCP all-gather and a
// tensor-parallel all-gather are the same primitive at two widths, and a single entry per
// op would have silently priced the narrow group at the wide group's floor. The gap is not
// a rounding error: at tp=8 and dcp=2 on h200 the two floors are 10.3us and 5.08us.
//
// Read from the resolved maps rather than from Provenance, which reports every coefficient
// the set carries rather than the ones this layout selected.
func TestTheDCPCollectivesArePricedAtTheirOwnWidthNotTheTensorParallelOne(t *testing.T) {
	k := dcpKernel(t, dcpMLAFixture, 2)
	if k.layout.TP != 8 {
		t.Fatalf("this fixture is expected to be tp=8, got %d: the test needs the two "+
			"widths to differ", k.layout.TP)
	}
	tp := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupTP}]
	dcp, ok := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupDCP}]
	if !ok || dcp <= 0 {
		t.Fatal("no all-gather floor resolved for the decode-context group")
	}
	if _, ok := k.collectiveFloors[collKey{
		Op: model.OpReduceScatter, Group: price.GroupDCP}]; !ok {
		t.Error("no reduce-scatter floor resolved for the decode-context group; the " +
			"ag_rs combine is a gather AND a scatter")
	}
	if dcp == tp {
		t.Errorf("the 2-rank DCP all-gather resolved the same floor as the 8-rank "+
			"tensor-parallel one (%v); the group is not selecting the width", dcp)
	}
	if dcp >= tp {
		t.Errorf("a 2-rank all-gather floor of %v is not below the 8-rank %v", dcp, tp)
	}
	if got, ok := k.groupSize(price.GroupDCP); !ok || got != 2 {
		t.Errorf("the decode-context group spans %d ranks, want 2", got)
	}

	// RESOLVING THE RIGHT TRIPLE IS NOT ENOUGH: the pricing path has to USE it. A
	// mutation keying the combine to the tensor-parallel axis leaves every assertion
	// above green while charging the narrow group the wide group's floor, which is the
	// precise failure the composite key exists to prevent.
	//
	// So price the combine and compare it against the floors it should have been built
	// from. Three collectives per layer under ag_rs -- two all-gathers and a
	// reduce-scatter -- each paying its own floor, which is the dominant term at this
	// payload. The 2-rank and 8-rank floor triples differ by about 2x on this part, the
	// same ratio that makes borrowing a width wrong in the first place.
	batch := decodeBatch(2, 1, 32768)
	link := func(width int) float64 {
		return dcpKernel(t, dcpMLAFixture, width).StepTime(batch).
			PerResource[kernel.ResourceNVLink].Seconds()
	}
	perLayer := (link(2) - link(1)) / float64(k.plan.TotalLayers)
	floorTriple := 2*dcp.Seconds() + k.collectiveFloors[collKey{
		Op: model.OpReduceScatter, Group: price.GroupDCP}].Seconds()
	tpTriple := 2*tp.Seconds() + k.collectiveFloors[collKey{
		Op: model.OpReduceScatter, Group: price.GroupTP}].Seconds()
	if perLayer < floorTriple {
		t.Errorf("the combine costs %.4fus per layer, below the %.4fus of floor its three "+
			"collectives must each pay", perLayer*1e6, floorTriple*1e6)
	}
	// The decisive bound: it must sit nearer the narrow triple than the wide one.
	if math.Abs(perLayer-floorTriple) >= math.Abs(perLayer-tpTriple) {
		t.Errorf("the combine costs %.4fus per layer, nearer the 8-rank floor triple "+
			"%.4fus than the 2-rank %.4fus; the decode-context collectives are being "+
			"priced at the tensor-parallel width",
			perLayer*1e6, tpTriple*1e6, floorTriple*1e6)
	}
}

// A deployment without DCP must not be REFUSED for want of a decode-context coefficient
// it has no use for.
//
// This is the one way the (op, group) key could regress a deployment that has nothing to
// do with DCP. An unmeasured width inside the search space is an error rather than a
// clamp, by design -- borrowing a narrower width's floor understates a collective by
// 1.53x to 1.91x across the parts measured at both -- so asking for a width a part lacks
// turns a working deployment into a construction failure.
//
// Stated as the OBSERVABLE consequence rather than as the absence of a map entry, because
// the absence has no consequence a caller can see on a registry where every part carries
// every width, which every part in this one does. So the test builds a coefficient set
// that is deliberately missing the narrow width and requires construction to succeed
// anyway while dcp is off, and to refuse once dcp asks for the width that is missing.
func TestADeploymentWithoutDCPIsNotRefusedForADCPCoefficient(t *testing.T) {
	// A part swept at 8 ranks only: a tensor-parallel group of 8 resolves, a 2-rank
	// decode-context group has nothing to resolve against.
	var entries []coefficient.Entry
	for _, op := range []string{"all_reduce", "all_gather", "reduce_scatter", "alltoall"} {
		entries = append(entries, tripleFor(op, "fp16", "h200", 8, 9.37)...)
	}
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}}
	k.layout.TP = 8

	// Tensor-parallel: resolves, because 8 is measured.
	if _, err := k.groupWidth(c, collKey{Op: model.OpAllGather, Group: price.GroupTP},
		"all_gather", "fp16", "h200"); err != nil {
		t.Errorf("a tensor-parallel group of 8 should resolve against an 8-rank sweep: %v",
			err)
	}
	// A decode-context group NARROWER than anything measured clamps to the narrowest
	// figure, which is the existing policy for a group below the sweep range and is the
	// conservative direction: the narrowest measured floor is the only one available, and
	// a narrower group cannot cost more than a wider one. What the resolver refuses is a
	// width INSIDE the search space that a part happens to lack, which is the case where
	// a fitted figure could exist and does not.
	k.layout.DCP = 2
	got, err := k.groupWidth(c, collKey{Op: model.OpAllGather, Group: price.GroupDCP},
		"all_gather", "fp16", "h200")
	if err != nil {
		t.Errorf("a 2-rank decode-context group below the sweep range should clamp to "+
			"the narrowest measured width, not refuse: %v", err)
	} else if got != 8 {
		t.Errorf("clamped to %d ranks; the only measured width is 8", got)
	}

	// And the refusal that does matter: a width inside the search space the part lacks.
	// Swept at 2 and 16 only, a 4-rank decode-context group must be refused rather than
	// priced at the 2-rank floor.
	var gapped []coefficient.Entry
	for _, op := range []string{"all_reduce", "all_gather", "reduce_scatter", "alltoall"} {
		for _, w := range []int{2, 16} {
			gapped = append(gapped, tripleFor(op, "fp16", "h200", w, 9.37)...)
		}
	}
	k.layout.DCP = 4
	if _, err := k.groupWidth(coeffs(t, gapped),
		collKey{Op: model.OpAllGather, Group: price.GroupDCP},
		"all_gather", "fp16", "h200"); err == nil {
		t.Error("a 4-rank decode-context group resolved against a part swept at 2 and " +
			"16 only; borrowing a narrower width's floor understates a collective by " +
			"1.53x to 1.91x and is the substitution liftCollectiveFloors refuses")
	}

	// And the behaviour that matters for every deployment already in the corpus: with DCP
	// off, a real kernel constructs and prices. If the DCP triples were resolved
	// unconditionally, a part lacking one of those widths would fail here instead.
	for _, scenario := range []string{
		dcpMLAFixture, dcpSparseFixture, dcpHybridFixture,
		"minimax-m25-h200-ep8.yaml", "minimax-m25-h200-ep72.yaml",
		"minimax-m25-gb300-tp4.yaml", "nemotron3-ultra-h100-agg.yaml",
		"kimi-k3-h100-nospec.yaml",
	} {
		if got := fixture(t, scenario).StepTime(decodeBatch(8, 1, 2048)).NoOverlap; got <= 0 {
			t.Errorf("%s priced a step at %v", scenario, got)
		}
	}
}

// The width a consumer reads must be the resolved one, and must floor at one so an absent
// field reads as "not sharded" rather than zero.
func TestDecodeContextParallelWidthComesFromTheResolvedLayout(t *testing.T) {
	if got := dcpKernel(t, dcpMLAFixture, 4).DecodeContextParallelWidth(); got != 4 {
		t.Errorf("got %d, want 4", got)
	}
	// The fixtures state no dcp at all, which is the absent case.
	if got := fixture(t, dcpMLAFixture).DecodeContextParallelWidth(); got != 1 {
		t.Errorf("an unstated width read as %d, want 1", got)
	}
}

// THE DEPLOYMENT THIS WAS WRITTEN FOR, priced arm by arm.
//
// The GLM-5.3-Flash canary runs tp=1 with --prefill-context-parallel-size 8,
// --decode-context-parallel-size 8, --dcp-comm-backend ag_rs and a 64-token KV block. That
// shape is why the two axes had to be separated rather than collapsed: at tp=1 there is no
// tensor-parallel width to divide anything, so before this change NOTHING in the layout
// moved the price of that deployment at all. glm5 stands in for it: the vendored catalog
// predates GLM-5.3-Flash, and both are sparse-MLA stacks.
//
// THREE ARMS, NOT FOUR. A DCP-only arm at tp=1 is a layout the engine refuses: with PCP off
// "DCP reuses the TP ranks", so tp must be divisible by dcp (vllm/config/parallel.py:566-569
// at v0.31.0), and blis-schemas v0.2.2 now enforces that too. An earlier form of this test
// priced tp=1, dcp=8 anyway, so its DCP-only assertions described a deployment that does
// not start. DCP alone is exercised at tp=8 by the tests above, where it is admissible.
//
// The backend is STATED, as the canary states it. GLM's model hook would otherwise choose
// a2a with a replicated query projection (models/config.py:43-50), which is a different
// set of collectives; stating ag_rs is what the measured deployment did.
//
// Asserted as a matrix of non-interference rather than as figures: PCP must move prefill
// and only prefill, and adding DCP to it must move decode and capacity and not prefill.
func TestTheContextParallelArmsOfTheCanaryDeploymentPriceIndependently(t *testing.T) {
	build := func(pcp, dcp int) *Kernel {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.TP = 1
		pool.Parallel.DP = 1
		pool.Parallel.PCP = pcp
		pool.Parallel.DCP = dcp
		pool.Engine.BlockSize = 64
		pool.Engine.DCPCommBackend = "ag_rs"
		k, err := New(in)
		if err != nil {
			t.Fatalf("tp=1 pcp=%d dcp=%d: %v", pcp, dcp, err)
		}
		return k
	}
	decode := decodeBatch(4, 1, 32768)
	prefill := decodeBatch(1, 2048, 2048)

	base := build(1, 1)
	pcpOnly := build(8, 1)
	both := build(8, 8)

	// PCP moves prefill and nothing else.
	if pcpOnly.StepTime(prefill).NoOverlap >= base.StepTime(prefill).NoOverlap {
		t.Error("pcp=8 did not lower the prefill step at tp=1, where no tensor-parallel " +
			"width can be doing the work instead")
	}
	if got, want := pcpOnly.StepTime(decode).NoOverlap,
		base.StepTime(decode).NoOverlap; got != want {
		t.Errorf("pcp=8 moved the decode step to %v from %v", got, want)
	}
	if got, want := pcpOnly.SequenceVariableBytes(32768),
		base.SequenceVariableBytes(32768); got != want {
		t.Errorf("pcp=8 moved the cache a rank holds to %d from %d", got, want)
	}

	// Adding DCP moves decode and capacity, and not an unchunked prefill: a prefill with no
	// computed prefix has no sharded context to read.
	if both.StepTime(decode).NoOverlap == pcpOnly.StepTime(decode).NoOverlap {
		t.Error("dcp=8 left the decode step where pcp alone put it; the shard and the " +
			"combine it adds both change a decode step")
	}
	if got, want := both.SequenceVariableBytes(32768),
		pcpOnly.SequenceVariableBytes(32768)/8; got != want {
		t.Errorf("dcp=8 holds %d bytes, want %d (an eighth of the unsharded %d)",
			got, want, pcpOnly.SequenceVariableBytes(32768))
	}
	if got, want := both.StepTime(prefill).NoOverlap,
		pcpOnly.StepTime(prefill).NoOverlap; got != want {
		t.Errorf("dcp=8 moved an unchunked prefill step to %v from %v", got, want)
	}
}
