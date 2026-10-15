package latencykernel

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Per-layer caches, as a graph states them (blis-schemas model.Cache), checked through
// what a simulator reads: SequenceVariableBytes, SequenceFixedBytes, PDTransferDemand and
// StepTime.

const v41 = "deepseek-v41-flash-h200-tp8.yaml"

// growth is the bytes a sequence's cache adds per context token far from any page edge.
func growth(k *Kernel) float64 {
	const lo, hi = 1 << 18, 1 << 20
	return float64(k.SequenceVariableBytes(hi)-k.SequenceVariableBytes(lo)) / (hi - lo)
}

// ownedRecordRate is what a fixture's owned growing caches hold per context token, before
// paging: each record over the tokens it covers, once per layer that owns it, on the graph
// as served on the fixture's chip and under the cache dtype its release resolves.
func ownedRecordRate(t *testing.T, in Inputs) float64 {
	t.Helper()
	family, _ := in.Chip.CapabilityMajor()
	g, err := in.Model.ForCapability(family)
	if err != nil {
		t.Fatal(err)
	}
	pool := in.Deployment.Pools[0]
	dtype := rulesFor(in.Scenario.EngineVersion).ResolveCacheDType(g.Global.Architecture,
		family, pool.Parallel.TP, pool.Engine.CacheDType)
	byID := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		byID[lk.ID] = lk
	}
	var rate float64
	for _, id := range g.Stack.Expand() {
		for _, n := range byID[id].Nodes {
			for _, c := range n.Caches {
				rate += c.BytesPerContextToken(dtype)
			}
		}
	}
	return rate
}

// The capacity a sequence takes grows at the rate of the records its layers OWN, not at one
// per-model width times every attention layer: DeepSeek-V4.1-Flash holds about 1,790 B a
// context token on SM90, where the inferred figure was 20,480. Paging adds at most the page
// alignment's rounding.
func TestStatedCachesGrowAtTheOwnedRecordRate(t *testing.T) {
	in := fixtureInputs(t, v41)
	want := ownedRecordRate(t, in)
	got := growth(build(t, in))
	if got < want || got > want*1.1 {
		t.Errorf("a sequence grows %.1f B a token; its owned records are %.1f B a token, "+
			"and paging adds at most a tenth", got, want)
	}

	// Inferring the caches instead -- the same graph with every cache statement removed --
	// charges every attention layer a full-length cache of its read width, which can only
	// be more.
	stripped := fixtureInputs(t, v41)
	stripCaches(stripped.Model)
	if inferred := growth(build(t, stripped)); inferred <= got {
		t.Errorf("inferred caches grow %.1f B a token, no more than the %.1f the owned "+
			"records hold", inferred, got)
	}
}

// stripCaches removes every cache statement and every reference to one, which is a graph
// that leaves its caches to inference.
func stripCaches(g *model.Graph) {
	for i := range g.LayerKinds {
		for j := range g.LayerKinds[i].Nodes {
			n := &g.LayerKinds[i].Nodes[j]
			n.Caches, n.WritesCache = nil, ""
			if n.Context != nil {
				n.Context.ReadsCache = ""
				n.Context.TokensPerEntry = max(n.Context.TokensPerEntry, 1)
			}
			if n.WritesCache == "" && n.PerTokens > 0 {
				n.PerTokens = 0
			}
		}
	}
	for i := range g.ArchOverrides {
		g.ArchOverrides[i].Caches = nil
	}
}

// Stating caches is a per-NODE opt-in. A layer whose scorer states its index cache while its
// primary attention states none keeps the primary's inferred cache: it is resident, and its
// decode read grows with context.
func TestCacheStatementsArePerNode(t *testing.T) {
	full := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	stripped := 0
	for i := range in.Model.LayerKinds {
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			switch {
			case n.Op == model.OpAttention && n.Role == "":
				stripped += len(n.Caches)
				n.Caches = nil
			case n.WritesCache != "" && n.WritesCache != "index":
				n.WritesCache, n.PerTokens = "", 0
			case n.Context != nil && n.Context.ReadsCache != "":
				n.Context.ReadsCache = ""
			}
		}
	}
	for i := range in.Model.ArchOverrides {
		var kept []model.CacheOverride
		for _, c := range in.Model.ArchOverrides[i].Caches {
			if c.Name == "index" {
				kept = append(kept, c)
			}
		}
		in.Model.ArchOverrides[i].Caches = kept
	}
	if stripped == 0 {
		t.Fatal("no primary attention stated a cache to strip")
	}
	in.Deployment.Pools[0].Engine.Speculative = nil
	partial := build(t, in)
	if p, f := growth(partial), growth(full); p <= f {
		t.Errorf("with its primaries' caches inferred a sequence grows %.1f B a token, no "+
			"more than the %.1f the stated records hold", p, f)
	}
	// Against the graph that states nothing: the primaries are inferred alike in both, and
	// the scorers' stated index caches only add to them. A layer whose scorer's statement
	// silenced its primary's inferred cache would grow less.
	none := fixtureInputs(t, v41)
	stripCaches(none.Model)
	none.Deployment.Pools[0].Engine.Speculative = nil
	if p, n := growth(partial), growth(build(t, none)); p <= n {
		t.Errorf("stating only the scorers' caches grows %.1f B a token, no more than "+
			"stating none (%.1f): a scorer's statement dropped its primary's cache", p, n)
	}
	short := partial.StepTime(decodeBatch(64, 1, 8192)).PerResource[kernel.ResourceHBM]
	long := partial.StepTime(decodeBatch(64, 1, 131072)).PerResource[kernel.ResourceHBM]
	if long <= short {
		t.Errorf("an inferred primary's decode read is flat in context: %v at 8K, %v at 128K",
			short, long)
	}
}

// A window ring is bounded per sequence: it is in SequenceFixedBytes, and a longer window
// grows that term and leaves the per-token growth alone.
func TestWindowRingsAreFixedPerSequence(t *testing.T) {
	base := build(t, fixtureInputs(t, v41))
	if base.SequenceFixedBytes() <= 0 {
		t.Fatal("a graph with window caches reports no fixed per-sequence bytes")
	}
	in := fixtureInputs(t, v41)
	for i := range in.Model.LayerKinds {
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			for c := range n.Caches {
				if w := n.Caches[c].Window; n.Caches[c].Kind == model.CacheWindow {
					// The read keeps whatever it reads beyond the ring (a parallel-block
					// draft also reads its block positions).
					n.Window += 3 * w
					n.Caches[c].Window = 4 * w
				}
			}
		}
	}
	wide := build(t, in)
	if wide.SequenceFixedBytes() <= base.SequenceFixedBytes() {
		t.Errorf("a 4x window holds %d fixed bytes a sequence, not more than %d",
			wide.SequenceFixedBytes(), base.SequenceFixedBytes())
	}
	if g, w := growth(base), growth(wide); g != w {
		t.Errorf("widening the window moved the per-token growth from %.1f to %.1f", g, w)
	}
}

// The drafter's rings are resident only while the deployment speculates.
func TestDraftCachesAreHeldOnlyWhileSpeculating(t *testing.T) {
	spec := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	in.Deployment.Pools[0].Engine.Speculative = nil
	plain := build(t, in)
	if spec.SequenceFixedBytes() <= plain.SequenceFixedBytes() {
		t.Errorf("speculating holds %d fixed bytes a sequence, not more than %d without",
			spec.SequenceFixedBytes(), plain.SequenceFixedBytes())
	}
}

// Which layer kind a graph declares last is not a property of the model. Reordering the
// declarations changes no price, and DeepSeek-V4.1-Flash, whose last declaration is a draft
// layer outside its target stack, is served the sparse-MLA cache dtype its target stack
// holds.
func TestLayerKindDeclarationOrderDoesNotPrice(t *testing.T) {
	a := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	lk := in.Model.LayerKinds
	for i, j := 0, len(lk)-1; i < j; i, j = i+1, j-1 {
		lk[i], lk[j] = lk[j], lk[i]
	}
	b := build(t, in)
	if a.FixedBytes() != b.FixedBytes() {
		t.Errorf("reversing the declarations moved FixedBytes from %+v to %+v",
			a.FixedBytes(), b.FixedBytes())
	}
	for _, n := range []int{1, 4096, 1 << 20} {
		if a.SequenceVariableBytes(n) != b.SequenceVariableBytes(n) {
			t.Errorf("reversing the declarations moved SequenceVariableBytes(%d)", n)
		}
	}
	for name, bt := range batchShapes() {
		if a.StepTime(bt).NoOverlap != b.StepTime(bt).NoOverlap {
			t.Errorf("%s: reversing the declarations moved the step time", name)
		}
	}

	ds := fixtureInputs(t, v41)
	ds.Deployment.Pools[0].Engine.CacheDType = "fp8_ds_mla"
	if _, err := New(ds); err != nil {
		t.Errorf("fp8_ds_mla refused for a sparse-MLA target stack: %v", err)
	}
}

// The graph is resolved for the chip it runs on: SM100's records are smaller than SM90's
// (and its compressed cache NVFP4 where mega attention serves it), so the same deployment on
// a B200 holds less a token. A chip that does not say which generation it is cannot run a
// graph that distinguishes generations.
func TestCachesFollowTheChipsArchitecture(t *testing.T) {
	h200 := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	b200 := chipInputs(t, "b200")
	in.Chip, in.Scenario.Cluster.Hardware = b200, "b200"
	if g, h := growth(build(t, in)), growth(h200); g >= h {
		t.Errorf("on B200 a sequence grows %.1f B a token, not less than H200's %.1f", g, h)
	}

	anon := fixtureInputs(t, v41)
	anon.Chip.ComputeCapability = ""
	if _, err := New(anon); err == nil {
		t.Error("a chip with no compute capability ran a graph that distinguishes generations")
	}
}

// auto is the engine fitting max_model_len to memory, capped at the model's window, and the
// kernel takes it at that cap: auto prices as stating the window does.
func TestAutoMaxModelLenPricesAsTheModelsWindow(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Deployment.Pools[0].Engine.MaxModelLen = deployment.MaxModelLenAuto
	auto := build(t, in)
	in = fixtureInputs(t, v41)
	in.Deployment.Pools[0].Engine.MaxModelLen =
		deployment.MaxModelLen(in.Model.Global.DefaultWindow())
	stated := build(t, in)
	if auto.SequenceFixedBytes() != stated.SequenceFixedBytes() {
		t.Errorf("auto holds %d fixed bytes, the stated window %d",
			auto.SequenceFixedBytes(), stated.SequenceFixedBytes())
	}
	for name, bt := range batchShapes() {
		if auto.StepTime(bt).NoOverlap != stated.StepTime(bt).NoOverlap {
			t.Errorf("%s: auto and the stated window price differently", name)
		}
	}
}

// A handoff moves the owned caches the way NIXL does: never more than the sequence holds,
// a window ring clipped to the pages its window spans, and the drafter's caches not at all.
func TestHandoffMovesOnlyWhatTheTargetOwns(t *testing.T) {
	k := build(t, fixtureInputs(t, v41))
	from, to := kernel.Placement{Node: 0}, kernel.Placement{Node: 1}
	for _, n := range []int{1, 1000, 8192, 131072} {
		moved := k.PDTransferDemand(n, from, to).Bytes
		held := k.SequenceVariableBytes(n) + k.SequenceFixedBytes()
		if moved > held {
			t.Errorf("a %d-token handoff moves %d bytes, more than the %d a sequence holds",
				n, moved, held)
		}
	}
	// Far from the start, a handoff grows at the owned records' rate: the rings stop
	// growing once they span their window.
	lo, hi := 1<<18, 1<<20
	rate := float64(k.PDTransferDemand(hi, from, to).Bytes-
		k.PDTransferDemand(lo, from, to).Bytes) / float64(hi-lo)
	if want := ownedRecordRate(t, fixtureInputs(t, v41)); rate < want || rate > 1.1*want {
		t.Errorf("a handoff grows %.1f B a token; the owned records are %.1f", rate, want)
	}
}

// A window bounds the decode read of every attention kind, not only sparse MLA: past its
// window, a windowed layer's HBM time does not grow with context.
func TestAWindowBoundsTheDecodeReadOfEveryKind(t *testing.T) {
	for _, c := range []struct {
		scenario string
		kind     model.AttentionKind
	}{
		{"minimax-m25-h200-tp8.yaml", model.AttentionSWA},
		{"aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml", model.AttentionMLA},
	} {
		t.Run(string(c.kind), func(t *testing.T) {
			in := fixtureInputs(t, c.scenario)
			for i := range in.Model.LayerKinds {
				for j := range in.Model.LayerKinds[i].Nodes {
					n := &in.Model.LayerKinds[i].Nodes[j]
					if n.Op == model.OpAttention && n.Role == "" {
						n.AttentionKind, n.Window = c.kind, 128
					}
				}
			}
			k := build(t, in)
			short := k.StepTime(decodeBatch(64, 1, 8192)).PerResource[kernel.ResourceHBM]
			long := k.StepTime(decodeBatch(64, 1, 131072)).PerResource[kernel.ResourceHBM]
			if long != short {
				t.Errorf("a 128-token window read %v at 8K and %v at 128K", short, long)
			}
		})
	}
}

// The selection bound, as properties over random layers and contexts: the read never
// shrinks as context grows, and a layer that selects a top-k reads at most its window plus
// that top-k whatever the context.
func TestSelectionBoundProperties(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		l := &price.PlannedLayer{AttnKind: model.AttentionSparseMLA}
		if r.Intn(2) == 0 {
			l.AttnWindow = 1 + r.Intn(4096)
		}
		if r.Intn(2) == 0 {
			l.AttnIndexTopK = 1 + r.Intn(4096)
		}
		if r.Intn(2) == 0 {
			l.AttnCompressRatio = 1 + r.Intn(128)
		}
		if !readBounded(l) {
			continue
		}
		prev := 0.0
		for _, ctx := range []int{1, 7, 128, 1000, 4096, 65536, 1 << 20} {
			got := selectedKVTokensFor(l, ctx)
			if got < prev {
				t.Fatalf("%+v: read %v at %d, less than %v at a shorter context", *l, got, ctx, prev)
			}
			if l.AttnIndexTopK > 0 &&
				got > float64(l.AttnWindow)+float64(l.AttnIndexTopK)+1e-9 {
				t.Fatalf("%+v: read %v at %d, more than window plus top-k", *l, got, ctx)
			}
			prev = got
		}
	}
}

// A cache sized at the record the graph states for the engine's own default is not an
// assumption: only a deployment running another dtype, which the graph lists no record for,
// is recorded.
func TestTheDefaultCacheRecordIsNotAnAssumption(t *testing.T) {
	for _, chip := range []string{"h200", "b200"} {
		in := fixtureInputs(t, v41)
		if chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
		}
		if recorded(build(t, in), "cache_record") {
			t.Errorf("%s: the default cache dtype's record was recorded as an assumption", chip)
		}
	}
}

// vLLM's block for DeepSeek-V4.1-Flash on SM90 at block 64 under fp8_ds_mla: the full-attention
// group's pages, 3 x 19,008 + 37,440 + 3 x 4,608 + 8,640 B, which every group's block costs
// (_get_kv_cache_bytes_per_block, vllm/v1/core/kv_cache_utils.py:1606-1632). It is the figure
// vLLM's own functions return for the same layers in blis-schemas' golden vectors
// (rules/testdata/kvgroups), where the 43 window layers fall into 8 groups of at most six.
const v41PoolBlock = 116928

// The stated caches are sized in whole blocks of the engine's shared pool, group by group: a
// sequence grows one block per 64 tokens of context, and holds in steady decode six pages in
// each of the eight window groups and one in the compressor-ring group, 49 blocks
// (inference-sim/blis-latency-kernel#46). At admission each window group holds its cap,
// cdiv(127 + 2 x 8,192, 32) + 1 = 517 pages, the two batches asynchronous scheduling keeps in
// flight (vllm/config/vllm.py:592-613; vllm/v1/kv_cache_interface.py:840-866).
func TestStatedCachesAreSizedInWholePoolBlocks(t *testing.T) {
	k := build(t, fixtureInputs(t, v41))
	if got := k.SequenceVariableBytes(64); got != v41PoolBlock {
		t.Errorf("64 tokens hold %d B, want one pool block of %d", got, v41PoolBlock)
	}
	if got := k.SequenceVariableBytes(131072); got != 2048*v41PoolBlock {
		t.Errorf("131,072 tokens hold %d B, want 2,048 pool blocks", got)
	}
	if got := k.SequenceFixedBytes(); got != (8*6+1)*v41PoolBlock {
		t.Errorf("steady decode holds %d B a sequence, want 49 pool blocks, %d", got,
			(8*6+1)*v41PoolBlock)
	}
	if got := k.SequenceAdmissionBytes(); got != (8*517+1)*v41PoolBlock {
		t.Errorf("admission reserves %d B a sequence, want 4,137 pool blocks, %d", got,
			(8*517+1)*v41PoolBlock)
	}
}

// engineCapacity is vLLM's "GPU KV cache size" for a kernel's figures, from a KV pool of
// gib GiB: the pool's blocks over a request's blocks at max_model_len, times max_model_len
// (get_kv_cache_capacity, vllm/v1/core/kv_cache_utils.py:2444-2471;
// get_max_concurrency_for_kv_cache_config, :1090-1122), with the slack a log's two-decimal
// GiB leaves in it.
func engineCapacity(k *Kernel, gib float64, maxModelLen int) (tokens, slack float64) {
	block := k.SequenceVariableBytes(k.blockSize)
	pool := int64(gib*(1<<30)) / block
	perRequest := (k.SequenceAdmissionBytes() + k.SequenceVariableBytes(maxModelLen)) / block
	tokens = float64(pool) / float64(perRequest) * float64(maxModelLen)
	return tokens, tokens * 0.005 * (1 << 30) / float64(block) / float64(pool)
}

// GOLDEN, from the engine's own log, H200. curvebender's DeepSeek-V4.1-Flash deployment on
// 8x H200 under vLLM v0.31.0 (curvebender-tools deployments/deepseek-v4.1-flash/scc-h200/
// model-server/deepseek-v4.1-flash.yaml: TP 8 with --enable-expert-parallel, DSpark K=5,
// --max-model-len 1048576, --max-num-batched-tokens 8192, --block-size 64,
// max_cudagraph_capture_size 64) reported 61.18 GiB of KV cache and 28,706,997 tokens
// (vllm:cache_config_info; reports/deepseek-v4.1-flash/deepseek-v4.1-flash-h200-vllm.html,
// and inference-sim/blis-latency-kernel#46). From the same 61.18 GiB the kernel's figures give
// the same count, to the rounding of the logged GiB; sized at the steady decode hold they
// would give 25% more.
func TestKVCapacityIsTheEnginesOwnFigureOnH200(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Deployment.Pools[0].Parallel.EnableExpertParallel = true
	e := &in.Deployment.Pools[0].Engine
	capture := 64
	e.MaxModelLen, e.BlockSize, e.MaxCUDAGraphCaptureSize = 1<<20, 64, &capture
	tokens, slack := engineCapacity(build(t, in), 61.18, 1<<20)
	if d := tokens - 28706997; d > slack || d < -slack {
		t.Errorf("the kernel sizes %.0f tokens in vLLM's 61.18 GiB, against its logged "+
			"28,706,997 (slack %.0f)", tokens, slack)
	}
}

// GOLDEN, the five arms of curvebender's H200 sizing table under vLLM v0.31.0
// (reports/deepseek-v4.1-flash/deepseek-v4.1-flash-h200-vllm.html, "Sizing", from vLLM boot
// logs and vllm:cache_config_info; the baseline is the deployment above). Each arm's capacity
// follows from its own logged KV pool: without DSpark its 40 window rings fill seven groups,
// not eight; at TP 4 a request's caches are the same, since a latent cache is not divided
// across tensor parallelism. The weights move between TP 8 and TP 4 as the kernel's sharded
// weights do, to 0.3%: 81.49 - 45.81 = 35.68 GiB measured. The weights themselves are not
// asserted: vLLM holds about 9 GiB more than the kernel at both widths, a term that does
// not scale with TP and that the kernel does not yet account for (see computeFixedBytes).
func TestCurvebenderH200ArmsAreTheEnginesFigures(t *testing.T) {
	type arm struct {
		name       string
		tp         int
		dspark     bool
		kvGiB      float64
		tokens     float64
		weightsGiB float64
	}
	arms := []arm{
		{"TP8-baseline", 8, true, 61.18, 28706997, 45.81},
		{"TP8-no-DSpark", 8, false, 65.17, 31367749, 42.64},
		{"TP8-no-KV-offload", 8, true, 61.26, 28743634, 45.73},
		{"TP8-no-Engram", 8, true, 61.18, 28706997, 45.81},
		{"TP4-baseline", 4, true, 25.35, 11893655, 81.49},
	}
	weights := map[string]float64{}
	for _, a := range arms {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.EnableExpertParallel = a.tp, true
		e := &in.Deployment.Pools[0].Engine
		capture := 64
		e.MaxModelLen, e.BlockSize, e.MaxCUDAGraphCaptureSize = 1<<20, 64, &capture
		if !a.dspark {
			e.Speculative = nil
		}
		k := build(t, in)
		tokens, slack := engineCapacity(k, a.kvGiB, 1<<20)
		if d := tokens - a.tokens; d > slack || d < -slack {
			t.Errorf("%s: the kernel sizes %.0f tokens in vLLM's %.2f GiB, against its "+
				"logged %.0f (slack %.0f)", a.name, tokens, a.kvGiB, a.tokens, slack)
		}
		weights[a.name] = float64(k.FixedBytes().Weights) / (1 << 30)
	}
	measured := arms[4].weightsGiB - arms[0].weightsGiB
	if got := weights["TP4-baseline"] - weights["TP8-baseline"]; math.Abs(got-measured) > 0.003*measured {
		t.Errorf("TP 8 to TP 4 adds %.2f GiB of weights a rank, against vLLM's %.2f", got,
			measured)
	}
}

// GOLDEN, from the engine's own log, H100. InferenceX's DeepSeek-V4.1-Flash AgentX recipe on
// 8x H100 (benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/h100-fp4-mtp/agentic.yaml,
// override_tp8_c1: TP 8 without expert parallelism, DSpark K=5, --max-model-len 1048576,
// --max-num-batched-tokens 4096, --max-num-seqs 2, max-cudagraph-capture-size 16,
// --gpu-memory-utilization 0.92, the V2 model runner) logged at concurrency 1 "13.47 GiB KV
// per GPU, 7,022,899 tokens, 6.70x maximum concurrency at 1M tokens per request"
// (perf-changelog.yaml, InferenceX PR 2974). Its image is a vLLM nightly
// (nightly-cd10ed6f), whose pool sizing this checks against v0.31.0's: 8 window groups at
// cdiv(127 + 2 x 4,096, 32) + 1 = 261 pages a request.
func TestKVCapacityIsTheEnginesOwnFigureOnH100(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, "h100"), "h100"
	e := &in.Deployment.Pools[0].Engine
	capture := 16
	e.MaxModelLen, e.MaxNumBatchedTokens, e.MaxNumSeqs = 1<<20, 4096, 2
	e.GPUMemoryUtilization, e.MaxCUDAGraphCaptureSize = 0.92, &capture
	tokens, slack := engineCapacity(build(t, in), 13.47, 1<<20)
	if d := tokens - 7022899; d > slack || d < -slack {
		t.Errorf("the kernel sizes %.0f tokens in vLLM's 13.47 GiB, against its logged "+
			"7,022,899 (slack %.0f)", tokens, slack)
	}
}

// PROPERTIES of the admission reservation: never below the steady decode hold, growing
// with the token budget, and the same as it for a model with no window cache.
func TestAdmissionReservationProperties(t *testing.T) {
	prev := int64(0)
	for _, mnbt := range []int{2048, 8192, 16384, 32768} {
		in := fixtureInputs(t, v41)
		in.Deployment.Pools[0].Engine.MaxNumBatchedTokens = mnbt
		k := build(t, in)
		adm := k.SequenceAdmissionBytes()
		if adm < k.SequenceFixedBytes() {
			t.Errorf("mnbt %d: admission %d B below the steady hold %d", mnbt, adm,
				k.SequenceFixedBytes())
		}
		if adm <= prev {
			t.Errorf("mnbt %d: admission %d B, not above a smaller budget's %d", mnbt, adm, prev)
		}
		prev = adm
	}
	v3 := fixture(t, memoryFixture)
	if v3.SequenceAdmissionBytes() != v3.SequenceFixedBytes() {
		t.Errorf("deepseek-v3 states no window cache, yet admits at %d B against %d",
			v3.SequenceAdmissionBytes(), v3.SequenceFixedBytes())
	}
}

// ungroupedRules is a rules pack that cannot group stated caches.
type ungroupedRules struct{ resolve.EngineRules }

func (ungroupedRules) KVCacheGroups(rules.KVCacheGroupsInput) (rules.KVCacheGrouping, error) {
	return rules.KVCacheGrouping{}, errors.New("not transcribed")
}

// Where the rules pack cannot group the stated caches, each is sized at its own page, and
// that is recorded rather than silent.
func TestAnUngroupedPoolIsRecorded(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Rules = ungroupedRules{in.Rules}
	k := build(t, in)
	if !recorded(k, "kv_cache_grouping") {
		t.Error("the caches were sized per cache with nothing in Provenance saying so")
	}
	if pooled := build(t, fixtureInputs(t, v41)); k.SequenceFixedBytes() >= pooled.SequenceFixedBytes() {
		t.Errorf("per-cache pages hold %d B a sequence, not less than the pool's %d",
			k.SequenceFixedBytes(), pooled.SequenceFixedBytes())
	}
}

// THE BOUNDED HOLD AS A FUNCTION OF WHAT IS IN FLIGHT. SequenceBoundedBytes(1 + K) is the
// steady decode hold and SequenceBoundedBytes(max_in_flight_tokens) the admission cap; the
// hold never falls as tokens in flight or tokens computed grow; and once a request has
// computed a window's worth, what it has computed no longer matters, as vLLM frees every
// block wholly behind the window (SlidingWindowManager.get_num_skipped_tokens and
// remove_skipped_blocks, vllm/v1/core/single_type_kv_cache_manager.py:687-725, 1153-1175 at
// v0.31.0). A request that has seen nothing holds only its compressor rings.
func TestSequenceBoundedBytesProperties(t *testing.T) {
	k := build(t, fixtureInputs(t, v41))
	if got, want := k.SequenceBoundedBytes(6), k.SequenceFixedBytes(); got != want {
		t.Errorf("SequenceBoundedBytes(1 + K) = %d, want SequenceFixedBytes %d", got, want)
	}
	if got, want := k.SequenceBoundedBytes(2*8192), k.SequenceAdmissionBytes(); got != want {
		t.Errorf("SequenceBoundedBytes(max_in_flight_tokens) = %d, want "+
			"SequenceAdmissionBytes %d", got, want)
	}
	// A ring crosses a page edge where window - 1 + in flight passes a multiple of 32:
	// 33 in flight page 160 tokens in 5 pages + 1, 34 page 161 in 6 + 1, one more block in
	// each of the eight window groups.
	if d := k.SequenceBoundedBytes(34) - k.SequenceBoundedBytes(33); d != 8*v41PoolBlock {
		t.Errorf("the 34th token in flight adds %d B, want one block in each of 8 window "+
			"groups, %d", d, 8*v41PoolBlock)
	}
	const window = 128
	prevFlight := int64(-1)
	for _, f := range []int{0, 1, 6, 31, 32, 33, 512, 8192, 16384, 65536} {
		b := k.SequenceBoundedBytes(f)
		if b < prevFlight {
			t.Errorf("%d tokens in flight hold %d B, below fewer tokens' %d", f, b, prevFlight)
		}
		prevFlight = b
		prevComputed := int64(-1)
		for _, c := range []int{0, 1, 31, 64, 127, 128, 129, 1000, 100000} {
			at := k.SequenceBoundedBytesAt(c, f)
			if at < prevComputed {
				t.Errorf("(%d computed, %d in flight) holds %d B, below fewer computed's %d",
					c, f, at, prevComputed)
			}
			prevComputed = at
			if c >= window && at != b {
				t.Errorf("(%d computed, %d in flight) holds %d B; past the window it must be "+
					"SequenceBoundedBytes(%d) = %d", c, f, at, f, b)
			}
			if at > b {
				t.Errorf("(%d computed, %d in flight) holds %d B, above the window's cap %d",
					c, f, at, b)
			}
		}
	}
	if got, want := k.SequenceBoundedBytesAt(0, 0), int64(v41PoolBlock); got != want {
		t.Errorf("a request that has seen nothing holds %d B, want its ring group's one "+
			"block, %d", got, want)
	}
}
