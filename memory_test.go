package latencykernel

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// Fixed memory occupancy is the registry's magnitudes in NVIDIA's composition, with nothing
// the kernel hardcodes and nothing Provenance cannot see (inference-sim/blis-latency-kernel#22).
// liftMemory states the form and its sources.

// memoryFixture is the deployment #22 measured: deepseek-v3, tp=8, fp8 on h200, 8,192
// batched tokens, PIECEWISE.
const memoryFixture = "aisimulate/deepseek-v3-h200-fp8-sglang-tp8.yaml"

// registryValue reads one coefficient the way the kernel resolved it for this deployment,
// so a test can state the composition against the registry rather than against literals a
// refit would invalidate.
func registryValue(t *testing.T, in Inputs, k *Kernel, name string) float64 {
	t.Helper()
	c, err := resolve.Load(in.Coefficients, resolve.Scope{
		Hardware: in.Chip.Name, Model: in.Model.Name, TP: k.layout.TP,
		EP: k.layout.ExpertWidth, NodesSpanned: k.layout.NodesSpanned,
	})
	if err != nil {
		t.Fatal(err)
	}
	v, err := c.Value(name)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// THE COMPOSITION, as an oracle over the registry, at every tensor-parallel width the
// registry declares a communicator for and below it.
//
//	CommBuffers    = nccl_communicator_bytes_<tp>rank (none at one rank) + engine_workspace_bytes
//	CUDAGraph      = cudagraph_capture_bytes_<mode>
//	ActivationPeak = max(activation_buffer_count_moe_<tp>rank * batched * hidden * 2,
//	                     activation_scratch_floor_bytes)
//
// deepseek-v3 is MoE and latent, so its multiple is the MoE row and its width is the hidden
// size. Every term must vary with tp exactly where its coefficient does -- which is the
// defect #22 opens with: the communicator was 392 MiB at every width.
func TestFixedOccupancyIsTheRegistrysComposition(t *testing.T) {
	for _, tp := range []int{1, 2, 4, 8} {
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Parallel.TP = tp
		k, err := New(in)
		if err != nil {
			t.Fatalf("tp=%d: %v", tp, err)
		}
		got := k.FixedBytes()
		v := func(name string) float64 { return registryValue(t, in, k, name) }

		comm := v("engine_workspace_bytes")
		if tp > 1 {
			comm += v(fmt.Sprintf("nccl_communicator_bytes_%drank", tp))
		}
		if got.CommBuffers != int64(comm) {
			t.Errorf("tp=%d: CommBuffers %d, want %d (communicator + workspace)",
				tp, got.CommBuffers, int64(comm))
		}
		if want := int64(v("cudagraph_capture_bytes_piecewise")); got.CUDAGraph != want {
			t.Errorf("tp=%d: CUDAGraph %d, want the piecewise capture %d", tp, got.CUDAGraph, want)
		}
		batched := float64(in.Deployment.Pools[0].Engine.MaxNumBatchedTokens)
		act := math.Max(v(fmt.Sprintf("activation_buffer_count_moe_%drank", tp))*batched*
			float64(in.Model.Global.HiddenSize)*2, v("activation_scratch_floor_bytes"))
		if got.ActivationPeak != int64(act) {
			t.Errorf("tp=%d: ActivationPeak %d, want %d", tp, got.ActivationPeak, int64(act))
		}
	}
}

// THE FIGURES #22 STATES, against the vendored registry (blis-registry v0.1.1). Pinned
// because the issue's verification names them; a re-vendoring that changes them is
// supposed to break this, and the commit that does so says why.
//
//	at tp=8: activation 1,174,405,120  capture 858,783,744  communicator 411,041,792
//	         workspace 3,758,096,384 -- 6,202,327,040 of fixed occupancy beside weights
//	at tp=2: the communicator is 358,612,992, not the 411,041,792 a constant gave
func TestFixedOccupancyMatchesTheFiguresTheIssueStates(t *testing.T) {
	k := fixture(t, memoryFixture)
	got := k.FixedBytes()
	if got.ActivationPeak != 1174405120 || got.CUDAGraph != 858783744 ||
		got.CommBuffers != 411041792+3758096384 {
		t.Errorf("tp=8: activation %d, capture %d, communicator+workspace %d; want "+
			"1174405120, 858783744 and 4169138176", got.ActivationPeak, got.CUDAGraph,
			got.CommBuffers)
	}
	if nonWeight := got.Total() - got.Weights - got.EPLBRedundant; nonWeight != 6202327040 {
		t.Errorf("tp=8: %d bytes of fixed occupancy beside weights, want 6202327040", nonWeight)
	}
	in := fixtureInputs(t, memoryFixture)
	in.Deployment.Pools[0].Parallel.TP = 2
	at2, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	if comm := at2.FixedBytes().CommBuffers - 3758096384; comm != 358612992 {
		t.Errorf("tp=2: communicator %d bytes, want 358612992", comm)
	}
}

// OCCUPANCY ENTERS NO STEP TIME. Scaling every memory magnitude in the registry threefold
// must move FixedBytes and leave every step, transfer and overhead exactly where it was:
// these terms decide how many sequences fit, not how long a step takes. It is the
// metamorphic form of #22's requirement that cmd/shape and cmd/score do not move.
func TestMemoryMagnitudesMoveOccupancyAndNoStepTime(t *testing.T) {
	memory := func(name string) bool {
		for _, p := range []string{"nccl_communicator_bytes_", "engine_workspace_bytes",
			"cudagraph_capture_bytes_", "activation_buffer_count_",
			"activation_scratch_floor_bytes"} {
			if strings.HasPrefix(name, p) {
				return true
			}
		}
		return false
	}
	base := fixtureInputs(t, memoryFixture)
	scaled := fixtureInputs(t, memoryFixture)
	for i, s := range scaled.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if memory(c.Coefficients[j].Name) {
				c.Coefficients[j].Value *= 3
			}
		}
		scaled.Coefficients[i] = &c
	}
	a, err := New(base)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(scaled)
	if err != nil {
		t.Fatal(err)
	}
	fa, fb := a.FixedBytes(), b.FixedBytes()
	if fb.CommBuffers != 3*fa.CommBuffers || fb.CUDAGraph != 3*fa.CUDAGraph ||
		fb.ActivationPeak != 3*fa.ActivationPeak {
		t.Errorf("tripling the memory magnitudes gave %+v from %+v; each term is linear in "+
			"its coefficient above the activation floor", fb, fa)
	}
	if fb.Weights != fa.Weights || fb.EPLBRedundant != fa.EPLBRedundant {
		t.Error("the memory magnitudes moved the weights or the EPLB replicas")
	}
	for shape, batch := range batchShapes() {
		if x, y := a.StepTime(batch), b.StepTime(batch); x.Overlap != y.Overlap ||
			x.NoOverlap != y.NoOverlap {
			t.Errorf("%s: the memory magnitudes moved the step from %v/%v to %v/%v",
				shape, x.Overlap, x.NoOverlap, y.Overlap, y.NoOverlap)
		}
	}
	if a.SequenceVariableBytes(8192) != b.SequenceVariableBytes(8192) ||
		a.AdmissionOverhead(1024) != b.AdmissionOverhead(1024) {
		t.Error("the memory magnitudes moved per-sequence capacity or a host overhead")
	}
}

// ACTIVATION SCRATCH DOES NOT FOLLOW THE SERVED WEIGHT FORMAT. #22 proposed sizing it at the
// served width, which would halve it for fp8. blis-registry#33 established the opposite and
// this kernel follows it: vLLM's fp8 linears return out_dtype=x.dtype
// (vllm/model_executor/kernels/linear/scaled_mm/cutlass.py at v0.31.0), so the residual
// stream and the inter-layer buffers stay bf16 under fp8 weights.
//
// qwen3-30b-a3b is a bf16 checkpoint, so serving it fp8 is a genuine change of weight
// format; the activation scratch must not move while the weights do.
func TestActivationScratchDoesNotFollowTheServedWeightFormat(t *testing.T) {
	qwen, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", "qwen3-30b-a3b", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(quant string) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		in.Model, in.Scenario.Model = qwen, qwen.Name
		in.Deployment.Pools[0].Engine.Quantization = quant
		k, err := New(in)
		if err != nil {
			t.Fatalf("quantization %q: %v", quant, err)
		}
		return k
	}
	bf16, fp8 := build(""), build("fp8")
	if bf16.FixedBytes().Weights <= fp8.FixedBytes().Weights {
		t.Fatalf("serving the bf16 checkpoint fp8 did not shrink its weights (%d against "+
			"%d); the comparison needs a real change of format",
			fp8.FixedBytes().Weights, bf16.FixedBytes().Weights)
	}
	if a, b := bf16.FixedBytes().ActivationPeak, fp8.FixedBytes().ActivationPeak; a != b {
		t.Errorf("activation scratch is %d bytes served bf16 and %d served fp8; activations "+
			"stay at the model's compute dtype", a, b)
	}
}

// THE FLOOR. A small token budget is charged NVIDIA's minimum scratch, not its product
// (AISimulate clamps from below at MIN_ACTIVATION_BYTES), and above the floor the scratch
// is linear in the budget.
func TestActivationScratchIsLinearInTheBudgetAboveItsFloor(t *testing.T) {
	at := func(batched int) int64 {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Engine.MaxNumBatchedTokens = batched
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.FixedBytes().ActivationPeak
	}
	in := fixtureInputs(t, memoryFixture)
	k := fixture(t, memoryFixture)
	floor := int64(registryValue(t, in, k, "activation_scratch_floor_bytes"))
	if got := at(1); got != floor {
		t.Errorf("a one-token budget gave %d bytes of scratch, want the floor %d", got, floor)
	}
	if a, b := at(4096), at(8192); b != 2*a || a <= floor {
		t.Errorf("4,096 and 8,192 batched tokens gave %d and %d bytes; above the floor the "+
			"scratch doubles with the budget", a, b)
	}
}

// h IS HEADS x HEAD_DIM, NOT HIDDEN_SIZE, for a model whose config declares a head_dim that
// differs. qwen3-30b-a3b runs 32 heads of 128 over a 2,048 hidden size, so its scratch per
// token is twice what hidden_size gives. The kernel reads it from the graph's attention
// node; a latent layer's d_h is a cache width, not a head_dim, so a latent model uses its
// hidden size (see activationWidth for the one place that can differ from NVIDIA's figure).
func TestActivationWidthIsHeadsTimesHeadDim(t *testing.T) {
	qwen, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", "qwen3-30b-a3b", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
	in.Model, in.Scenario.Model = qwen, qwen.Name
	in.Deployment.Pools[0].Engine.Quantization = ""
	k, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := k.activationWidth, 4096.0; got != want {
		t.Errorf("qwen3-30b-a3b's activation width is %v, want 32 x 128 = %v, not its "+
			"hidden size %d", got, want, qwen.Global.HiddenSize)
	}
	if got := fixture(t, memoryFixture).activationWidth; got != 7168 {
		t.Errorf("deepseek-v3's activation width is %v, want its hidden size 7168: its "+
			"config states no head_dim", got)
	}
}

// THE CAPTURE IS KEYED BY THE MODE THAT RUNS, and an unstated mode is vLLM v0.31.0's
// default, FULL_AND_PIECEWISE (vllm/config/compilation.py:614) -- for memory and for host
// time alike, and disclosed as an assumption because the engine downgrades it where a
// backend cannot capture a full graph.
func TestTheCaptureIsTheResolvedModesAndAnUnstatedModeIsTheDefault(t *testing.T) {
	build := func(mode string) *Kernel {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Engine.CUDAGraphMode = mode
		k, err := New(in)
		if err != nil {
			t.Fatalf("mode %q: %v", mode, err)
		}
		return k
	}
	in := fixtureInputs(t, memoryFixture)
	ref := fixture(t, memoryFixture)
	for _, mode := range []string{"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"} {
		want := int64(registryValue(t, in, ref, "cudagraph_capture_bytes_"+strings.ToLower(mode)))
		if got := build(mode).FixedBytes().CUDAGraph; got != want {
			t.Errorf("%s: capture %d, want %d", mode, got, want)
		}
	}
	if build("NONE").FixedBytes().CUDAGraph != 0 {
		t.Error("an eager deployment holds capture memory")
	}

	unstated, stated := build(""), build("FULL_AND_PIECEWISE")
	if unstated.FixedBytes() != stated.FixedBytes() {
		t.Error("an unstated mode's occupancy differs from FULL_AND_PIECEWISE's")
	}
	for shape, b := range batchShapes() {
		if unstated.StepTime(b).NoOverlap != stated.StepTime(b).NoOverlap {
			t.Errorf("%s: an unstated mode priced differently from FULL_AND_PIECEWISE", shape)
		}
	}
	if !hasAssumption(unstated, "cudagraph_mode") || hasAssumption(stated, "cudagraph_mode") {
		t.Error("an unstated mode must be disclosed as a kernel assumption, and a stated one " +
			"must not be")
	}
}

// THE MODE IS CHOSEN PER BATCH. vLLM v0.31.0 dispatches a uniform decode batch to a full
// graph under FULL_AND_PIECEWISE and FULL_DECODE_ONLY, and every other batch to piecewise or
// to no graph respectively (vllm/config/compilation.py:604-640,
// vllm/v1/cudagraph_dispatcher.py:233-310). So the host term of each mode must equal the
// host term of the single mode that batch actually runs under. A uniform decode batch with
// speculation verifies 1 + num_spec_tokens tokens per request (gpu_model_runner.py:883).
func TestTheGraphModeIsChosenPerBatch(t *testing.T) {
	host := func(mode string, spec int, b kernel.Batch) int64 {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		e := &in.Deployment.Pools[0].Engine
		e.CUDAGraphMode = mode
		if spec > 0 {
			e.Speculative = &deployment.Speculative{Method: "mtp", NumSpecTokens: spec}
		}
		k, err := New(in)
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		return int64(k.StepTime(b).PerResource[kernel.ResourceHost])
	}
	decode := decodeBatch(32, 1, 4096)
	mixed := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 1, Computed: 4095, PromptLen: 4096},
		{Scheduled: 256, Computed: 0, PromptLen: 256}, // 257 tokens: inside the 512 capture
	}}
	for _, c := range []struct {
		mode        string
		decodeAs    string
		otherwiseAs string
	}{
		{"FULL_AND_PIECEWISE", "FULL", "PIECEWISE"},
		{"FULL_DECODE_ONLY", "FULL", "NONE"},
	} {
		if a, b := host(c.mode, 0, decode), host(c.decodeAs, 0, decode); a != b {
			t.Errorf("%s priced a uniform decode batch's host time at %d, want %s's %d",
				c.mode, a, c.decodeAs, b)
		}
		if a, b := host(c.mode, 0, mixed), host(c.otherwiseAs, 0, mixed); a != b {
			t.Errorf("%s priced a mixed batch's host time at %d, want %s's %d",
				c.mode, a, c.otherwiseAs, b)
		}
	}
	// The three modes differ on a mixed batch, or the comparison above proves nothing.
	if host("FULL", 0, mixed) == host("PIECEWISE", 0, mixed) ||
		host("PIECEWISE", 0, mixed) == host("NONE", 0, mixed) {
		t.Fatal("FULL, PIECEWISE and NONE price a mixed batch's host time alike")
	}
	// Under speculation a decode of 1 + k tokens per request is the uniform batch.
	spec := decodeBatch(32, 3, 4096)
	if a, b := host("FULL_AND_PIECEWISE", 2, spec), host("FULL", 2, spec); a != b {
		t.Errorf("a uniform speculative decode (3 tokens, 2 speculative) priced %d under "+
			"FULL_AND_PIECEWISE, want FULL's %d", a, b)
	}
}

// PROVENANCE SEES EVERY FIGURE FixedBytes REPORTS, and Evidence counts them. Before #22 the
// 1.3 GB of hardcoded occupancy had no entry at all, so Evidence's "measured of total"
// overstated the prediction's footing. Every memory coefficient the composition read must
// now appear, from the registry set that supplied it; and an engine default the kernel
// filled in must appear as a kernel assumption, which a fully stated deployment has none of.
func TestProvenanceSeesEveryFigureFixedBytesReports(t *testing.T) {
	k := fixture(t, memoryFixture)
	sets := map[string]string{}
	for _, o := range k.Provenance() {
		sets[o.Name] = o.Set
	}
	for _, name := range []string{
		"nccl_communicator_bytes_8rank", "engine_workspace_bytes",
		"cudagraph_capture_bytes_piecewise", "activation_buffer_count_moe_8rank",
		"activation_scratch_floor_bytes",
	} {
		set, ok := sets[name]
		if !ok {
			t.Errorf("Provenance has no entry for %s, which FixedBytes reports", name)
		}
		if set == KernelAssumptionSet {
			t.Errorf("%s is attributed to the kernel; it is a registry magnitude", name)
		}
	}
	for _, o := range k.Provenance() {
		if o.Set == KernelAssumptionSet {
			t.Errorf("a deployment stating every engine setting reported the kernel "+
				"assumption %s", o.Name)
		}
	}

	// Leave the three settings unstated and each becomes a disclosed assumption, counted by
	// Evidence among the assumptions and in the total.
	in := fixtureInputs(t, memoryFixture)
	e := &in.Deployment.Pools[0].Engine
	e.MaxNumBatchedTokens, e.BlockSize, e.CUDAGraphMode = 0, 0, ""
	open, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	_, totalStated, _ := k.Evidence()
	_, totalOpen, assumed := open.Evidence()
	for _, name := range []string{"max_num_batched_tokens", "block_size", "cudagraph_mode"} {
		if !hasAssumption(open, name) {
			t.Errorf("an unstated %s is not disclosed in Provenance", name)
		}
		found := false
		for _, a := range assumed {
			found = found || a == name
		}
		if !found {
			t.Errorf("Evidence does not list %s among the assumptions", name)
		}
	}
	if totalOpen != totalStated+3 {
		t.Errorf("Evidence counts %d entries with three settings unstated against %d "+
			"stated; each assumption is one entry", totalOpen, totalStated)
	}
}

// An unstated token budget is vLLM v0.31.0's own default for the part in the API-server
// context (vllm/engine/arg_utils.py:2858-2887), which the kernel can compute because it holds
// the chip: the activation scratch it yields equals stating that figure.
func TestAnUnstatedTokenBudgetIsTheEnginesDefaultForThePart(t *testing.T) {
	want := map[string]int{
		"a100-80": 2048, "a100-sxm": 2048, "l40s": 2048,
		"h100": 8192, "h200": 8192,
		"b200": 16384, "b300": 16384, "gb200-nvl72": 16384, "gb300": 16384,
	}
	for name, budget := range want {
		chip, err := schemas.LoadChip(filepath.Join(catalogRoot, "hardware", name+".yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if got := defaultMaxNumBatchedTokens(*chip); got != budget {
			t.Errorf("%s (%.1f GiB): default budget %d, want %d", name, chip.MemoryGiB, got,
				budget)
		}
	}
	in := fixtureInputs(t, memoryFixture)
	in.Deployment.Pools[0].Engine.MaxNumBatchedTokens = 0
	unstated, err := New(in)
	if err != nil {
		t.Fatal(err)
	}
	stated := fixture(t, memoryFixture) // states 8,192, h200's default
	if unstated.FixedBytes().ActivationPeak != stated.FixedBytes().ActivationPeak {
		t.Error("an unstated budget on h200 did not size scratch as 8,192 tokens")
	}
}

// Every magnitude is required. A scenario that does not list cost-model-memory is refused,
// naming the coefficient, rather than priced with zero occupancy for the terms it lacks --
// which would overstate how many sequences fit by gigabytes per rank.
func TestAScenarioWithoutTheMemorySetIsRefused(t *testing.T) {
	in := fixtureInputs(t, memoryFixture)
	var kept []*coefficient.Set
	for _, s := range in.Coefficients {
		if s.Name != "cost-model-memory" {
			kept = append(kept, s)
		}
	}
	if len(kept) == len(in.Coefficients) {
		t.Fatal("the fixture does not list cost-model-memory")
	}
	in.Coefficients = kept
	in.Scenario.Coefficients = in.Scenario.Coefficients[:len(in.Scenario.Coefficients)-1]
	_, err := New(in)
	if err == nil || !strings.Contains(err.Error(), "cost-model-memory") {
		t.Errorf("New without the memory set returned %v; want a refusal naming it", err)
	}
}

func hasAssumption(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name && o.Set == KernelAssumptionSet &&
			o.Method == string(vocab.MethodAssumed) {
			return true
		}
	}
	return false
}

// AN MLA TOKEN IS ONE LATENT VECTOR, end to end. deepseek-v3 caches kv_lora_rank +
// qk_rope_head_dim = 576 elements per token per layer, over 61 layers, at one byte under an
// fp8 cache, with no value tensor (vllm/v1/kv_cache_interface.py:674-675 at v0.31.0) and no
// tensor-parallel reduction of its single head. One 16-token page is therefore
// 16 x 61 x 576 bytes on every rank, at tp=8 as at tp=1.
func TestAnMLATokenCachesOneLatentVector(t *testing.T) {
	for _, tp := range []int{1, 8} {
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Parallel.TP = tp
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := k.SequenceVariableBytes(16), int64(16*61*576); got != want {
			t.Errorf("tp=%d: a 16-token page holds %d bytes, want %d (one 576-wide latent "+
				"vector per token per layer, fp8)", tp, got, want)
		}
	}
}

// EPLB'S REPLICAS ARE COUNTED ONCE. A rank holds its share of the model's experts plus its
// share of the redundant ones, and that physical total is what FixedBytes().Total() must
// report: enabling EPLB with R redundant experts may raise Total by exactly the replicas'
// bytes, which is what EPLBRedundant reports. An earlier form reported the replicas in
// EPLBRedundant AND inside Weights, so Total rose by twice that.
//
// minimax-m2.5 at EP 8 holds 256 experts as 32 per rank; 8 redundant ones make it 33, one
// expert per layer more, which is EPLBRedundant's figure too.
func TestEPLBReplicasAreCountedOnce(t *testing.T) {
	build := func(redundant int) kernel.MemoryBreakdown {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
		if redundant > 0 {
			in.Deployment.Pools[0].Engine.EPLB = &deployment.EPLB{
				Enabled: true, NumRedundantExperts: redundant,
			}
		}
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.FixedBytes()
	}
	off, on := build(0), build(8)
	if on.EPLBRedundant <= 0 {
		t.Fatalf("EPLB with 8 redundant experts reported %d replica bytes", on.EPLBRedundant)
	}
	if grew := on.Total() - off.Total(); grew != on.EPLBRedundant {
		t.Errorf("enabling 8 redundant experts raised Total by %d bytes, want the replicas' "+
			"%d: a replica is held once", grew, on.EPLBRedundant)
	}
	if on.Weights != off.Weights {
		t.Errorf("enabling EPLB moved Weights from %d to %d; the replicas belong in "+
			"EPLBRedundant", off.Weights, on.Weights)
	}
}

// AN UNSTATED RECURRENT CACHE MODE IS WHAT THE ENGINE RUNS. The config default is "none"
// (vllm/config/cache.py:190 at v0.31.0), but vLLM sets a hybrid model's mode to "align"
// whenever prefix caching is on (vllm/model_executor/models/config.py:640-642), and prefix
// caching is on by default (cache.py:142). So on Nemotron-3-Ultra an unstated mode must hold
// what "align" holds, and "none" only once prefix caching is turned off.
func TestAnUnstatedRecurrentCacheModeIsTheEnginesEffectiveDefault(t *testing.T) {
	build := func(mode string, prefixCaching *bool) int64 {
		t.Helper()
		in := fixtureInputs(t, "nemotron3-ultra-h100-agg.yaml")
		e := &in.Deployment.Pools[0].Engine
		e.MambaCacheMode, e.EnablePrefixCaching = mode, prefixCaching
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.SequenceFixedBytes()
	}
	off := false
	align, none := build("align", nil), build("none", nil)
	if align <= none {
		t.Fatalf("align holds %d bytes per sequence against none's %d; the comparison needs "+
			"them to differ", align, none)
	}
	if got := build("", nil); got != align {
		t.Errorf("an unstated mode with prefix caching on holds %d bytes, want align's %d",
			got, align)
	}
	if got := build("", &off); got != none {
		t.Errorf("an unstated mode with prefix caching off holds %d bytes, want none's %d",
			got, none)
	}
}

// COLLECTIVES MOVE ACTIVATIONS, AND ACTIVATIONS ARE BF16. Serving a bf16 checkpoint fp8
// halves its weight bytes and must leave every collective exactly where it was: a quantized
// linear returns its input's dtype (vllm/model_executor/kernels/linear/scaled_mm/
// cutlass.py:147,153 at v0.31.0), so the hidden state a tensor-parallel all-reduce carries is
// bf16 either way. An earlier form sized it at the served width and halved it under fp8.
func TestServingFP8LeavesTheCollectivesWhereTheyWere(t *testing.T) {
	qwen, err := schemas.LoadModelGraph(
		filepath.Join(catalogRoot, "models", "qwen3-30b-a3b", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	build := func(quant string) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		in.Model, in.Scenario.Model = qwen, qwen.Name
		in.Deployment.Pools[0].Engine.Quantization = quant
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	bf16, fp8 := build(""), build("fp8")
	for shape, b := range batchShapes() {
		x, y := bf16.StepTime(b).PerResource, fp8.StepTime(b).PerResource
		if x[kernel.ResourceNVLink] != y[kernel.ResourceNVLink] {
			t.Errorf("%s: collectives cost %v served bf16 and %v served fp8", shape,
				x[kernel.ResourceNVLink], y[kernel.ResourceNVLink])
		}
	}
}

// A DATA-PARALLEL MoE WITH EXPERT PARALLELISM OFF IS ONE SHARED MoE, NOT dp REPLICAS.
// vLLM v0.31.0 shards every expert over the dp x tp ranks (flatten_tp_across_dp_and_pcp,
// vllm/model_executor/layers/fused_moe/config.py:1090-1098) and gathers every rank's tokens
// to it with an all-gather/reduce-scatter dispatch, whatever all2all backend is named
// ("Detected DP deployment with no --enable-expert-parallel. Falling back to
// AllGather+ReduceScatter", all2all_utils.py:202-214). Four consequences, each asserted on
// minimax-m2.5 at tp=8 with dp 1, 2 and 4:
//
//   - a rank's expert weights divide by dp: the expert part of Weights halves per doubling;
//   - per-rank routed compute does not move: dp times the tokens, each on a 1/dp slice;
//   - a dispatch now runs over the MoE group, so inflating the all-to-all floors moves a
//     step at dp > 1 and not at dp = 1, where no dispatch exists;
//   - a routed backend named with expert parallelism off prices as the fallback it is.
func TestADataParallelMoEWithoutExpertParallelismIsShared(t *testing.T) {
	build := func(dp int, edit func(*Inputs)) *Kernel {
		t.Helper()
		in := fixtureInputs(t, "minimax-m25-h200-tp8.yaml")
		pool := &in.Deployment.Pools[0]
		pool.Parallel.EnableExpertParallel = false
		pool.Parallel.DP = dp
		pool.Nodes, in.Scenario.Cluster.Nodes = dp, dp
		if dp > 1 {
			in.Scenario.Cluster.Fabric = "ib-400g"
			f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", "ib-400g.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			in.Fabric = f
		}
		if edit != nil {
			edit(&in)
		}
		k, err := New(in)
		if err != nil {
			t.Fatalf("dp=%d: %v", dp, err)
		}
		return k
	}
	w1, w2, w4 := build(1, nil).FixedBytes().Weights, build(2, nil).FixedBytes().Weights,
		build(4, nil).FixedBytes().Weights
	if d12, d24 := float64(w1-w2), float64(w2-w4); d12 <= 0 || math.Abs(d12/d24-2) > 1e-6 {
		t.Errorf("expert weights per rank did not halve per doubling of dp: Weights %d, %d, "+
			"%d at dp 1, 2, 4", w1, w2, w4)
	}

	// Per-rank routed compute: the SM term of a large decode is the same at dp 1 and 2,
	// since the dense work per rank is unchanged and the routed work is dp tokens on 1/dp.
	big := decodeBatch(256, 1, 4096)
	if a, b := build(1, nil).StepTime(big).PerResource[kernel.ResourceSM],
		build(2, nil).StepTime(big).PerResource[kernel.ResourceSM]; a != b {
		t.Errorf("per-rank compute moved from %v at dp=1 to %v at dp=2; the gathered tokens "+
			"each reach a half-size slice", a, b)
	}

	inflate := func(in *Inputs) { in.Coefficients = scaleFloors(in.Coefficients, "alltoall", 10) }
	decode := decodeBatch(32, 1, 4096)
	if a, b := build(1, nil).StepTime(decode).NoOverlap,
		build(1, inflate).StepTime(decode).NoOverlap; a != b {
		t.Errorf("at dp=1 inflating the all-to-all floors moved the step from %v to %v; a "+
			"tensor-parallel MoE dispatches nothing", a, b)
	}
	if a, b := build(2, nil).StepTime(decode).NoOverlap,
		build(2, inflate).StepTime(decode).NoOverlap; b <= a {
		t.Errorf("at dp=2 inflating the all-to-all floors left the step at %v; the MoE's "+
			"dispatch runs over the data-parallel ranks", b)
	}

	named := build(2, func(in *Inputs) {
		in.Deployment.Pools[0].Engine.All2AllBackend = "deepep_low_latency"
	})
	if a, b := build(2, nil).StepTime(decode).NoOverlap, named.StepTime(decode).NoOverlap; a != b {
		t.Errorf("naming deepep_low_latency with expert parallelism off priced %v against %v "+
			"for the default; the engine falls back to all-gather/reduce-scatter", b, a)
	}
}

// PAST THE CAPTURE CEILING NO GRAPH RUNS. vLLM v0.31.0 captures token counts up to
// max_cudagraph_capture_size -- unstated, min(max_num_seqs x decode_query_len x 2, 512) off
// data-center Blackwell (vllm/config/vllm.py:2442-2460) -- and dispatches a larger batch with
// no graph whatever the mode (vllm/v1/worker/gpu/cudagraph_utils.py:499-529). The fixture
// states max_num_seqs 256, so its ceiling is 512 tokens: a 512-token prefill runs piecewise,
// a 513-token one runs eager, and under every capturing mode the two sides of the ceiling
// must price like PIECEWISE and NONE respectively.
func TestABatchPastTheCaptureCeilingRunsWithNoGraph(t *testing.T) {
	host := func(mode string, tokens int) time.Duration {
		t.Helper()
		in := fixtureInputs(t, memoryFixture)
		in.Deployment.Pools[0].Engine.CUDAGraphMode = mode
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.StepTime(decodeBatch(1, tokens, tokens)).PerResource[kernel.ResourceHost]
	}
	for _, mode := range []string{"PIECEWISE", "FULL_AND_PIECEWISE"} {
		if a, b := host(mode, 512), host("PIECEWISE", 512); a != b {
			t.Errorf("%s: a 512-token prefill priced %v of host time, want PIECEWISE's %v",
				mode, a, b)
		}
		if a, b := host(mode, 513), host("NONE", 513); a != b {
			t.Errorf("%s: a 513-token prefill priced %v of host time, want NONE's %v; it is "+
				"past every captured size", mode, a, b)
		}
	}
	if host("PIECEWISE", 512) == host("NONE", 512) {
		t.Fatal("PIECEWISE and NONE price alike, so this test cannot discriminate")
	}
}

// A SPARSE-MLA LAYER IS TWO SPLIT POINTS. PIECEWISE splits at every op in
// CompilationConfig._attention_ops (vllm/config/compilation.py:764-782 at v0.31.0), and a DSA
// layer runs two of them: its indexer (vllm::sparse_attn_indexer) and its attention. The
// graph states the indexer as a second, cache-less attention node, so the segment count is
// the layer count, plus one per indexed sparse-MLA layer, plus one -- on glm5, every layer
// indexed; on deepseek-v3, none.
func TestASparseMLALayersIndexerIsAPiecewiseSplitPoint(t *testing.T) {
	for _, c := range []struct {
		fixture string
		indexed bool
	}{
		{dcpSparseFixture, true},
		{dcpMLAFixture, false},
	} {
		k := fixture(t, c.fixture)
		want := k.plan.TotalLayers + 1
		if c.indexed {
			want += k.plan.TotalLayers
		}
		if got := k.plan.PiecewiseSegments(); got != want {
			t.Errorf("%s: %d piecewise segments, want %d", c.fixture, got, want)
		}
	}
}

// A PACKED SPARSE-MLA CACHE IS ITS OWN SIZE. fp8_ds_mla stores 656 bytes a token per layer
// and nvfp4_ds_mla 352 (vllm/model_executor/layers/attention/mla_attention.py:1361-1363 at
// v0.31.0), where the per-element formula gives 576 at one byte. Stated on glm5, a 16-token
// page must hold exactly that times its layers; a plain "fp8" keeps the per-element size,
// because the backends v0.31.0 prefers for an fp8 sparse cache do not repack it.
func TestAPackedSparseMLACacheIsItsOwnSize(t *testing.T) {
	page := func(cache string) int64 {
		t.Helper()
		in := fixtureInputs(t, dcpSparseFixture)
		in.Deployment.Pools[0].Engine.CacheDType = cache
		k, err := New(in)
		if err != nil {
			t.Fatal(err)
		}
		return k.SequenceVariableBytes(16)
	}
	layers := int64(fixture(t, dcpSparseFixture).kvLayers)
	for _, c := range []struct {
		cache string
		cell  int64
	}{{"fp8_ds_mla", 656}, {"nvfp4_ds_mla", 352}, {"fp8", 576}} {
		if got, want := page(c.cache), 16*c.cell*layers; got != want {
			t.Errorf("%s: a 16-token page holds %d bytes, want %d (%d a token per layer)",
				c.cache, got, want, c.cell)
		}
	}
}
