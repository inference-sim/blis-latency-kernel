package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/vocab"
)

// dep8 is DeepSeek-V4.1-Flash as one DEP8 engine on the fixture's node: tp 1, dp 8, expert
// parallelism on, so the MoE dispatch runs over an 8-rank expert group inside the node.
// entries are extra moe_a2a_* coefficients for the fixture's part, as {name: value}.
func dep8(t *testing.T, backend, mode string, mnbt int, entries map[string]float64) *Kernel {
	t.Helper()
	in := fixtureInputs(t, v41)
	pl := &in.Deployment.Pools[0].Parallel
	pl.TP, pl.DP, pl.EnableExpertParallel = 1, 8, true
	e := &in.Deployment.Pools[0].Engine
	e.All2AllBackend, e.CUDAGraphMode, e.MaxNumBatchedTokens = backend, mode, mnbt
	set := &coefficient.Set{Kind: "CoefficientSet", Name: "test-deepep"}
	for name, v := range entries {
		units := "bytes_per_us"
		if strings.HasPrefix(name, "moe_a2a_floor_") {
			units = "us_per_transfer"
		}
		set.Coefficients = append(set.Coefficients, coefficient.Entry{
			Name: name, Value: v, Units: vocab.Unit(units), Method: "measured", Rationale: "test",
			Scope: coefficient.Scope{Hardware: []string{"h200"}}})
	}
	if len(entries) > 0 {
		in.Coefficients = append(append([]*coefficient.Set{}, in.Coefficients...), set)
	}
	return build(t, in)
}

// moeLayers counts the MoE layers a step runs, the drafter's included while it speculates.
func moeLayers(k *Kernel) int {
	n := 0
	layers := k.plan.Layers
	if d := k.plan.Draft; d != nil && k.speculating {
		layers = append(append(layers[:len(layers):len(layers)], d.Layers...), d.Context...)
	}
	for _, l := range layers {
		if l.TopK > 0 {
			n += l.Count
		}
	}
	return n
}

func priced(variant string, floorUS, rate float64) map[string]float64 {
	return map[string]float64{
		"moe_a2a_floor_" + variant + "_8rank_h200": floorUS,
		"moe_a2a_rate_" + variant + "_8rank_h200":  rate,
	}
}

// A measured DeepEP backend prices each MoE layer's dispatch and combine as its own floor
// plus bytes over its own rate, not as NCCL's all-to-all (blis-latency-kernel#57): raising
// the floor by 100 us raises a decode step by 100 us per MoE layer.
func TestDeepEPDispatchIsPricedFromItsOwnMeasurement(t *testing.T) {
	b := decodeBatch(32, 1, 20000)
	lo := dep8(t, "deepep_low_latency", "FULL_AND_PIECEWISE", 256,
		priced("deepep_low_latency", 100, 1e5))
	hi := dep8(t, "deepep_low_latency", "FULL_AND_PIECEWISE", 256,
		priced("deepep_low_latency", 200, 1e5))
	want := time.Duration(moeLayers(lo)) * 100 * time.Microsecond
	got := hi.StepTime(b).NoOverlap - lo.StepTime(b).NoOverlap
	if d := got - want; d < -time.Microsecond || d > time.Microsecond {
		t.Errorf("100 us more floor moved the step by %v; %d MoE layers want %v",
			got, moeLayers(lo), want)
	}
}

// Without a measurement for the part, the routed all-to-all stands and that is recorded.
func TestAnUnmeasuredDeepEPBackendIsRecorded(t *testing.T) {
	k := dep8(t, "deepep_high_throughput", "", 8192, nil)
	if k.deepEP != nil {
		t.Fatal("priced from a DeepEP measurement the scenario does not carry")
	}
	for _, o := range k.Provenance() {
		if o.Name == "moe_a2a_deepep_high_throughput" {
			return
		}
	}
	t.Error("the fallback to the generic routed all-to-all is not in Provenance")
}

// DeepEP v2 runs its capturable path whenever enforce_eager is off and its expanding eager
// path otherwise (all2all_utils.py:286-298), so the two take different measurements.
func TestDeepEPV2TakesItsPathFromTheGraphMode(t *testing.T) {
	entries := priced("deepep_v2_graph", 50, 1e5)
	for n, v := range priced("deepep_v2_eager", 500, 1e5) {
		entries[n] = v
	}
	for mode, want := range map[string]string{
		"FULL_AND_PIECEWISE": "deepep_v2_graph", "NONE": "deepep_v2_eager"} {
		k := dep8(t, "deepep_v2", mode, 2048, entries)
		if k.deepEP == nil || k.deepEP.variant != want {
			t.Errorf("cudagraph_mode %s: priced %v, want %s", mode, k.deepEP, want)
		}
	}
}

// DeepEP's buffers are made after vLLM's memory snapshot and so come out of the KV cache:
// HT holds VLLM_DEEPEP_BUFFER_SIZE_MB of NVLink buffer, LL that plus an RDMA buffer sized by
// max_num_batched_tokens, and the all-gather dispatch none.
func TestDeepEPBuffersComeOutOfTheKVCache(t *testing.T) {
	const mib = 1 << 20
	ag := dep8(t, "allgather_reducescatter", "", 256, nil).FixedBytes()
	ht := dep8(t, "deepep_high_throughput", "", 256, nil).FixedBytes()
	ll := dep8(t, "deepep_low_latency", "", 256, nil).FixedBytes()
	ll2 := dep8(t, "deepep_low_latency", "", 512, nil).FixedBytes()
	if ag.CommBuffers != 0 {
		t.Errorf("all-gather dispatch holds %d B of DeepEP buffer", ag.CommBuffers)
	}
	if ht.CommBuffers != 1024*mib {
		t.Errorf("HT holds %d B; want the 1024 MiB NVLink buffer", ht.CommBuffers)
	}
	rdma := lowLatencyRDMABytes(256, 5120, 8, 384)
	if ll.CommBuffers != 1024*mib+rdma {
		t.Errorf("LL holds %d B; want 1024 MiB + %d B RDMA", ll.CommBuffers, rdma)
	}
	if ll2.CommBuffers <= ll.CommBuffers {
		t.Error("LL's RDMA buffer does not grow with max_num_batched_tokens")
	}
}

// DeepEP's own sizing for DeepSeek-V4.1-Flash's shape (hidden 5120, 384 experts, top-6,
// EP8), from LowLatencyLayout and ElasticBuffer::calculate_buffer_size at d4f41e4e93:
// 4 x E x T x (2H + 4H/128) for LL once the experts dominate, and 8 x T x 10,336 B
// for v2, 2 MiB aligned.
func TestDeepEPBufferSizesFollowDeepEP(t *testing.T) {
	ll := lowLatencyRDMABytes(256, 5120, 8, 384)
	want := int64(4*384*256*(2*5120+4*40) + 2*1536)
	want = (want + 128) / 128 * 128
	if ll != want {
		t.Errorf("LL RDMA bytes %d, want %d", ll, want)
	}
	v2 := elasticBufferBytes(8192, 5120, 6, 8)
	if want := int64((8*8192*10336 + (2 << 20) - 1) / (2 << 20) * (2 << 20)); v2 != want {
		t.Errorf("v2 buffer %d, want %d", v2, want)
	}
}

// allgather_reducescatter gathers every DP rank's tokens to every rank and reduce-scatters
// them back (AgRsAll2AllManager, all2all.py:84-157), so at DEP8 a rank's dispatch moves the
// whole group's rows through an NCCL all-gather and reduce-scatter, not its own rows once.
func TestAllGatherDispatchMovesTheWholeGroupsTokens(t *testing.T) {
	k := dep8(t, "allgather_reducescatter", "", 8192, nil)
	const tokens = 4096
	rows := float64(tokens) * 5120 * 2
	ag := collKey{Op: model.OpAllGather, Group: price.GroupExpert}
	rs := collKey{Op: model.OpReduceScatter, Group: price.GroupExpert}
	perLayer := k.collectiveSeconds(ag, 8*rows) + k.collectiveSeconds(rs, 8*rows)
	got := k.StepTime(prefillOf(tokens, 0)).PerResource[kernel.ResourceNVLink]
	// The target's MoE layers: a prefill step runs no drafter forward over its tokens.
	layers := 0
	for _, l := range k.plan.Layers {
		if l.TopK > 0 {
			layers += l.Count
		}
	}
	want := time.Duration(float64(layers) * perLayer * float64(time.Second))
	// Within 0.5%: the step's other NVLink terms are a fraction of a millisecond.
	if r := float64(got) / float64(want); r < 1 || r > 1.005 {
		t.Errorf("a %d-token DEP8 prefill charges %v of NVLink; %d MoE layers of an 8-rank "+
			"all-gather and reduce-scatter of the group's %d rows want %v",
			tokens, got, layers, 8*tokens, want)
	}
}

// vLLM sizes DeepEP's low-latency RDMA buffer by max_num_batched_tokens and nothing else
// (FusedMoEConfig.max_num_tokens, fused_moe/layer.py:272,359; all2all_utils.py:244-246), so
// a DEP8 DeepSeek-V4.1 engine at 8,192 tokens asks for ~122 GiB per rank and fails at weight
// loading. BLIS refuses it, names the cause, and accepts the same engine at 256.
func TestALowLatencyBufferThatCannotBeAllocatedIsRefused(t *testing.T) {
	in := fixtureInputs(t, v41)
	pl := &in.Deployment.Pools[0].Parallel
	pl.TP, pl.DP, pl.EnableExpertParallel = 1, 8, true
	e := &in.Deployment.Pools[0].Engine
	e.All2AllBackend, e.MaxNumBatchedTokens = "deepep_low_latency", 8192
	_, err := New(in)
	if err == nil || !strings.Contains(err.Error(), "DeepEP buffers") {
		t.Fatalf("LL at max_num_batched_tokens 8192 built (err %v); vLLM fails allocating "+
			"its buffer", err)
	}
	e.MaxNumBatchedTokens = 256
	build(t, in)
}

// Unstated, max_num_batched_tokens is 256 under a batched data-parallel MoE
// (arg_utils.py:3046-3051), and that is what sizes the LL buffer. A layout that runs no
// dispatch -- a tp8 dp1 decode pool -- builds no DeepEP buffer at all.
func TestTheLowLatencyBufferFollowsTheEnginesTokenBudget(t *testing.T) {
	ll := dep8(t, "deepep_low_latency", "", 0, nil)
	if want := int64(1024<<20) + lowLatencyRDMABytes(256, 5120, 8, 384); ll.FixedBytes().CommBuffers != want {
		t.Errorf("unstated budget: LL holds %d B, want %d (256 tokens)",
			ll.FixedBytes().CommBuffers, want)
	}
	in := fixtureInputs(t, v41)
	in.Deployment.Pools[0].Engine.All2AllBackend = "deepep_low_latency"
	if b := build(t, in).FixedBytes().CommBuffers; b != 0 {
		t.Errorf("tp8 dp1 dispatches nothing yet holds %d B of DeepEP buffer", b)
	}
}
