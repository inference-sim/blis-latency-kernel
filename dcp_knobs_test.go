package latencykernel

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules/v0_29"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/vocab"

	schemas "github.com/inference-sim/blis-schemas"
)

// The three decode-context-parallel engine knobs blis-schemas v0.2.2 made expressible --
// dcp_comm_backend, dcp_q_replicate, cp_kv_cache_interleave_size -- priced the way vLLM
// v0.31.0 runs them. resolve.ResolveDecodeContext and dcpDecodeCollectives carry the
// citations; these pin the behaviour.

// dcpVariant is a committed fixture with one pool's DCP, PCP, TP and node count set, and
// any engine edits applied, laid out as the engine needs it (pcpInputs adds the nodes and
// fabric a PCP layout requires).
func dcpVariant(t *testing.T, fixture string, tp, pcp, dcp int, edit func(*Inputs)) (*Kernel, error) {
	t.Helper()
	in := pcpInputs(t, fixture, 1)
	pool := &in.Deployment.Pools[0]
	pool.Parallel.TP, pool.Parallel.PCP, pool.Parallel.DCP = tp, pcp, dcp
	nodes := max((tp*max(pcp, 1)+7)/8, 1)
	pool.Nodes, in.Scenario.Cluster.Nodes = nodes, nodes
	if edit != nil {
		edit(&in)
	}
	return New(in)
}

func mustDCPVariant(t *testing.T, fixture string, tp, pcp, dcp int, edit func(*Inputs)) *Kernel {
	t.Helper()
	k, err := dcpVariant(t, fixture, tp, pcp, dcp, edit)
	if err != nil {
		t.Fatalf("%s tp=%d pcp=%d dcp=%d: %v", fixture, tp, pcp, dcp, err)
	}
	return k
}

func withBackend(b string) func(*Inputs) {
	return func(in *Inputs) { in.Deployment.Pools[0].Engine.DCPCommBackend = b }
}

// scaleFloors returns a copy of the coefficient sets with every collective floor of one
// primitive multiplied by factor. It is how a test asks which collectives a layout
// launches without reading the kernel's internals: inflate one primitive's floors and see
// which step times move.
func scaleFloors(sets []*coefficient.Set, primitive string, factor float64) []*coefficient.Set {
	prefix := "collective_floor_" + primitive + "_"
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if strings.HasPrefix(c.Coefficients[j].Name, prefix) {
				c.Coefficients[j].Value *= factor
			}
		}
		out[i] = &c
	}
	return out
}

// WHICH COLLECTIVES EACH LAYOUT LAUNCHES, asked of the step rather than of the code.
//
// Inflating one primitive's floors tenfold moves a decode step exactly when that layout
// launches the primitive. On deepseek-v3 and glm5, whose graphs emit only all-reduces (and
// an all-to-all only with expert parallelism, which these layouts leave off), the
// decode-context combine is the only source of a reduce-scatter or an all-to-all. The
// PCP row is glm5 because PCP with DCP runs only on DSA sparse-MLA layers
// (resolveContextParallel); deepseek-v3's plain MLA refuses it.
//
//	ag_rs, pcp off   reduce-scatter                  (cp_lse_ag_out_rs, dcp.py:493)
//	a2a,   pcp off   all-to-all, and no reduce-scatter (dcp_a2a_lse_reduce, :939-1010)
//	ag_rs, pcp on    neither: it all-reduces instead  (cp_lse_ag_out_ar, :526; :1525-1531)
//
// A step whose combine launches the primitive must rise; one that does not must not move
// at all. Decode only -- the combine is a decode-row collective -- and a prefill step is
// checked not to move under either, since no combine runs on it.
func TestEachDCPLayoutLaunchesTheCombineTheEngineRuns(t *testing.T) {
	decode := decodeBatch(8, 1, 8192)
	prefill := decodeBatch(1, 2048, 2048)
	for _, c := range []struct {
		name               string
		fixture            string
		tp, pcp, dcp       int
		backend            string
		reduceScatter, a2a bool
	}{
		{"ag_rs without pcp", dcpMLAFixture, 8, 1, 8, "ag_rs", true, false},
		{"a2a without pcp", dcpMLAFixture, 8, 1, 8, "a2a", false, true},
		{"ag_rs with pcp", dcpSparseFixture, 2, 4, 4, "ag_rs", false, false},
	} {
		for _, prim := range []struct {
			name     string
			launched bool
		}{
			{"reduce_scatter", c.reduceScatter},
			{"alltoall", c.a2a},
		} {
			base := mustDCPVariant(t, c.fixture, c.tp, c.pcp, c.dcp, withBackend(c.backend))
			inflated := mustDCPVariant(t, c.fixture, c.tp, c.pcp, c.dcp, func(in *Inputs) {
				withBackend(c.backend)(in)
				in.Coefficients = scaleFloors(in.Coefficients, prim.name, 10)
			})
			before, after := base.StepTime(decode).NoOverlap, inflated.StepTime(decode).NoOverlap
			switch {
			case prim.launched && after <= before:
				t.Errorf("%s: inflating %s floors left the decode step at %v (was %v); "+
					"this combine launches a %s", c.name, prim.name, after, before, prim.name)
			case !prim.launched && after != before:
				t.Errorf("%s: inflating %s floors moved the decode step from %v to %v; "+
					"this combine launches no %s", c.name, prim.name, before, after, prim.name)
			}
			if a, b := base.StepTime(prefill).NoOverlap,
				inflated.StepTime(prefill).NoOverlap; a != b {
				t.Errorf("%s: inflating %s floors moved a prefill step from %v to %v; no "+
					"decode-context combine runs on a step without decode rows",
					c.name, prim.name, a, b)
			}
		}
	}
}

// AND WHAT IT DOES LAUNCH UNDER PCP: an LSE all-gather and an all-reduce over the
// decode-context group (cp_lse_ag_out_ar, vllm/v1/attention/ops/dcp.py:503-526, chosen at
// :1525-1531). Asserted positively, since the row above only shows what it does not launch.
// At tp=2, pcp=4, dcp=4 the decode-context group is the only 4-rank group a decode step
// touches -- the graph's own all-reduce is 2-rank, and the PCP gather runs on prefill rows
// only -- so inflating either 4-rank primitive must move a decode step.
func TestUnderPCPTheCombineIsAnLSEGatherAndAnAllReduce(t *testing.T) {
	decode := decodeBatch(8, 1, 8192)
	base := mustDCPVariant(t, dcpSparseFixture, 2, 4, 4, nil).StepTime(decode).NoOverlap
	for _, prim := range []string{"all_reduce_fp16_4rank", "all_gather_fp16_4rank"} {
		inflated := mustDCPVariant(t, dcpSparseFixture, 2, 4, 4, func(in *Inputs) {
			in.Coefficients = scaleFloors(in.Coefficients, prim, 10)
		}).StepTime(decode).NoOverlap
		if inflated <= base {
			t.Errorf("inflating %s floors left the decode step at %v (was %v); under PCP the "+
				"combine launches it over the decode-context group", prim, inflated, base)
		}
	}
}

// THE QUERY GATHER UNDER PCP. With prefill-context parallelism on, a DSA layer -- the only
// kind that runs PCP with DCP -- gathers its query over the tensor-parallel group only when
// the decode-context group spans the whole tp x pcp block, and not at all when it spans the
// PCP axis alone (vllm/models/deepseek_v32/attention.py:560-563 at v0.31.0).
//
// Asked the same way: at tp=2 the tensor-parallel all-gather is the 2-rank one, and the
// decode-context group is 4 or 8 ranks wide, so inflating the 2-rank all-gather floor moves
// the decode step exactly when that gather runs.
func TestUnderPCPTheQueryIsGatheredOverTPOnlyWhenDCPSpansTheBlock(t *testing.T) {
	decode := decodeBatch(8, 1, 8192)
	for _, c := range []struct {
		dcp      int
		gathered bool
	}{
		{4, false}, // dcp == pcp: the PCP axis alone
		{8, true},  // dcp == tp*pcp: the whole block, after a tensor-parallel gather
	} {
		base := mustDCPVariant(t, dcpSparseFixture, 2, 4, c.dcp, nil)
		inflated := mustDCPVariant(t, dcpSparseFixture, 2, 4, c.dcp, func(in *Inputs) {
			in.Coefficients = scaleFloors(in.Coefficients, "all_gather_fp16_2rank", 10)
		})
		before, after := base.StepTime(decode).NoOverlap, inflated.StepTime(decode).NoOverlap
		if c.gathered && after <= before {
			t.Errorf("dcp=%d: inflating the 2-rank all-gather left the decode step at %v; "+
				"the query is gathered over the tensor-parallel group here", c.dcp, after)
		}
		if !c.gathered && after != before {
			t.Errorf("dcp=%d: inflating the 2-rank all-gather moved the decode step from "+
				"%v to %v; with dcp == pcp no tensor-parallel gather runs", c.dcp, before, after)
		}
	}
}

// The backend is a decode-only choice: switching it must leave a prefill step exactly where
// it was, and must change a decode step.
func TestTheDCPBackendChangesDecodeAndNothingElse(t *testing.T) {
	agrs := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("ag_rs"))
	a2a := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("a2a"))
	prefill := decodeBatch(1, 2048, 2048)
	if a, b := agrs.StepTime(prefill).NoOverlap, a2a.StepTime(prefill).NoOverlap; a != b {
		t.Errorf("the backend moved a prefill step: ag_rs %v, a2a %v", a, b)
	}
	if a, b := agrs.StepTime(decodeBatch(8, 1, 8192)).NoOverlap,
		a2a.StepTime(decodeBatch(8, 1, 8192)).NoOverlap; a == b {
		t.Errorf("the backend did not change a decode step (%v under both)", a)
	}
	if a, b := agrs.SequenceVariableBytes(8192), a2a.SequenceVariableBytes(8192); a != b {
		t.Errorf("the backend moved capacity: %d against %d", a, b)
	}
}

// An unstated backend is the stock default, ag_rs, and that is an ASSUMPTION the kernel
// discloses: a model's configuration hook may choose otherwise, and the kernel cannot see
// the model class. A stated one is the deployment's own fact and is not reported as an
// assumption.
func TestAnUnstatedDCPBackendIsTheStockDefaultAndIsDisclosed(t *testing.T) {
	unstated := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, nil)
	stated := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("ag_rs"))
	decode := decodeBatch(8, 1, 8192)
	if a, b := unstated.StepTime(decode).NoOverlap, stated.StepTime(decode).NoOverlap; a != b {
		t.Errorf("an unstated backend priced %v against %v for ag_rs stated", a, b)
	}
	disclosed := func(k *Kernel) bool {
		for _, o := range k.Provenance() {
			if o.Name == "dcp_comm_backend" && o.Set == KernelAssumptionSet &&
				o.Method == string(vocab.MethodAssumed) && strings.HasPrefix(o.Scope, "ag_rs") {
				return true
			}
		}
		return false
	}
	if !disclosed(unstated) {
		t.Error("an unstated backend resolved to ag_rs without a kernel assumption in " +
			"Provenance")
	}
	if disclosed(stated) {
		t.Error("a stated backend was reported as an assumption")
	}
	_, _, assumed := unstated.Evidence()
	found := false
	for _, a := range assumed {
		found = found || a == "dcp_comm_backend"
	}
	if !found {
		t.Errorf("Evidence does not list the backend default among the assumptions: %v",
			assumed)
	}
	// And with dcp at one the knob does nothing, so nothing is assumed about it.
	if disclosed(mustDCPVariant(t, dcpMLAFixture, 8, 1, 1, nil)) {
		t.Error("a deployment without DCP reported an assumption about its DCP backend")
	}
}

// The backend names a release accepts are the release's rules pack's to say. A pack that
// does not list a2a refuses it; the v0.29 pack lists both; a name no release has is refused
// even where DCP is off, because the engine types the setting as a closed literal and
// validates it at startup whatever the width (vllm/config/parallel.py:40, :371).
func TestTheDCPBackendIsGatedByTheReleaseAndByWhatTheKernelPrices(t *testing.T) {
	if _, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("a2a")); err != nil {
		t.Errorf("a2a, which the v0.29 pack lists, was refused: %v", err)
	}
	narrow := *v0_29.Pack()
	narrow.DCPCommBackends = map[string]bool{"ag_rs": true}
	_, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, func(in *Inputs) {
		withBackend("a2a")(in)
		in.Rules = &narrow
	})
	if err == nil {
		t.Error("a2a was priced under a rules pack that does not accept it")
	}
	for _, dcp := range []int{1, 8} {
		if _, err := dcpVariant(t, dcpMLAFixture, 8, 1, dcp,
			withBackend("all_gather_only")); err == nil {
			t.Errorf("dcp=%d: an unknown backend name was priced", dcp)
		}
	}
}

// A rules value with no list of the release's backend names cannot gate a stated backend,
// and the kernel says so rather than letting the check vanish: the stated name is still
// held to what the kernel prices, and Provenance records that the release was not consulted.
func TestAStatedBackendTheReleaseCouldNotCheckIsDisclosed(t *testing.T) {
	noList := *v0_29.Pack()
	noList.DCPCommBackends = nil
	k, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, func(in *Inputs) {
		withBackend("a2a")(in)
		in.Rules = &noList
	})
	if err != nil {
		t.Fatalf("a2a under a pack with no backend list was refused: %v", err)
	}
	if !hasAssumption(k, "dcp_comm_backend") {
		t.Error("a stated backend the release could not check left no trace in Provenance")
	}
	checked := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, withBackend("a2a"))
	if hasAssumption(checked, "dcp_comm_backend") {
		t.Error("a stated backend the release did check was reported as an assumption")
	}
}

// a2a is refused alongside PCP: "MRV2 PCP + DCP requires dcp_comm_backend='ag_rs'"
// (vllm/v1/worker/gpu/pcp_manager.py:188-194 at v0.31.0).
func TestA2AIsRefusedAlongsidePCP(t *testing.T) {
	if _, err := dcpVariant(t, dcpSparseFixture, 2, 4, 4, withBackend("a2a")); err == nil {
		t.Error("a2a with pcp=4, dcp=4 was priced; the engine refuses it")
	}
	if _, err := dcpVariant(t, dcpSparseFixture, 2, 4, 4, withBackend("ag_rs")); err != nil {
		t.Errorf("ag_rs with pcp=4, dcp=4 was refused: %v", err)
	}
}

// A replicated query projection is refused where it would take effect, and accepted where
// the engine ignores it.
//
// It takes effect only on a latent layer with dcp > 1 and pcp <= 1
// (vllm/model_executor/models/deepseek_v2.py:1072-1076 at v0.31.0); there it costs a
// projection built over tp/dcp ranks instead of tp (linear.py:632-659), which this kernel
// does not price, so it refuses rather than crediting the skipped gather alone. Everywhere
// else the request changes nothing and must change nothing here.
func TestAReplicatedQueryIsRefusedOnlyWhereItWouldTakeEffect(t *testing.T) {
	yes, no := true, false
	qrep := func(v *bool) func(*Inputs) {
		return func(in *Inputs) { in.Deployment.Pools[0].Engine.DCPQReplicate = v }
	}
	if _, err := dcpVariant(t, dcpMLAFixture, 8, 1, 8, qrep(&yes)); err == nil {
		t.Error("dcp_q_replicate on a latent model with dcp=8 was priced; its replicated " +
			"projection is not")
	}
	plain := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, nil)
	decode := decodeBatch(8, 1, 8192)
	for _, c := range []struct {
		name         string
		fixture      string
		tp, pcp, dcp int
		v            *bool
	}{
		{"stated false", dcpMLAFixture, 8, 1, 8, &no},
		{"dcp off", dcpMLAFixture, 8, 1, 1, &yes},
		{"pcp on", dcpSparseFixture, 2, 4, 4, &yes},
		{"full attention", "minimax-m25-h200-tp8.yaml", 8, 1, 8, &yes},
	} {
		k, err := dcpVariant(t, c.fixture, c.tp, c.pcp, c.dcp, qrep(c.v))
		if err != nil {
			t.Errorf("%s: refused, but the engine ignores the request here: %v", c.name, err)
			continue
		}
		if c.name == "stated false" && k.StepTime(decode).NoOverlap != plain.StepTime(decode).NoOverlap {
			t.Error("stating dcp_q_replicate false priced differently from leaving it unset")
		}
	}
}

// THE STRIPE. cp_kv_cache_interleave_size sets how many consecutive tokens a rank holds,
// and the slowest rank binds, so a coarse stripe makes a short context's read land on one
// rank (vllm/v1/attention/backends/utils.py:1143-1156 at v0.31.0).
//
// At a 100-token context over 8 ranks the slowest rank holds 13 tokens at a stripe of 1 and
// 64 at a stripe of 64, so a decode read must cost more; at a context that is a multiple of
// stripe x dcp every rank holds the same share under both, so the two must agree exactly.
func TestTheInterleaveSetsTheSlowestRanksShareOfTheRead(t *testing.T) {
	interleave := func(n int) func(*Inputs) {
		return func(in *Inputs) {
			in.Deployment.Pools[0].Engine.CPKVCacheInterleaveSize = n
			in.Deployment.Pools[0].Engine.BlockSize = 64
		}
	}
	fine := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, interleave(1))
	coarse := mustDCPVariant(t, dcpMLAFixture, 8, 1, 8, interleave(64))
	hbm := func(k *Kernel, ctx int) float64 {
		return k.StepTime(decodeBatch(16, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
	}
	if a, b := hbm(coarse, 100), hbm(fine, 100); a <= b {
		t.Errorf("a 100-token context read %.6f ms of HBM at a stripe of 64 against %.6f at "+
			"1; the slowest rank holds 64 tokens against 13", a*1e3, b*1e3)
	}
	if a, b := hbm(coarse, 4096), hbm(fine, 4096); a != b {
		t.Errorf("a 4,096-token context read %.6f ms at a stripe of 64 against %.6f at 1; "+
			"it is a multiple of stripe x dcp, so every rank holds 512 under both", a*1e3, b*1e3)
	}
	// Capacity does not see the stripe: the engine allocates block_size x dcp tokens per
	// logical block whatever the interleave (kv_cache_interface.py:545-550), so a rank
	// holds the same pages either way.
	if a, b := coarse.SequenceVariableBytes(100), fine.SequenceVariableBytes(100); a != b {
		t.Errorf("the stripe moved capacity: %d against %d", a, b)
	}
}

// UNDER NIXL AN UNSTATED STRIPE IS THE BLOCK SIZE. v0.31.0 pins an unstated
// cp_kv_cache_interleave_size to the local block size when NixlConnector is configured, and
// honours a stated one; any other connector leaves the default of 1. vLLM looks inside a
// MultiConnector for a NIXL child (vllm/config/kv_transfer.py:156-163), but a deployment does
// not name its children, so the answer is unknown and the kernel refuses rather than guess
// (vllm/config/vllm.py:3333-3378).
//
// Priced on the decode pool of a disaggregated deployment, which is the case the pin exists
// for: a decode pool sharding its cache while NIXL moves blocks in from prefill.
func TestUnderNIXLAnUnstatedInterleaveIsTheBlockSize(t *testing.T) {
	r := writeDisaggregated(t)
	build := func(connector string, interleave int) (*Kernel, error) {
		t.Helper()
		in, err := OpenInputs("disagg.yaml", r, 1)
		if err != nil {
			t.Fatal(err)
		}
		in.Deployment.PDTransfer.Connector = connector
		pool := &in.Deployment.Pools[1]
		pool.Parallel.DCP = 4
		pool.Engine.BlockSize = 64
		pool.Engine.CPKVCacheInterleaveSize = interleave
		return New(in)
	}
	must := func(connector string, interleave int) *Kernel {
		t.Helper()
		k, err := build(connector, interleave)
		if err != nil {
			t.Fatalf("%s, interleave %d: %v", connector, interleave, err)
		}
		return k
	}
	decode := decodeBatch(16, 1, 100)
	price := func(k *Kernel) float64 { return k.StepTime(decode).NoOverlap.Seconds() }

	pinned := must("NixlConnector", 0)
	if a, b := price(pinned), price(must("NixlConnector", 64)); a != b {
		t.Errorf("an unstated stripe under NIXL priced %.6f ms against %.6f with the block "+
			"size stated; the engine pins it to the block size", a*1e3, b*1e3)
	}
	if a, b := price(must("NixlConnector", 1)), price(must("MooncakeConnector", 0)); a != b {
		t.Errorf("a stated stripe of 1 under NIXL priced %.6f ms against %.6f for the "+
			"default under another connector; a stated size is honoured", a*1e3, b*1e3)
	}
	if price(pinned) == price(must("MooncakeConnector", 0)) {
		t.Error("the NIXL pin did not change the price of a short-context decode")
	}
	recorded := false
	for _, o := range pinned.Resolved().Overrides {
		recorded = recorded || (o.Field == "cp_kv_cache_interleave_size" && o.Resolved == "64")
	}
	if !recorded {
		t.Errorf("the pin is not recorded in Resolved: %+v", pinned.Resolved().Overrides)
	}
	if _, err := build("MultiConnector", 0); err == nil {
		t.Error("an unstated stripe under a MultiConnector was priced; whether it is " +
			"pinned depends on children a deployment cannot name")
	}
	if _, err := build("MultiConnector", 16); err != nil {
		t.Errorf("a stated stripe under a MultiConnector was refused: %v", err)
	}
}

// The fabric pcpInputs adds must be loadable from the same catalog the fixtures read; this
// keeps dcpVariant's multi-node layouts honest if the catalog moves.
func TestTheFabricDCPLayoutsDeclareIsInTheCatalog(t *testing.T) {
	if _, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", "ib-400g.yaml")); err != nil {
		t.Fatal(err)
	}
}
