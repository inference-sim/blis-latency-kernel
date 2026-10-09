package latencykernel

import (
	"math"
	"path/filepath"
	"testing"

	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"

	"github.com/inference-sim/blis-latency-kernel/internal/price"
)

// Prefill-context parallelism splits a prefill sequence across ranks, and these pin that
// behaviourally.
//
// THE DEFECT. PCP was half-modelled where DCP was not modelled at all. It reached a price,
// but only as a side effect of the expert group -- ExpertParallelWidth widens with pcp --
// so a wider PCP sharded more experts and paid a wider all-to-all. The sharding PCP is
// NAMED for -- "Number of ranks that split prefill sequence computation"
// (vllm/config/parallel.py:131-133 at v0.31.0) -- was not credited at all, so a PCP
// deployment's prefill was priced as whole work on every rank. The error is one-sided and
// the opposite of DCP's: prefill was OVERSTATED.
//
// WHAT THE ENGINE DOES, which these tests encode:
//
//	a rank runs a forward over its own share   worker/gpu/pcp_manager.py:485
//	the split is a zigzag over 2*pcp chunks    worker/gpu/pcp_manager.py:235-273
//	decodes are replicated, not split          worker/gpu/pcp_manager.py:260-263
//	the KV cache stays replicated              config/parallel.py:131-133
//	so the ranks all-gather what they wrote     attention/ops/pcp.py:31-35
//
// The asymmetry against DCP is the thing most easily got backwards, and two tests here
// assert the non-effects rather than leaving them to inference.
const pcpFixture = dcpMLAFixture

// pcpKernel builds a kernel from a committed fixture with the prefill-context-parallel
// width changed, laid out the way the engine actually runs it.
func pcpKernel(t testing.TB, fixture string, pcp int) *Kernel {
	t.Helper()
	in := pcpInputs(t, fixture, pcp)
	k, err := New(in)
	if err != nil {
		t.Fatalf("%s at pcp=%d: %v", fixture, pcp, err)
	}
	return k
}

// pcpInputs is a committed fixture's Inputs with the prefill-context-parallel width set to
// pcp AND the nodes to hold it.
//
// PCP EXPANDS THE WORLD SIZE. vLLM's ParallelConfig sets world_size = pp * tp * pcp
// (vllm/config/parallel.py:899-903 at v0.31.0) -- DCP reuses ranks, PCP adds them -- so a
// tp=8 fixture on one 8-GPU node at pcp=2 needs sixteen GPUs. An earlier form of this
// helper set the width alone and priced exactly that sixteen-rank layout on eight GPUs,
// which the engine cannot start; blis-schemas v0.2.2 refuses it ("needs 16 GPUs (pp 1 x
// tp 8 x pcp 2 x dp 1, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8").
//
// So the pool and the cluster grow by the split, which is the shape PCP is deployed in:
// the tensor-parallel group stays inside a node and the PCP ranks are added across nodes.
// The ranks are laid out DP x PP x PCP x TP with TP innermost (parallel_state.py:2054-2060),
// which is what places a PCP group's members one tensor-parallel width apart and therefore
// on different nodes here.
//
// A cross-node cluster must name its fabric, a rule blis-schemas held before PCP existed.
// The fabric is set at EVERY width, pcp=1 included, so a comparison across widths differs
// in the split and nothing else; nothing crosses a node at pcp=1, so it prices no term there.
func pcpInputs(t testing.TB, fixture string, pcp int) Inputs {
	t.Helper()
	in := fixtureInputs(t, fixture)
	if len(in.Deployment.Pools) != 1 {
		t.Fatalf("%s has %d pools; pcpInputs grows one pool and its cluster together",
			fixture, len(in.Deployment.Pools))
	}
	pool := &in.Deployment.Pools[0]
	pool.Parallel.PCP = pcp
	if pcp > 1 {
		pool.Nodes *= pcp
		in.Scenario.Cluster.Nodes *= pcp
	}
	if in.Fabric == nil {
		const fabric = "ib-400g"
		f, err := schemas.LoadFabric(filepath.Join(catalogRoot, "networks", fabric+".yaml"))
		if err != nil {
			t.Fatalf("fabric %s: %v", fabric, err)
		}
		in.Scenario.Cluster.Fabric, in.Fabric = fabric, f
	}
	return in
}

// ---------------------------------------------------------------------------
// The partition arithmetic, against the engine's own chunking.
// ---------------------------------------------------------------------------

// vllmPCPRankTokens is _iter_rank_chunks transcribed for one rank
// (vllm/v1/worker/gpu/pcp_manager.py:235-273 at v0.31.0): each prefill is cut into 2*pcp
// chunks of ceil(sched/(2*pcp)), and rank r takes chunks r and 2*pcp-1-r. It is the ORACLE
// the closed form must match.
func vllmPCPRankTokens(sched, pcp, rank int) int {
	numChunks := 2 * pcp
	chunk := (sched + numChunks - 1) / numChunks
	total := 0
	for _, idx := range [...]int{rank, numChunks - 1 - rank} {
		lo := idx * chunk
		hi := lo + chunk
		if hi > sched {
			hi = sched
		}
		if hi > lo {
			total += hi - lo
		}
	}
	return total
}

// The busiest rank binds, because the ranks gather before the next layer and the step waits
// for the slowest. And the shards must conserve the chunk: a split that lost or duplicated
// tokens would misprice every prefill.
//
// Mutation-checked: a nominal sched/pcp, a floor instead of a ceiling, and 2*pcp in place
// of pcp each fail here.
func TestPCPLocalTokensMatchesTheBusiestRankAndConservesTheChunk(t *testing.T) {
	for _, pcp := range []int{2, 4, 8} {
		for sched := 1; sched < 600; sched++ {
			want, sum := 0, 0
			for rank := 0; rank < pcp; rank++ {
				v := vllmPCPRankTokens(sched, pcp, rank)
				if v > want {
					want = v
				}
				sum += v
			}
			if sum != sched {
				t.Fatalf("the oracle does not conserve: pcp=%d sched=%d shards sum to %d",
					pcp, sched, sum)
			}
			if got := pcpLocalTokens(sched, pcp, 1); got != want {
				t.Fatalf("pcp=%d sched=%d: got %d, want %d (the busiest rank's share)",
					pcp, sched, got, want)
			}
		}
	}
}

// Replication of a short prefill happens ONLY ALONGSIDE DCP, and the condition is not the
// one the mechanism suggests.
//
// replicated_requests (pcp_manager.py:222-233 at v0.31.0) computes `drops_a_chunk` inside
// `if self.dcp_world_size > 1`, so with PCP alone a prefill is partitioned however short
// it is, and a rank that owns no chunk simply contributes nothing. An earlier draft of
// this test asserted the opposite -- that any prefill under 2*pcp tokens is replicated at
// every width -- and it failed against correct code, which is recorded here because the
// gate is the kind of detail a reader reconstructs wrongly from the mechanism alone.
//
// The condition is `(2*pcp - 1) * ceil(sched/(2*pcp)) >= sched`, which detects a partition
// with an empty chunk. It is NOT "shorter than 2*pcp": at pcp 8 it holds at 17 tokens and
// fails at 16.
func TestPCPReplicatesOnlyAlongsideDCPAndOnTheEnginesOwnCondition(t *testing.T) {
	for _, pcp := range []int{2, 4, 8} {
		numChunks := 2 * pcp
		for sched := 1; sched < 80; sched++ {
			chunk := (sched + numChunks - 1) / numChunks
			replicates := (numChunks-1)*chunk >= sched

			// With DCP off there is no replication branch: the busiest rank holds its
			// own two chunks, each truncated where it runs past the query. Taken from
			// the oracle rather than as 2*chunk, which is right only when every chunk
			// is full -- at two tokens over two ranks the chunks are one token each and
			// the busiest rank holds ONE, not two.
			withoutDCP := pcpLocalTokens(sched, pcp, 1)
			wantWithout := 0
			for rank := 0; rank < pcp; rank++ {
				if v := vllmPCPRankTokens(sched, pcp, rank); v > wantWithout {
					wantWithout = v
				}
			}
			if withoutDCP != wantWithout {
				t.Errorf("pcp=%d sched=%d dcp=1: got %d, want %d", pcp, sched,
					withoutDCP, wantWithout)
			}

			// With DCP on, a partition that would drop a chunk is replicated instead.
			withDCP := pcpLocalTokens(sched, pcp, 2)
			if replicates {
				if withDCP != sched {
					t.Errorf("pcp=%d sched=%d dcp=2: got %d, want the whole query %d "+
						"(this partition drops a chunk, so the engine replicates)",
						pcp, sched, withDCP, sched)
				}
			} else if withDCP != wantWithout {
				t.Errorf("pcp=%d sched=%d dcp=2: got %d, want %d (no chunk is dropped, "+
					"so the query is partitioned)", pcp, sched, withDCP, wantWithout)
			}
		}
	}
	// The documented boundary: at pcp 8 the condition holds at 17 and fails at 16, which
	// is what rules out reading it as "shorter than 2*pcp".
	if got := pcpLocalTokens(17, 8, 2); got != 17 {
		t.Errorf("17 tokens over 8 prefill-context ranks alongside DCP: got %d, want 17", got)
	}
	if got := pcpLocalTokens(16, 8, 2); got != 2 {
		t.Errorf("16 tokens over 8 prefill-context ranks alongside DCP: got %d, want 2 "+
			"(sixteen chunks of one, two per rank)", got)
	}
	// Absent or degenerate widths read as "no split", never as a division by zero.
	for _, pcp := range []int{1, 0, -2} {
		if got := pcpLocalTokens(4096, pcp, 1); got != -1 {
			t.Errorf("pcp=%d should read as absent: got %d, want -1", pcp, got)
		}
	}
	if got := pcpLocalTokens(0, 4, 1); got != -1 {
		t.Errorf("an empty chunk should read as absent: got %d", got)
	}
}

// The causal count must reproduce the unsharded figure's own CONTINUUM convention, so an
// inactive split changes nothing and a 2*pcp-divisible one is exactly the whole count over
// pcp.
//
// The repository's unsharded term is s*c + s^2/2, which is half a token below the exact
// per-position sum s*c + s(s+1)/2. windowedCausalFLOPs already subtracts that half-token
// for the same reason. Getting it wrong here would have made every PCP prefill 0.005%
// expensive at a 4,096-token chunk and, worse, made an exactly-divisible split disagree
// with the figure it replaces -- which is how this was caught.
func TestPCPCausalCountKeepsTheUnshardedContinuumConvention(t *testing.T) {
	for _, c := range []struct{ sched, prefix int }{
		{4096, 8192}, {2048, 0}, {8192, 131072}, {1024, 203760},
	} {
		s := float64(c.sched)
		whole := 2 * 2 * (s*float64(c.prefix) + s*s*0.5)
		for _, pcp := range []int{2, 4, 8} {
			if c.sched%(2*pcp) != 0 {
				continue // only the exactly-divisible case has a closed expectation
			}
			got := pcpCausalPairs(c.sched, c.prefix, pcp, 1)
			if want := whole / float64(pcp); math.Abs(got-want)/want > 1e-12 {
				t.Errorf("sched=%d prefix=%d pcp=%d: %v pairs, want %v (the unsharded "+
					"count over the split)", c.sched, c.prefix, pcp, got, want)
			}
		}
	}
	// A RAGGED PARTITION COSTS MORE THAN THE NOMINAL SHARE, and the exact count is what
	// says by how much. This is the case that separates counting the engine's partition
	// from dividing the whole figure by pcp: where 2*pcp divides the query the two agree
	// exactly, and where it does not the busiest rank carries measurably more, because a
	// ceil-based chunking hands one rank two chunks from the expensive end.
	//
	// Pinned at both ends so neither form can pass: exact agreement at 4,096 tokens over
	// 8 ranks, and a strict excess at 1,001 over 8 (1.4%) and 100 over 8 (25.4%).
	for _, c := range []struct {
		sched, prefix, pcp int
		wantRatio          float64
		tol                float64
	}{
		{4096, 0, 8, 1.000000, 1e-12}, // 2*pcp divides the query: the two forms agree
		{1001, 0, 8, 1.014035, 1e-5},  // ragged
		{1001, 8192, 4, 1.007398, 1e-5},
		{100, 0, 8, 1.254400, 1e-5}, // a query barely longer than the chunk count
	} {
		s := float64(c.sched)
		nominal := 2 * 2 * (s*float64(c.prefix) + s*s*0.5) / float64(c.pcp)
		got := pcpCausalPairs(c.sched, c.prefix, c.pcp, 1)
		if ratio := got / nominal; math.Abs(ratio-c.wantRatio) > c.tol {
			t.Errorf("sched=%d prefix=%d pcp=%d: the busiest rank carries %.6f of the "+
				"nominal share, want %.6f; the count must follow the engine's partition "+
				"rather than dividing the whole figure",
				c.sched, c.prefix, c.pcp, ratio, c.wantRatio)
		}
	}

	// A ragged partition costs the busiest rank MORE than the ideal share, and the
	// overhead is bounded: the engine's chunking is ceil-based, so one rank carries two
	// chunks from the expensive end. Asserted as a bound rather than a value because the
	// figure depends on where the remainder falls.
	sched, prefix, pcp := 1001, 0, 8
	s := float64(sched)
	ideal := 2 * 2 * (s*float64(prefix) + s*s*0.5) / float64(pcp)
	got := pcpCausalPairs(sched, prefix, pcp, 1)
	if got < ideal {
		t.Errorf("a ragged split gave %v pairs, below the ideal share %v; the busiest "+
			"rank cannot carry less than an even share", got, ideal)
	}
	if got > ideal*1.05 {
		t.Errorf("a ragged split gave %v pairs, more than 5%% above the ideal share %v",
			got, ideal)
	}
}

// THE ZIGZAG IS WHAT MAKES A SINGLE DIVISOR DEFENSIBLE for the causal term, and this test
// is the evidence rather than the claim.
//
// Pairing chunk r with chunk 2*pcp-1-r pairs a cheap early chunk, with few keys to its
// left, against an expensive late one. Where 2*pcp divides the chunk the per-rank causal
// pair counts come out EXACTLY equal, so dividing the batch-wide count by pcp is the right
// per-rank figure and not an approximation. A contiguous 1/pcp slice would not balance at
// all: the last rank would carry the whole upper triangle.
//
// Asserted on the engine's own partition, over prefixes too, since a chunked prefill is the
// case the repository's causal count was corrected for.
func TestPCPZigzagBalancesTheCausalWorkExactly(t *testing.T) {
	pairs := func(sched, prefix, pcp, rank int) float64 {
		numChunks := 2 * pcp
		chunk := (sched + numChunks - 1) / numChunks
		var total float64
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := idx * chunk
			hi := lo + chunk
			if hi > sched {
				hi = sched
			}
			for j := lo; j < hi; j++ {
				total += float64(prefix + j + 1)
			}
		}
		return total
	}
	for _, c := range []struct{ sched, prefix int }{
		{2048, 0}, {4096, 8192}, {1024, 131072}, {8192, 0}, {2048, 203760},
	} {
		for _, pcp := range []int{2, 4, 8} {
			var ideal float64
			for j := 0; j < c.sched; j++ {
				ideal += float64(c.prefix + j + 1)
			}
			ideal /= float64(pcp)
			lo, hi := math.Inf(1), 0.0
			for rank := 0; rank < pcp; rank++ {
				v := pairs(c.sched, c.prefix, pcp, rank)
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
			// Exactly equal across ranks, so the ideal share IS each rank's share.
			if hi != lo {
				t.Errorf("sched=%d prefix=%d pcp=%d: ranks carry between %.0f and %.0f "+
					"causal pairs; the zigzag must balance them exactly",
					c.sched, c.prefix, pcp, lo, hi)
			}
			if math.Abs(hi-ideal)/ideal > 1e-12 {
				t.Errorf("sched=%d prefix=%d pcp=%d: each rank carries %.0f pairs against "+
					"an ideal share of %.0f", c.sched, c.prefix, pcp, hi, ideal)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Step time: the half of the issue PCP names.
// ---------------------------------------------------------------------------

// THE ACCEPTANCE CRITERION for the PCP half: a wider split must make a prefill cheaper,
// monotonically. Before this the prefill term did not move at all.
//
// Four widths, not two, so a constant divisor cannot pass.
func TestPCPLowersAPrefillStepAndWideningItLowersFurther(t *testing.T) {
	batch := decodeBatch(1, 4096, 4096)
	var prev float64
	for i, pcp := range []int{1, 2, 4, 8} {
		got := pcpKernel(t, pcpFixture, pcp).StepTime(batch).NoOverlap.Seconds()
		if i > 0 && got >= prev {
			t.Errorf("pcp=%d priced %.6f ms, not below the %.6f ms of the previous "+
				"width: splitting a prefill further must compute less per rank",
				pcp, got*1e3, prev*1e3)
		}
		prev = got
	}
}

// The causal attention term must divide by exactly the split. Isolated the same way the
// DCP read is: with work(p) = W/p plus terms independent of p, the difference of
// differences cancels everything else and must be exactly 2.
//
// Charged to SM, because prefill attention is compute-bound -- which is itself the reason
// the regime carries its own measured floor and scale.
func TestPCPDividesTheCausalTermByExactlyTheSplit(t *testing.T) {
	batch := decodeBatch(1, 4096, 4096)
	sm := func(pcp int) float64 {
		return pcpKernel(t, pcpFixture, pcp).StepTime(batch).
			PerResource[kernel.ResourceSM].Seconds()
	}
	s1, s2, s4 := sm(1), sm(2), sm(4)
	if s2 >= s1 || s4 >= s2 {
		t.Fatalf("the SM term is not falling with the split: %.6f, %.6f, %.6f ms",
			s1*1e3, s2*1e3, s4*1e3)
	}
	if ratio := (s1 - s2) / (s2 - s4); math.Abs(ratio-2) > 1e-6 {
		t.Errorf("(t1-t2)/(t2-t4) is %.9f, want 2: the prefill work is not dividing by "+
			"exactly the split width", ratio)
	}

	// AND THE DIVISOR MUST BE THE ENGINE'S PARTITION, NOT A NOMINAL 1/pcp. The two agree
	// whenever 2*pcp divides the chunk -- 4,096 is divisible by 16, which is why the ratio
	// above cannot tell them apart -- and diverge on a ragged one: at 100 tokens over 8
	// ranks the busiest rank carries 25.4% more causal work than an even share, because a
	// ceil-based chunking hands it two chunks from the expensive end.
	//
	// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. An exact call-site ratio is NOT
	// recoverable, and the reason is itself a modelled effect rather than a weakness of
	// the test. Prefill SM time is floors plus causal count over (peak * efficiency), and
	// the efficiency ramp is evaluated at the RANK-LOCAL token count -- correctly, since a
	// prefill-context rank runs its forward over its own share -- so the rate differs
	// between two widths and does not cancel under differencing. Two drafts of this test
	// tried: differencing across widths read 3.09 where the partition predicts 9.00, and
	// dividing out the rate implied a pair count 1.69x the true one, because the ramp had
	// moved. On top of that, the chunk sizes where the two divisors differ most are the
	// ones where the per-call floors are the largest share of the step: 1.4% excess at
	// 1,001 tokens where floors are 12.6% of SM, against 0.024% excess at 8,191 where
	// they are 2.1%.
	//
	// So the exact arithmetic is pinned on the FUNCTION, by
	// TestPCPCausalCountKeepsTheUnshardedContinuumConvention, which checks the busiest
	// rank's count against the engine's partition including the ragged cases. What is
	// pinned here is that the call site consumes it: the excess exists and the step
	// carries a causal term above its floors.
	for _, ragged := range []int{100, 1001} {
		s := float64(ragged)
		nominal := 2 * 2 * (s * s * 0.5) / 8
		exact := pcpCausalPairs(ragged, 0, 8, 1)
		if exact <= nominal {
			t.Errorf("at %d tokens over 8 ranks the engine's partition gives %v pairs "+
				"against an even share of %v; a ceil-based chunking cannot give less",
				ragged, exact, nominal)
		}
		// The step must reflect that excess: pricing the nominal share would make this
		// strictly cheaper, and the gap is bounded below by the excess over the rate.
		got := pcpKernel(t, pcpFixture, 8).StepTime(decodeBatch(1, ragged, ragged)).
			PerResource[kernel.ResourceSM].Seconds()
		floors := pcpKernel(t, pcpFixture, 8).attentionPrefillFloor.Seconds() *
			float64(pcpKernel(t, pcpFixture, 8).plan.TotalLayers)
		if got <= floors {
			t.Errorf("a %d-token prefill at pcp=8 priced %.6f ms, at or below its %.6f ms "+
				"of per-layer floors; the causal term is missing", ragged, got*1e3, floors*1e3)
		}
	}
}

// PCP MUST NOT TOUCH THE DECODE READ. It splits prefill computation and leaves the KV
// cache replicated -- "PCP expands the process world size but does not increase the
// KV-cache shard count" (config/parallel.py:131-133) -- which is exactly the opposite of
// DCP. Conflating the two is the single easiest thing to get backwards, so the non-effect
// is asserted rather than inferred.
func TestPCPDoesNotShardTheDecodeRead(t *testing.T) {
	decode := decodeBatch(32, 1, 32768)
	want := pcpKernel(t, pcpFixture, 1).StepTime(decode).NoOverlap
	for _, pcp := range []int{2, 4, 8} {
		if got := pcpKernel(t, pcpFixture, pcp).StepTime(decode).NoOverlap; got != want {
			t.Errorf("pcp=%d priced a decode-only step at %v against %v at pcp=1; a "+
				"decode row is replicated across prefill-context ranks, and the cache "+
				"it reads is replicated too", pcp, got, want)
		}
	}
}

// And the converse, so neither axis can quietly acquire the other's divisor: PCP must not
// shard the capacity a rank holds, where DCP does.
func TestPCPDoesNotShardCapacityWhereDCPDoes(t *testing.T) {
	want := pcpKernel(t, pcpFixture, 1).SequenceVariableBytes(32768)
	for _, pcp := range []int{2, 8} {
		if got := pcpKernel(t, pcpFixture, pcp).SequenceVariableBytes(32768); got != want {
			t.Errorf("pcp=%d holds %d bytes against %d at pcp=1; the KV cache is "+
				"replicated across prefill-context ranks", pcp, got, want)
		}
	}
	// The same quantity DOES move with dcp, which is what makes the contrast meaningful
	// rather than a tautology about an unread field.
	if sharded := dcpKernel(t, pcpFixture, 8).SequenceVariableBytes(32768); sharded >= want {
		t.Errorf("dcp=8 holds %d bytes against %d unsharded; the decode axis must shard "+
			"the cache even though the prefill axis does not", sharded, want)
	}
}

// The per-call prefill floor must NOT divide with the split, and the split must NOT make a
// rank's work exactly linear either.
//
// Two claims, isolated by holding the RANK-LOCAL token count fixed: at pcp 1, 2 and 4 with
// chunks of 1024, 2048 and 4096, every rank computes 1,024 token-rows, so the GEMM shapes
// and the efficiency ramp's argument are identical across the three. What differs is the
// GLOBAL sequence each of those tokens attends, and the floors.
//
// The SM term must then RISE with the split, not fall and not hold: a rank doing the same
// number of rows against a longer global sequence does strictly more attention work, while
// its per-call floors stay put. A floor scaled by the split would pull this the other way.
//
// Measured: 38.39, 40.63 and 45.13 ms at pcp 1, 2 and 4.
func TestPCPDoesNotDivideThePerCallFloorOrMakeARankLinear(t *testing.T) {
	const perRank = 1024
	sm := func(pcp, sched int) float64 {
		return pcpKernel(t, pcpFixture, pcp).StepTime(decodeBatch(1, sched, sched)).
			PerResource[kernel.ResourceSM].Seconds()
	}
	var prev float64
	for i, pcp := range []int{1, 2, 4} {
		sched := perRank * pcp
		if local := pcpLocalTokens(sched, pcp, 1); pcp > 1 && local != perRank {
			t.Fatalf("pcp=%d: a %d-token chunk leaves %d tokens on a rank, want %d; the "+
				"comparison needs the rank-local count held fixed", pcp, sched, local, perRank)
		}
		got := sm(pcp, sched)
		if i > 0 && got <= prev {
			t.Errorf("pcp=%d priced %.6f ms against %.6f ms at the previous width; a rank "+
				"computing the same %d rows against a longer global sequence does more "+
				"attention work, and its floors do not shrink", pcp, got*1e3, prev*1e3, perRank)
		}
		prev = got
	}

	// And the floors are really there: a chunk short enough that the causal work vanishes
	// still costs at least one prefill floor per layer, at every width.
	for _, pcp := range []int{1, 2, 4} {
		k := pcpKernel(t, pcpFixture, pcp)
		tiny := k.StepTime(decodeBatch(1, 2*2*pcp, 2*2*pcp)).
			PerResource[kernel.ResourceSM].Seconds()
		floors := k.attentionPrefillFloor.Seconds() * float64(k.plan.TotalLayers)
		if tiny < floors {
			t.Errorf("pcp=%d: a near-empty prefill priced %.6f ms, below the %.6f ms its "+
				"per-layer floors alone cost", pcp, tiny*1e3, floors*1e3)
		}
	}
}

// The structural check, as for DCP: poke the resolved layout on a built kernel and require
// the prefill price to move. This catches a field that is plumbed but never read, which is
// the state PCP's prefill split was in.
func TestPCPGeometryReachesThePricer(t *testing.T) {
	k := fixture(t, pcpFixture)
	batch := decodeBatch(1, 4096, 4096)
	before := k.StepTime(batch).NoOverlap
	k.layout.PCP = 4
	after := k.StepTime(batch).NoOverlap
	if after >= before {
		t.Errorf("setting the resolved PCP width to 4 left the prefill step at %v "+
			"against %v; the layout field is not reaching the pricer", after, before)
	}
}

// ---------------------------------------------------------------------------
// The gather PCP pays for keeping every rank's cache whole.
// ---------------------------------------------------------------------------

// A PCP rank writes only its share of the new KV, but every rank must hold the whole
// cache, so the ranks all-gather what they wrote. Omitting it would make PCP look like a
// pure win, which it is not.
//
// Two properties: the gather exists on a prefill step, and it does NOT exist on a
// decode-only one -- the engine keeps replicated decode writes local and gathers
// partitioned prefills only (attention/ops/pcp.py:16, :31-35).
func TestPCPAddsAKVGatherOnPrefillOnly(t *testing.T) {
	link := func(pcp int, b kernel.Batch) float64 {
		return pcpKernel(t, pcpFixture, pcp).StepTime(b).
			PerResource[kernel.ResourceNVLink].Seconds()
	}
	// A decode-only step carries no PCP gather at all.
	decode := decodeBatch(32, 1, 32768)
	if on, off := link(2, decode), link(1, decode); on != off {
		t.Errorf("a decode-only step carried %.6f ms of on-node collective at pcp=2 "+
			"against %.6f ms at pcp=1; a replicated decode write is not gathered",
			on*1e3, off*1e3)
	}

	// ON A SHORT PREFILL THE GATHER CHANGES THE SIGN, which is the shape that makes its
	// omission visible. Splitting halves the tensor-parallel payload, so the on-node total
	// would fall if nothing were added -- but the gather pays a floor on every KV-holding
	// layer, and at a small chunk those floors outweigh what the halving saves. So the
	// on-node term must RISE.
	//
	// At a long chunk it falls again, because the bytes saved dominate the floors added.
	// Both directions are asserted: a test that only looked at the long chunk would pass
	// with the gather deleted.
	short := decodeBatch(1, 64, 64)
	if whole, split := link(1, short), link(2, short); split <= whole {
		t.Errorf("a 64-token prefill priced %.6f ms of on-node collective at pcp=2 "+
			"against %.6f ms at pcp=1; the KV gather's per-layer floors must outweigh "+
			"the halved tensor-parallel payload at this size", split*1e3, whole*1e3)
	}
	long := decodeBatch(1, 4096, 4096)
	if whole, split := link(1, long), link(2, long); split >= whole {
		t.Errorf("a 4,096-token prefill priced %.6f ms at pcp=2 against %.6f ms at "+
			"pcp=1; at this size the bytes saved must dominate the floors added",
			split*1e3, whole*1e3)
	}

	// And the charge must be at least one floor per KV-holding layer, at the PCP group's
	// own width. Pricing it at the tensor-parallel width would roughly double it -- the
	// 8-rank all-gather floor is 10.3us against the 2-rank 5.08us on this part -- so the
	// bound below is stated against the width this group actually spans.
	k := pcpKernel(t, pcpFixture, 2)
	floor := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupPCP}]
	if floor <= 0 {
		t.Fatal("no all-gather floor resolved for the prefill-context group")
	}
	floors := floor.Seconds() * float64(k.plan.TotalLayers)
	tpFloor := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupTP}]
	tpFloors := tpFloor.Seconds() * float64(k.plan.TotalLayers)

	// The NET change on the short prefill is the gather added MINUS the tensor-parallel
	// payload the split saved, so it cannot be compared against the gather's floors
	// directly -- an earlier draft of this test did exactly that and failed against
	// correct code. What the net DOES bound is the width: the gather is charged at the
	// prefill-context group's own 2-rank floor, and at the 8-rank floor its floors alone
	// would be tpFloors, which exceeds the whole on-node term at this size. So a net
	// change smaller than the difference between the two floor totals is only reachable
	// at the narrower width.
	added := link(2, short) - link(1, short)
	if added >= tpFloors-floors {
		t.Errorf("the split changed the on-node term by %+.6f ms, at or above the "+
			"%.6f ms by which %d layers of 8-rank floor exceed %d layers of 2-rank; the "+
			"gather is being priced at the tensor-parallel width",
			added*1e3, (tpFloors-floors)*1e3, k.plan.TotalLayers, k.plan.TotalLayers)
	}
	if added <= 0 {
		t.Errorf("the split changed the on-node term by %+.6f ms; the gather must add "+
			"more than the halved payload saves at this size", added*1e3)
	}
}

// The gather is charged on PREFILL tokens only, and on KV-HOLDING layers only.
//
// Two scoping facts the engine states and a looser implementation would blur. It gathers
// "partitioned prefills" while keeping "replicated decode writes local" (pcp.py:16), and
// what it gathers is cache input, so a layer with no cache contributes nothing.
//
// Both are asserted against a shape where the distinction is visible, because on a pure
// prefill of an all-attention model neither is: the first needs decode rows present, and
// the second needs a model whose layers are not all attention. Nemotron-3-Ultra is that
// model -- 96 of its 108 layers hold no KV.
func TestThePCPGatherIsScopedToPrefillTokensAndKVLayers(t *testing.T) {
	// dp is cleared so the hybrid fixture's three replicas do not multiply the node count
	// pcpInputs grows: the gather's scoping is a per-rank question, and a wider expert or
	// data-parallel layout would only add collectives this comparison has to subtract.
	build := func(fixture string, pcp int) *Kernel {
		t.Helper()
		in := pcpInputs(t, fixture, pcp)
		in.Deployment.Pools[0].Parallel.DP = 1
		k, err := New(in)
		if err != nil {
			t.Fatalf("%s at pcp=%d: %v", fixture, pcp, err)
		}
		return k
	}
	link := func(fixture string, pcp int, b kernel.Batch) float64 {
		return build(fixture, pcp).StepTime(b).
			PerResource[kernel.ResourceNVLink].Seconds()
	}

	// PREFILL TOKENS ONLY. A 64-token prefill alone, and the same prefill beside 64
	// decode rows, must add exactly the same gather -- the decode rows are replicated, so
	// their KV is already on every rank and is not gathered.
	alone := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 64, Computed: 0, PromptLen: 64},
	}}
	withDecodes := kernel.Batch{DecodeThreshold: 8, Reqs: append(
		[]kernel.ReqShape{{Scheduled: 64, Computed: 0, PromptLen: 64}},
		make([]kernel.ReqShape, 64)...)}
	for i := 1; i < len(withDecodes.Reqs); i++ {
		withDecodes.Reqs[i] = kernel.ReqShape{
			Scheduled: 1, Computed: 32767, PromptLen: 32768,
		}
	}
	addedAlone := link(pcpFixture, 2, alone) - link(pcpFixture, 1, alone)
	addedMixed := link(pcpFixture, 2, withDecodes) - link(pcpFixture, 1, withDecodes)
	if math.Abs(addedMixed-addedAlone) > 1e-9 {
		t.Errorf("the gather added %.6f ms beside 64 decode rows against %.6f ms without "+
			"them, on the same 64-token prefill; a replicated decode write is not "+
			"gathered", addedMixed*1e3, addedAlone*1e3)
	}

	// KV-HOLDING LAYERS ONLY. On a model whose layers are mostly cache-free, the gather
	// must cost at most what its KV-holding layers' floors come to -- charging every
	// layer would be several times that.
	const hybrid = "nemotron3-ultra-h100-agg.yaml"
	k := build(hybrid, 2)
	var kvLayers, allLayers int
	for i := range k.plan.Layers {
		l := &k.plan.Layers[i]
		allLayers += l.Count
		if l.AttnQHeads > 0 {
			kvLayers += l.Count
		}
	}
	if kvLayers == 0 || kvLayers >= allLayers {
		t.Skipf("%s holds KV on %d of %d layers, so this cannot discriminate",
			hybrid, kvLayers, allLayers)
	}
	floor := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupPCP}]
	// Bounded against the KV-layer floors, not the all-layer ones: on this model the two
	// differ ninefold (0.0617 ms against 0.5551 ms over 12 KV layers of 108), and a
	// ceiling set at the larger figure is one a correct implementation clears so easily
	// that charging every layer would pass it too.
	//
	// The NET change also includes whatever the split saves on the tensor-parallel
	// collectives, so what is bounded is the gather's own contribution: it cannot exceed
	// the floors of the layers that hold KV by more than the bytes those layers move,
	// which at a 64-token prefill is far below one floor.
	kvFloors := floor.Seconds() * float64(kvLayers)
	allFloors := floor.Seconds() * float64(allLayers)
	added := link(hybrid, 2, alone) - link(hybrid, 1, alone)

	// Charging every layer rather than only the KV-holding ones would add
	// (allFloors - kvFloors) of floor that does not belong -- 0.49 ms on this part, over
	// 96 cache-free layers. The split SAVES on the tensor-parallel collectives, so the
	// net change is negative either way; what separates the two readings is HOW negative.
	// Measured: -0.522 ms scoped to KV layers against -0.028 ms charging all of them.
	//
	// Bounded at the midpoint, which is sound in both directions rather than fitted: the
	// spurious floors are a known quantity, so the scoped reading must sit at least half
	// of them below the unscoped one.
	if added > -(allFloors-kvFloors)/2 {
		t.Errorf("the split changed the on-node term by %+.6f ms on a model holding KV "+
			"on %d of %d layers; charging every layer would add %.6f ms of floor that "+
			"no cache-free layer pays, and the change is not far enough below that to "+
			"show the gather is scoped",
			added*1e3, kvLayers, allLayers, (allFloors-kvFloors)*1e3)
	}
}

// The PCP gather must resolve coefficients at ITS OWN width, not the tensor-parallel one or
// the decode-context one. Same hazard the composite coefficient key exists to prevent, now
// with three axes able to run an all-gather.
func TestThePCPGatherIsPricedAtItsOwnWidth(t *testing.T) {
	k := pcpKernel(t, pcpFixture, 2)
	if k.layout.TP != 8 {
		t.Fatalf("this fixture is expected to be tp=8, got %d: the test needs the two "+
			"widths to differ", k.layout.TP)
	}
	if got, ok := k.groupSize(price.GroupPCP); !ok || got != 2 {
		t.Errorf("the prefill-context group spans %d ranks, want 2", got)
	}
	pcpFloor, ok := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupPCP}]
	if !ok {
		t.Fatal("no all-gather floor resolved for the prefill-context group")
	}
	tpFloor := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupTP}]
	if pcpFloor >= tpFloor {
		t.Errorf("a 2-rank prefill-context all-gather floor of %v is not below the "+
			"8-rank tensor-parallel %v; the group is not selecting the width",
			pcpFloor, tpFloor)
	}
	// A deployment with PCP off must not demand the coefficient at all, so a part missing
	// a narrow width is not newly refused. Asserted through construction, which is the
	// observable consequence.
	if _, ok := fixture(t, pcpFixture).collectiveFloors[collKey{
		Op: model.OpAllGather, Group: price.GroupPCP}]; ok {
		t.Error("a prefill-context all-gather was resolved at pcp=1")
	}
}

// The expert group spans the prefill-context ranks as well as the tensor- and data-parallel
// ones: vLLM builds it over all three axes together, `data_parallel_size *
// prefill_context_model_parallel_size * tensor_model_parallel_size`
// (vllm/distributed/parallel_state.py:2212-2220 at v0.31.0).
//
// THE CASE THAT SEPARATES THE TWO FORMULAS is dp > 1 AND pcp > 1. blis-schemas v0.2.0
// computed tp * max(dp, pcp), which agrees with the product whenever one of dp and pcp is
// one -- every layout that version admitted -- and is a factor of min(dp, pcp) narrow once
// both exceed one, which v0.31.0 runs ("DP 4 x PCP 8 is an expert group of 32, not 8", in
// v0.2.2's own changelog). So the grid below includes that corner, and a regression to the
// max would fail exactly there.
//
// Asserted on the RESOLVED width, which is what selects the routed-expert shard and the
// all-to-all's group, and which a consumer reads.
func TestTheExpertGroupSpansTensorPrefillContextAndDataParallelRanks(t *testing.T) {
	for _, c := range []struct{ pcp, dp int }{{1, 1}, {2, 1}, {1, 2}, {2, 2}, {4, 2}} {
		in := pcpInputs(t, pcpFixture, c.pcp)
		pool := &in.Deployment.Pools[0]
		pool.Parallel.EnableExpertParallel = true
		pool.Parallel.DP = c.dp
		pool.Nodes *= c.dp
		in.Scenario.Cluster.Nodes *= c.dp
		k, err := New(in)
		if err != nil {
			t.Fatalf("pcp=%d dp=%d: %v", c.pcp, c.dp, err)
		}
		want := pool.Parallel.TP * c.pcp * c.dp
		if got := k.Resolved().ExpertParallelWidth; got != want {
			t.Errorf("tp=%d pcp=%d dp=%d: the expert group is %d ranks, want %d "+
				"(tp x pcp x dp)", pool.Parallel.TP, c.pcp, c.dp, got, want)
		}
	}
}

// The width a consumer reads must be the resolved one and floor at one, so an absent field
// reads as "not split" rather than zero.
func TestPrefillContextParallelWidthComesFromTheResolvedLayout(t *testing.T) {
	if got := pcpKernel(t, pcpFixture, 4).PrefillContextParallelWidth(); got != 4 {
		t.Errorf("got %d, want 4", got)
	}
	if got := fixture(t, pcpFixture).PrefillContextParallelWidth(); got != 1 {
		t.Errorf("an unstated width read as %d, want 1", got)
	}
	// The two context axes are independent and must not read each other's field.
	k := pcpKernel(t, pcpFixture, 2)
	if got := k.DecodeContextParallelWidth(); got != 1 {
		t.Errorf("setting pcp moved the decode-context width to %d", got)
	}
}

// Changing how the step's token count is derived must not reprice a batch PCP does not
// touch, and the request that exposes this is the one the accumulation loop SKIPS.
//
// THE DEFECT THIS WAS WRITTEN AGAINST. Crediting the prefill split meant the step's token
// count could no longer be the batch's own, so a draft re-summed it from the per-request
// loop. That loop skips a request whose context is non-positive -- `Computed + Scheduled
// <= 0`, reachable with a negative Computed, which the schema documents as a real state
// ("an engine advances it optimistically at dispatch and rolls it back on speculative
// rejection") -- while Batch.Tokens() still counts its scheduled tokens. The re-sum
// dropped them from every term derived from the token count, repricing such a batch by
// 10% with PCP off entirely: 11.64 ms against 12.87 ms.
//
// The fix subtracts what the split WITHHOLDS rather than rebuilding the total, so a
// skipped request contributes exactly what it always did.
func TestTheStepTokenCountStillCountsARequestTheLoopSkips(t *testing.T) {
	k := fixture(t, pcpFixture)

	withSkipped := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 2, Computed: -2},
		{Scheduled: 4, Computed: 0, PromptLen: 4},
	}}
	withoutSkipped := kernel.Batch{DecodeThreshold: 8, Reqs: []kernel.ReqShape{
		{Scheduled: 4, Computed: 0, PromptLen: 4},
	}}
	if withSkipped.Tokens() != 6 || withoutSkipped.Tokens() != 4 {
		t.Fatalf("the batches must differ by the skipped request's tokens: %d and %d",
			withSkipped.Tokens(), withoutSkipped.Tokens())
	}

	with := k.StepTime(withSkipped).NoOverlap
	without := k.StepTime(withoutSkipped).NoOverlap
	if with <= without {
		t.Errorf("a batch of %d tokens priced %v, at or below the %v of a batch of %d; "+
			"the tokens of a request the accumulation loop skips are being dropped from "+
			"the step's token count", withSkipped.Tokens(), with, without,
			withoutSkipped.Tokens())
	}
	// And PCP must not change that: it withholds only what it actually splits, and it
	// splits nothing here, since neither request is in the prefill regime.
	for _, pcp := range []int{2, 8} {
		if got := pcpKernel(t, pcpFixture, pcp).StepTime(withSkipped).NoOverlap; got != with {
			t.Errorf("pcp=%d repriced a batch it splits nothing in: %v against %v",
				pcp, got, with)
		}
	}
}

// A SLIDING-WINDOW layer's prefill must be split by WHERE this rank's tokens sit, not
// merely by how many it has.
//
// THE DEFECT THIS WAS WRITTEN AGAINST. prefillChunks carries `chunk{sched, prefix}`, which
// windowedCausalFLOPs reads POSITIONALLY: a query at chunk-local offset j attends
// min(prefix+j+1, window) keys. Crediting the split by replacing `sched` with the rank's
// token COUNT while leaving `prefix` at the request's own offset described the rank as one
// short run at the cheap start of the sequence. The zigzag gives it two runs, one near the
// start and one near the END, where every query reads a full window. Measured on the
// function: a 4,096-token chunk on no prefix at pcp 8 and window 2,176 was charged 524,288
// pair-units against a true 2,359,296 -- understated 4.5x.
//
// WHY IT NEEDS A WIDE WINDOW TO SHOW, which is why a first draft of this test missed it:
// the error needs the window to exceed the rank's token count. At window 128 a rank
// holding 512 tokens saturates almost everywhere regardless of position, so both readings
// agree to the digit -- and gpt-oss-120b's window IS 128, so the hybrid fixture the DCP
// tests use cannot see this at all.
func TestPCPSplitsAWindowedPrefillByPositionNotByCount(t *testing.T) {
	// THROUGH THE PRICER'S OWN PATH. appendPrefillRuns is what stepTime calls, and it
	// reads the split widths from the resolved layout -- so this exercises the same code
	// the step does rather than a free function beside it.
	k8 := pcpKernel(t, pcpFixture, 8)
	runs := k8.appendPrefillRuns(nil, 4096, 0)
	if len(runs) != 2 {
		t.Fatalf("a split prefill over 8 ranks should yield two runs, got %d: %+v",
			len(runs), runs)
	}
	lo, hi := runs[0], runs[1]
	if hi.prefix < 4096/2 {
		t.Errorf("the high run starts at %d, below the sequence midpoint; a zigzag rank "+
			"owns a chunk from the expensive end and that is what makes its queries "+
			"saturate a wide window", hi.prefix)
	}
	if lo.prefix+lo.sched > hi.prefix {
		t.Errorf("the runs overlap (%d+%d > %d)", lo.prefix, lo.sched, hi.prefix)
	}
	if held, want := lo.sched+hi.sched, pcpLocalTokens(4096, 8, 1); held != want {
		t.Errorf("the runs hold %d tokens, want the busiest rank's %d", held, want)
	}
	// Without the split, the runs are the request itself, so no existing deployment moves.
	if off := fixture(t, pcpFixture).appendPrefillRuns(nil, 4096, 13); len(off) != 1 ||
		off[0].sched != 4096 || off[0].prefix != 13 {
		t.Errorf("with pcp off the runs should be the request itself, got %+v", off)
	}

	// THE WINDOWED TERM, as two metamorphic relations rather than pinned figures. A window
	// WIDER than this rank's token count must cost more positionally than collapsed to the
	// start, because the high run then reads a full window; a window much NARROWER must
	// cost exactly the same, because it saturates wherever the tokens sit. Both sides are
	// computed from the kernel's own function, so a registry refit cannot break them.
	for _, pcp := range []int{2, 4, 8} {
		positional := pcpKernel(t, pcpFixture, pcp).appendPrefillRuns(nil, 4096, 0)
		held := 0
		for _, c := range positional {
			held += c.sched
		}
		flat := []chunk{{sched: held, prefix: 0}}
		if wide, wideFlat := windowedCausalFLOPs(positional, 2176),
			windowedCausalFLOPs(flat, 2176); wide <= wideFlat {
			t.Errorf("pcp=%d: a 2,176-token window costs %v positionally against %v by "+
				"token count; the high run must be charged for a full window",
				pcp, wide, wideFlat)
		}
		if narrow, narrowFlat := windowedCausalFLOPs(positional, 8),
			windowedCausalFLOPs(flat, 8); narrow != narrowFlat {
			t.Errorf("pcp=%d: an 8-token window costs %v positionally against %v by "+
				"token count; a window this narrow saturates wherever the tokens sit",
				pcp, narrow, narrowFlat)
		}
	}

	// COVERAGE LIMIT, STATED RATHER THAN PAPERED OVER. No sound relation on the STEP
	// separates the two readings. On the most windowed fixture committed -- minimax-m3,
	// 57 of 60 layers at a 2,176-token window -- the defect moves a 2,048-token prefill by
	// 5.1% of the SM term, because a dense GEMM dominates a windowed attention term
	// roughly five to one. Three step-level relations were tried and all three hold under
	// the defect as well as without it: a pinned ratio (also fitted to the current
	// registry, so unacceptable regardless), a sandwich between the unsplit figure and a
	// same-length contiguous chunk, and a sandwich between a saturating and an unbounded
	// window. A bound loose enough to survive a refit is wider than 5.1%.
	//
	// So the guard is the chunk geometry above, taken through the pricer's own method.
	// Closing the step-level gap needs a fixture whose windowed layers dominate its GEMMs,
	// which the catalog does not carry.
}
