package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// A term the kernel cannot price from its inputs is refused or recorded, never charged at
// nothing in silence.

// dropEntries returns sets without the entries whose names start with any prefix.
func dropEntries(sets []*coefficient.Set, prefixes ...string) []*coefficient.Set {
	out := make([]*coefficient.Set, len(sets))
	for i, s := range sets {
		c := *s
		c.Coefficients = nil
		for _, e := range s.Coefficients {
			keep := true
			for _, p := range prefixes {
				keep = keep && !strings.HasPrefix(e.Name, p)
			}
			if keep {
				c.Coefficients = append(c.Coefficients, e)
			}
		}
		out[i] = &c
	}
	return out
}

// recorded reports whether a kernel's provenance names an entry.
func recorded(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name {
			return true
		}
	}
	return false
}

// With no measured attention form, a scorer's scan is still a read the step pays for. The
// scan grows with context where the index store, which writes per scheduled token, does
// not: so a larger index record widens the gap between a long-context and a short-context
// decode's HBM time.
func TestAScorerWithNoMeasuredFormIsStillCharged(t *testing.T) {
	gap := func(scale int) float64 {
		in := fixtureInputs(t, v41)
		in.Coefficients = dropEntries(in.Coefficients, "attention_decode_")
		// Owners and readers alike, as a reader's record must match its owner's.
		editNodes(in.Model, role("block_index_scores"), func(n *model.Node) {
			for c := range n.Caches {
				if n.Caches[c].Kind == model.CacheIndex {
					n.Caches[c].BytesPerEntry *= scale
				}
			}
		})
		k := build(t, in)
		hbm := func(ctx int) float64 {
			return k.StepTime(decodeBatch(64, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		return hbm(65536) - hbm(8192)
	}
	if small, large := gap(1), gap(4); large <= small {
		t.Errorf("a 4x index record widens the long-context HBM gap to %.3g s, no more "+
			"than %.3g: the scan is not charged", large, small)
	}
}

// A host-resident table with no rate to cross the host link at is refused.
func TestAHostTableWithNoLinkRateIsRefused(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Coefficients = dropEntries(in.Coefficients, "host_gather_rate_uva", "host_link_bandwidth")
	if _, err := New(in); err == nil {
		t.Error("an Engram table in host memory was priced with no host link rate")
	}
}

// A scorer the registry has no form for on this part is priced with its kind's pair, and
// says so.
func TestAScorerWithoutItsOwnRateIsRecorded(t *testing.T) {
	in := fixtureInputs(t, v41)
	in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, "b200"), "b200"
	if k := build(t, in); !recorded(k, "attention_decode_rate_block_index_scores") {
		t.Error("the indexer on B200 has no rate of its own and Provenance does not say so")
	}
}

// The DeepSeek-V4.1 layouts vLLM v0.31.0 refuses at startup are refused, not priced.
func TestV41LayoutsTheEngineRefuses(t *testing.T) {
	for name, edit := range map[string]func(*Inputs){
		"a bf16 cache": func(in *Inputs) { in.Deployment.Pools[0].Engine.CacheDType = "bfloat16" },
		"mega-MoE without expert parallelism": func(in *Inputs) {
			in.Deployment.Pools[0].Engine.MoEBackend = "deep_gemm_mega_moe"
		},
		"dual-batch overlap with Engram": func(in *Inputs) {
			in.Deployment.Pools[0].Engine.DBO = &deployment.DBO{Enabled: true}
		},
		"MTP drafting": func(in *Inputs) {
			in.Deployment.Pools[0].Engine.Speculative.Method = "mtp"
		},
	} {
		in := fixtureInputs(t, v41)
		edit(&in)
		if _, err := New(in); err == nil {
			t.Errorf("%s: priced a layout vLLM v0.31.0 refuses at startup", name)
		}
	}
}

// A sparse layer gathers its window and top-k for every query token, so the context it reads
// grows once per verify token, where a dense layer reads its shared prefix once. Measured as
// how much more HBM time a longer context costs -- below the top-k, where a sparse read
// still grows -- which per-token terms that do not read context leave untouched.
func TestSparseReadsArePerQueryToken(t *testing.T) {
	k := fixture(t, "glm5-h200-tp8.yaml") // DSA: top-k 2048, read per query
	gap := func(q int) float64 {
		hbm := func(ctx int) float64 {
			return k.StepTime(decodeBatch(32, q, ctx)).PerResource[kernel.ResourceHBM].Seconds()
		}
		return hbm(2000) - hbm(1000)
	}
	if one, six := gap(1), gap(6); six < 3*one {
		t.Errorf("a 6-token verify's context costs %.3g s of HBM against a 1-token decode's "+
			"%.3g; six queries each gather their own selection", six, one)
	}
}
