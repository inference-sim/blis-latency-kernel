package resolve

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/spec/deployment"
)

// The decode-context-parallel combine backends this kernel prices. They are the engine's
// DCPCommBackend literal at v0.31.0 (vllm/config/parallel.py:40), and each launches a
// different set of collectives per decode layer; see the DCP block in stepTime.
const (
	// DCPAllGatherReduceScatter is the engine default: a log-sum-exp all-gather, then an
	// output reduce-scatter (vllm/v1/attention/ops/dcp.py:438-500).
	DCPAllGatherReduceScatter = "ag_rs"
	// DCPAllToAll packs the partial output and its log-sum-exp into one buffer and
	// exchanges it with a single all-to-all (dcp.py:939-1010).
	DCPAllToAll = "a2a"
)

// DecodeContext is how a decode-context-parallel group will run, after resolution.
type DecodeContext struct {
	// CommBackend is the combine that runs: DCPAllGatherReduceScatter or DCPAllToAll.
	CommBackend string
	// Interleave is how many consecutive tokens one rank holds before the next takes
	// over (cp_kv_cache_interleave_size), as it will run.
	Interleave int
}

// ResolveDecodeContext settles the three decode-context-parallel engine knobs that blis-schemas
// v0.2.2 made expressible, the way vLLM v0.31.0 settles them.
//
// accepted is the set of backend names the scenario's engine release accepts, from its
// rules pack; nil means the caller has no pack to consult, and only the names this kernel
// can price are checked. blockSize is the KV block size the kernel resolved.
//
// What it does NOT settle is the model's own preference. An unstated backend is the stock
// default here, ag_rs, but a model's configuration hook may choose otherwise before the
// stock default applies (ParallelConfig.set_dcp_defaults, parallel.py:581-594, called from
// vllm/config/vllm.py:1434-1440) -- GlmMoeDsaForCausalLM selects a2a with a replicated query
// projection (vllm/model_executor/models/config.py:43-50). The kernel holds no model
// identity to dispatch on, so a deployment of such a model must state the backend it runs;
// the caller records the default as an assumption so the gap is visible.
func ResolveDecodeContext(pool deployment.Pool, dep *deployment.Deployment, blockSize int,
	accepted map[string]bool) (DecodeContext, []Override, error) {
	e := pool.Engine
	dcp, pcp := pool.Parallel.DCP, pool.Parallel.PCP
	out := DecodeContext{CommBackend: DCPAllGatherReduceScatter, Interleave: 1}

	// The backend is a closed literal the engine validates whatever the DCP width, so an
	// unknown name is refused even where it would have no effect.
	if b := e.DCPCommBackend; b != "" {
		if accepted != nil && !accepted[b] {
			return DecodeContext{}, nil, fmt.Errorf(
				"dcp_comm_backend %q is not one this engine release accepts (%s)",
				b, strings.Join(sortedNames(accepted), ", "))
		}
		if b != DCPAllGatherReduceScatter && b != DCPAllToAll {
			return DecodeContext{}, nil, fmt.Errorf(
				"dcp_comm_backend %q is not one this kernel prices; add its collectives "+
					"to the DCP combine rather than letting it fall through", b)
		}
		out.CommBackend = b
	}
	if dcp <= 1 {
		// Nothing is sharded, so neither the combine nor the stripe exists.
		return out, nil, nil
	}
	// Where PCP with DCP runs at all (the DSA layers that opt in; New refuses the rest), the
	// combine is an all-gather and an all-reduce, and the engine refuses the all-to-all
	// there outright: "MRV2 PCP + DCP requires dcp_comm_backend='ag_rs'"
	// (vllm/v1/worker/gpu/pcp_manager.py:188-194).
	if pcp > 1 && out.CommBackend == DCPAllToAll {
		return DecodeContext{}, nil, fmt.Errorf(
			"dcp_comm_backend a2a with pcp %d: the engine requires ag_rs when prefill- "+
				"and decode-context parallelism are combined", pcp)
	}

	var overrides []Override
	switch stated := e.CPKVCacheInterleaveSize; {
	case stated > 0:
		// Stated: v0.31.0 honours it, NIXL or not. Only an unstated size is auto-resolved
		// (_allow_auto_resolve_cp_interleave_size is set exactly when the flag is omitted,
		// vllm/engine/arg_utils.py:2509-2516).
		out.Interleave = stated
	default:
		pinned, err := nixlPinsInterleave(dep)
		if err != nil {
			return DecodeContext{}, nil, err
		}
		if pinned && blockSize > 1 {
			// With NIXL configured, the engine sets an unstated size to the LOCAL block
			// size -- deliberately not the dcp-scaled one -- for block-level alignment
			// across the transfer (vllm/config/vllm.py:3333-3378). Recorded, because a
			// reader of the deployment sees no interleave at all.
			out.Interleave = blockSize
			overrides = append(overrides, Override{
				Field:     "cp_kv_cache_interleave_size",
				Requested: "unstated (engine default 1)",
				Resolved:  fmt.Sprint(blockSize),
				Reason: "with NixlConnector configured and the size unstated, the " +
					"engine aligns the decode-context stripe to the KV block so " +
					"transferred blocks stay whole",
			})
		}
	}
	return out, overrides, nil
}

// nixlPinsInterleave reports whether the deployment configures the NIXL connector, which
// is the only one that auto-resolves the interleave at v0.31.0 (vllm/config/vllm.py:3356-3360
// returns early unless kv_transfer_config.has_connector("NixlConnector")).
//
// A deployment names a connector for prefill-to-decode transfer and one for offload, and
// the engine's has_connector also looks inside a MultiConnector's children
// (vllm/config/kv_transfer.py:156-163). Those children are not expressible in a deployment,
// so a MultiConnector leaves the answer unknown, and that is an error rather than a guess:
// guessing 1 or the block size misstates a rank's share of every sequence.
func nixlPinsInterleave(dep *deployment.Deployment) (bool, error) {
	if dep == nil {
		return false, nil
	}
	var names []string
	if dep.PDTransfer != nil {
		names = append(names, dep.PDTransfer.Connector)
	}
	if dep.Offload != nil {
		names = append(names, dep.Offload.Connector)
	}
	pinned := false
	for _, n := range names {
		switch n {
		case "NixlConnector":
			pinned = true
		case "MultiConnector":
			return false, fmt.Errorf(
				"cp_kv_cache_interleave_size is unstated under a MultiConnector, whose " +
					"children a deployment cannot name; the engine pins the size to the " +
					"block size only if one of them is NixlConnector, so state the size")
		}
	}
	return pinned, nil
}

func sortedNames(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
