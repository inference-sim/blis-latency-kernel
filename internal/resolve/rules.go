package resolve

// EngineRules is the version-specific behaviour a resolver needs.
//
// It is an interface rather than a set of constants because these facts belong to one
// engine release. A scenario names its version; the caller supplies the matching rules.
// Compiling them in would mean this package churns whenever an engine ships, and would
// silently misjudge a scenario pinned to an older release.
type EngineRules interface {
	// SequenceParallelMoE reports whether the engine makes the MoE input
	// sequence-parallel, which replaces a layer's reduction with a different pair of
	// collectives. It depends on the backend and on both parallel widths.
	SequenceParallelMoE(backend string, expertParallel bool, tp, dp int) bool

	// CustomAllReduceSupportsWidth reports whether the SM-consuming reduction kernel
	// handles a given rank count.
	CustomAllReduceSupportsWidth(tp int) bool

	// TritonFetchWithholdsSMs reports how many SMs a concurrent offload fetch takes from
	// the forward pass, given the page size it moves. Zero when the fetch uses a copy
	// engine instead.
	TritonFetchWithholdsSMs(pageBytes int) int

	// DBOEngages reports whether dual-batch overlap splits a batch, given its total
	// token count and whether every request schedules the same number of tokens.
	DBOEngages(enabled bool, totalTokens int, uniformDecode bool) bool
}
