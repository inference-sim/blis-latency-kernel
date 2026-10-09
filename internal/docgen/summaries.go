package main

// The plain-language heading and summary of each marked note, which the divergences page
// shows above the note itself. A note is keyed by its file, the declaration it is in, and
// its kind. generate fails on a note with no entry here and on an entry with no note, so
// adding, moving or removing a note means editing this table too.
//
// A summary says what the kernel does differently and in which deployments, in terms a
// reader of the Concepts pages knows. The note beside it keeps the evidence and the vLLM
// citations.
var summaries = map[string]struct{ title, summary string }{
	"internal/price/memory.go|KVBytesPerToken|Coverage limit": {
		"KV cache layouts not sized",
		"The kernel does not size every KV cache layout vLLM has. It refuses the per-token-head " +
			"quantized formats rather than size them, and it does not model padded pages or a " +
			"cache that compresses several tokens into one stored state.",
	},
	"internal/price/plan.go|ActivationBytes|Coverage limit": {
		"Expert dispatch priced at 16 bits",
		"The tokens sent to the experts are priced at 16 bits per element. On a quantized " +
			"deployment some expert kernels receive them at 8 or 4 bits, and the kernel cannot " +
			"tell which kernel serves a layer, so an fp8 deployment whose experts take quantized " +
			"input has its dispatch charged about twice its bytes.",
	},
	"kernel.go|(*Kernel).SequenceVariableBytes|Coverage limit": {
		"Hybrid attention under decode-context parallelism",
		"For a model that mixes sliding-window and full-attention layers, decode-context " +
			"parallelism does not shard the KV cache the kernel reports, which overstates it. vLLM " +
			"refuses such a deployment under its default KV cache manager, so this matters only " +
			"with that manager turned off.",
	},
	"kernel.go|(*Kernel).stepTime|Known over-charge": {
		"Routed experts under tensor parallelism",
		"Under tensor parallelism each rank is charged the arithmetic of whole routed experts, " +
			"where vLLM gives it a tp-th slice of each. Correcting this makes the measured accuracy " +
			"worse, because the over-charge offsets a cost not yet identified, so it is kept until " +
			"that cost is found.",
	},
	"kernel.go|(*Kernel).moeDPFunnel|Coverage limit": {
		"Expert tokens under prefill-context parallelism",
		"The tokens reaching a rank's experts are multiplied by the data-parallel width, " +
			"because vLLM gathers every data-parallel rank's tokens to them, but not by the " +
			"prefill-context width. How the experts gather a mix of split prefill tokens and " +
			"replicated decode tokens is not modeled.",
	},
	"new.go|(*Kernel).lift|Coverage limit": {
		"Expert weights under prefill-context parallelism",
		"With expert parallelism off and prefill-context parallelism on, each rank is " +
			"charged for a 1/(dp × tp) slice of each expert, where vLLM gives it 1/(dp × pcp × " +
			"tp). Dividing by pcp as well would leave out a reduction the kernel does not price, " +
			"and make the experts cheaper than the engine runs them.",
	},
	"sparse_mla.go|sparseMLABackend|Coverage limit": {
		"Choosing the sparse attention backend",
		"For DeepSeek sparse attention, which backend vLLM chooses depends on facts the model " +
			"graph does not state: whether the checkpoint quantizes its KV cache, the installed " +
			"FlashInfer version, and one head dimension. The kernel assumes the common case for " +
			"each, and the catalog has no chip of the SM120 generation, whose backend differs.",
	},
}
