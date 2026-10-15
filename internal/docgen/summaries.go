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
			"quantized formats rather than size them. Padded pages and a cache that compresses " +
			"several tokens into one stored state are sized only where the model graph states " +
			"its caches; a graph that does not is sized as one full-length cache per layer.",
	},
	"internal/price/plan.go|ActivationBytes|Coverage limit": {
		"Expert dispatch priced at 16 bits",
		"The tokens sent to the experts are priced at 16 bits per element. On a quantized " +
			"deployment some expert kernels receive them at 8 or 4 bits, and the kernel cannot " +
			"tell which kernel serves a layer, so an fp8 deployment whose experts take quantized " +
			"input has its dispatch charged about twice its bytes.",
	},
	"internal/price/plan.go|planNodes|Coverage limit": {
		"Every expert priced as gated",
		"Each expert is priced as three matrices, a gated MLP. An expert with no gate, such as " +
			"Nemotron-H's relu2 experts, holds two, and blis-schemas has no field saying which an " +
			"expert is, so such a model's expert weights and arithmetic are over-stated by half.",
	},
	"new.go|(*Kernel).engineBufferBytes|Known over-charge": {
		"DeepSeek-V4.1's prefill workspace beside the activation scratch",
		"DeepSeek-V4.1's prefill gather workspace and its indexer's logits stand-in are charged " +
			"in full beside the registry's activation scratch. vLLM holds the workspace in one " +
			"shared scratch buffer that its MoE kernels also grow, and the stand-in is a " +
			"transient at the profile peak, so the engine reserves the larger of each pair " +
			"where the kernel charges their sum.",
	},
	"caches.go|(*Kernel).maxInFlightTokens|Coverage limit": {
		"Window admission under pipeline parallelism",
		"A window cache's admission reservation is sized for one more batch in flight than " +
			"the pipeline-parallel width, as vLLM's default model runner holds under " +
			"asynchronous scheduling. Its older runner holds one fewer with pipeline " +
			"parallelism on, and the kernel cannot see which runs, so such a deployment's " +
			"SequenceAdmissionBytes is up to one token budget a window group high.",
	},
	"draft.go|(*Kernel).priceDraft|Coverage limit": {
		"Draft passes not padded under data-parallel lockstep",
		"A data-parallel rank's pad tokens run the target model's forward but not the " +
			"drafter's passes. vLLM pads the draft passes across the group too, to a count set " +
			"by every rank's draft batch, which one rank's batch does not state, so a padded " +
			"rank's draft work is under-stated by its pad.",
	},
	"kernel.go|(*Kernel).PreSnapshotBytes|Known divergence": {
		"The engine workspace is all outside the memory budget",
		"The registry's engine workspace is one lumped allowance for the CUDA context, cuBLAS " +
			"and allocator slack, and the kernel reports all of it outside the " +
			"gpu_memory_utilization budget, with the communicator. vLLM charges the part that " +
			"arises after its memory snapshot, library workspaces made at the first forward, " +
			"against the KV cache, which the kernel leaves out.",
	},
	"kernel.go|(*Kernel).pdTransferBytes|Coverage limit": {
		"P/D handoffs between pools of different widths",
		"Each decode rank's pull is priced as one rank's transfer at one rank's link rate. That " +
			"is exact when the prefill and decode pools have the same tensor-parallel width. Where " +
			"the decode pool is wider, several decode ranks pull from one prefill rank, whose link " +
			"then carries several payloads, and a placement that names a node rather than a rank " +
			"cannot express that.",
	},
	"kernel.go|(*Kernel).SequenceVariableBytes|Coverage limit": {
		"Hybrid attention under decode-context parallelism",
		"For a model that mixes sliding-window and full-attention layers, decode-context " +
			"parallelism does not shard the KV cache the kernel reports, which overstates it. vLLM " +
			"refuses such a deployment under its default KV cache manager, so this matters only " +
			"with that manager turned off.",
	},
	"step.go|(*Kernel).priceLayers|Known over-charge": {
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
