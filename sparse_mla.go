package latencykernel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// Which attention backend serves a DSA sparse-MLA layer, as vLLM v0.31.0 chooses it, and
// what that choice decides about the deployment: the cache layout the engine allocates, the
// block size it runs, and the context-parallel layouts it refuses.
//
// SCOPE. This is the DSA family -- DeepSeek-V3.2 and GLM-5, served by DeepseekV32Attention
// (vllm/models/deepseek_v32/attention.py) -- which the catalog states as a sparse_mla layer
// with a 576-wide head (kv_lora_rank 512 plus a 64-wide rope; MLAAttention sets head_size
// to that sum, vllm/model_executor/layers/attention/mla_attention.py:479). DeepSeek-V4's
// sparse_mla layers, 512 wide, are served by that model's own backends
// (vllm/models/deepseek_v4/sparse_mla.py), which this does not describe.
//
// PARTS. The catalog's parts that run a sparse-MLA backend at all are Hopper (SM90) and
// data-center Blackwell (SM100): every sparse backend v0.31.0 ships requires one of the two
// (supports_compute_capability in vllm/v1/attention/backends/mla/flashattn_mla_sparse.py,
// flashmla_sparse.py, flashinfer_mla_sparse.py, flashinfer_mla_sparse_sm90.py). The catalog
// records no compute capability, so data-center Blackwell is recognised, as elsewhere in
// this kernel, by native NVFP4 support, and every other part is treated as Hopper. The
// pre-Hopper parts (A100, L40S) have no sparse-MLA backend; a deployment on them is one the
// engine refuses, and pricing it here is not a claim that it runs.

// dsaHeadDim is the head width that marks a DSA sparse-MLA layer: kv_lora_rank 512 plus
// qk_rope_head_dim 64.
const dsaHeadDim = 576

// dsaBlockSize is the block every DSA deployment runs on CUDA. The indexer's backend
// accepts exactly 64 (DeepseekV32IndexerBackend.get_supported_kernel_block_sizes,
// vllm/v1/attention/backends/mla/indexer.py:203-204), so with no stated size the platform
// picks 64 (update_block_size_for_backend, vllm/platforms/interface.py:666-689, through
// _preferred_block_size_for_backends, :616-656), and a stated size 64 does not divide
// leaves no common kernel block (select_common_block_size, vllm/v1/worker/utils.py:330-391,
// "No common block size").
const dsaBlockSize = 64

// isDSA reports whether a graph's attention is the DSA family's sparse MLA.
func isDSA(g *model.Graph) bool {
	if g == nil {
		return false
	}
	_, headDim, _, kind := kvGeometry(g)
	return kind == model.AttentionSparseMLA && headDim == dsaHeadDim
}

// The sparse-MLA backends v0.31.0 ships for SM90 and SM100, by AttentionBackendEnum name.
const (
	backendFlashAttnMLASparse  = "FLASH_ATTN_MLA_SPARSE"
	backendFlashMLASparse      = "FLASHMLA_SPARSE"
	backendFlashInferMLASparse = "FLASHINFER_MLA_SPARSE"
	backendFlashInferSM90      = "FLASHINFER_MLA_SPARSE_SM90"
)

// sparseMLABackendRequest is what backend selection reads.
type sparseMLABackendRequest struct {
	cache        string // the stated cache dtype; "" is "auto"
	headsPerRank int    // query heads per tensor-parallel rank
	sm100        bool
	dcp, pcp     int
	dcpComm      string // the resolved DCP combine backend
}

// sparseMLABackend returns the backend v0.31.0 runs a DSA layer on and the cache layout that
// backend allocates, or an error naming why the engine would not start.
//
// SELECTION is the platform's priority list, each candidate filtered by its own checks
// (CudaPlatform.get_valid_backends and validate_configuration; the lists at
// vllm/platforms/cuda.py:95-153). For a DSA head the lists are:
//
//   - SM100: FlashInfer then FlashMLA-sparse for a quantized cache, and for a 16-bit one
//     when a rank holds at most 16 heads; FlashMLA-sparse first otherwise.
//   - SM90: FlashAttention's sparse MLA, FlashMLA-sparse, FlashInfer's SM90 sparse MLA.
//
// and a candidate is skipped where it does not list the cache dtype, or the layout uses a
// context parallelism it does not support:
//
//   - FlashAttention sparse: auto, float16, bfloat16 (flashattn_mla_sparse.py:36-40); no DCP
//     (:99-100).
//   - FlashMLA-sparse: auto, bfloat16, fp8_ds_mla, "fp8" as an alias for it, nvfp4_ds_mla
//     (flashmla_sparse.py:133-139); supports DCP (:680).
//   - FlashInfer (SM100): auto, float16, bfloat16, fp8, fp8_e4m3, and not fp8_ds_mla
//     (flashinfer_mla_sparse.py:76-82, :126-129); not PCP with DCP (:117-125).
//   - FlashInfer SM90: auto, bfloat16, fp8, fp8_e4m3 (flashinfer_mla_sparse_sm90.py:141-146);
//     no DCP (its implementation inherits supports_dcp = False,
//     vllm/v1/attention/backend.py:845, checked at :345-346).
//
// LAYOUT. FlashMLA-sparse serves any quantized cache other than nvfp4_ds_mla as fp8_ds_mla
// (_canonicalize_sparse_mla_kv_cache_dtype, mla_attention.py:358-375), which is a 656-byte
// cell rather than the 576 bytes "fp8" would hold.
//
// FLASHMLA-SPARSE'S OWN DCP CHECKS run after it is chosen, so they refuse the layout rather
// than pass it to another backend (FlashMLASparseMetadataBuilder.__init__,
// flashmla_sparse.py:409-435): DCP must combine with ag_rs; with PCP the cache must be
// fp8_ds_mla, the format its gather upconverts; and without PCP a rank must hold fewer than
// 32 heads (MIN_HEADS_FOR_BF16_PREFILL, :73), the mixed-batch path that returns a
// log-sum-exp for every row.
//
// COVERAGE LIMITS: FlashInfer's SM90 backend needs FlashInfer 0.6.18 or later
// (flashinfer_mla_sparse_sm90.py:194-199), which this assumes is installed; FlashInfer
// (SM100) checks the model's qk_nope_head_dim, which the DSA family satisfies (128 and 192, within its [128, 192]) and
// the graph does not state; and the SM120 parts' backend, which repacks "auto" too, has no
// part in the catalog.
func sparseMLABackend(r sparseMLABackendRequest) (backend, layout string, err error) {
	cache := r.cache
	if cache == "" {
		cache = "auto"
	}
	quantized := strings.HasPrefix(cache, "fp8") || strings.HasPrefix(cache, "nvfp4") ||
		strings.HasSuffix(cache, "per_token_head")
	lists := func(dtypes ...string) bool {
		for _, d := range dtypes {
			if d == cache {
				return true
			}
		}
		return false
	}
	accepts := map[string]bool{
		backendFlashAttnMLASparse: r.dcp <= 1 && lists("auto", "float16", "bfloat16"),
		backendFlashMLASparse:     lists("auto", "bfloat16", "fp8_ds_mla", "fp8", "nvfp4_ds_mla"),
		backendFlashInferMLASparse: !(r.pcp > 1 && r.dcp > 1) &&
			lists("auto", "float16", "bfloat16", "fp8", "fp8_e4m3"),
		backendFlashInferSM90: r.dcp <= 1 && lists("auto", "bfloat16", "fp8", "fp8_e4m3"),
	}
	var order []string
	switch {
	case !r.sm100:
		order = []string{backendFlashAttnMLASparse, backendFlashMLASparse, backendFlashInferSM90}
	case quantized || r.headsPerRank <= 16:
		order = []string{backendFlashInferMLASparse, backendFlashMLASparse}
	default:
		order = []string{backendFlashMLASparse, backendFlashInferMLASparse}
	}
	for _, b := range order {
		if accepts[b] {
			backend = b
			break
		}
	}
	if backend == "" {
		return "", "", fmt.Errorf("no sparse-MLA backend in vLLM v0.31.0 serves a %q cache "+
			"with dcp %d, pcp %d on this part (candidates %s): the engine refuses this "+
			"layout at startup", cache, r.dcp, r.pcp, strings.Join(order, ", "))
	}
	layout = cache
	if backend == backendFlashMLASparse && quantized && cache != "nvfp4_ds_mla" {
		layout = "fp8_ds_mla"
	}
	if backend == backendFlashMLASparse && r.dcp > 1 {
		switch {
		case r.dcpComm != resolve.DCPAllGatherReduceScatter:
			return "", "", fmt.Errorf("%s runs decode-context parallelism only with the %q "+
				"combine, and %q is resolved: the engine refuses this layout at startup",
				backend, resolve.DCPAllGatherReduceScatter, r.dcpComm)
		case r.pcp > 1 && layout != "fp8_ds_mla":
			return "", "", fmt.Errorf("%s runs prefill- with decode-context parallelism "+
				"only on an fp8_ds_mla cache, and this one is %q: the engine refuses this "+
				"layout at startup", backend, layout)
		case r.pcp <= 1 && r.headsPerRank >= 32:
			return "", "", fmt.Errorf("%s runs decode-context parallelism only with fewer "+
				"than 32 heads a rank, and this layout puts %d on each: the engine refuses "+
				"this layout at startup", backend, r.headsPerRank)
		}
	}
	return backend, layout, nil
}
