# Upstream projects and citation

FlashRT-llama.cpp is the native ggml integration in the
[FlashRT](https://github.com/flashrt-project/FlashRT) project family. It
compiles into a pinned [llama.cpp](https://github.com/ggml-org/llama.cpp)
host; model loading, GGUF support, graph execution, KV management and
serving remain host responsibilities.

| Component | What this repository uses | Source and attribution |
|---|---|---|
| FlashRT | Selected NVFP4 GEMM and quantization kernels, plus the fusion/dataflow approach | [FlashRT](https://github.com/flashrt-project/FlashRT), Apache-2.0; the exact kernel revision and hashes are in [the vendor manifest](../vendor/FLASHRT_VENDOR.txt) |
| llama.cpp / ggml | The inference host, CUDA backend and arithmetic that the model-specific windows integrate with | [llama.cpp](https://github.com/ggml-org/llama.cpp), [MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE); revisions and patches are under [hosts/](../hosts/) |
| CUTLASS | NVIDIA CUDA templates used by the block-scaled GEMMs | [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), BSD-3-Clause; pinned as [a submodule](../.gitmodules), with its own license notices |
| FlashAttention-4 | Attention implementations used to generate the Thor AOT modules | [FlashAttention](https://github.com/Dao-AILab/flash-attention) and FlashRT's FA4 integration; [artifact provenance and regeneration](../fa4_aot/README.md) |
| Jetson-PI-Edge | The ggml-based π0 / π0.5 host integration | [Upstream project](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge) and [integration guide](../hosts/jetson-pi-edge/README.md) |

## Source ownership

- `src/` and `include/` contain this integration's native launch helpers and
  model-specific ggml windows.
- `vendor/csrc/` is a one-way snapshot of selected FlashRT sources. Make
  shared-kernel changes upstream in FlashRT, then use `vendor/sync.sh` and
  `vendor/verify.sh` to update and verify the snapshot.
- Host changes stay in `hosts/*/patches/` (or the Thor integration patch).
  Each host's `PIN` fixes the base that the patches apply to.
- `fa4_aot/` contains generated headers and objects, not the full FA4 source
  tree. Regeneration uses a FlashRT checkout supplied through `FLASHRT_DIR`.

The repository's [Apache-2.0 license](../LICENSE) applies to this integration;
it does not replace upstream licenses or notices.

## Citation

Use [CITATION.cff](../CITATION.cff) for citation metadata. The preferred
reference follows the FlashRT main repository:

```bibtex
@misc{su2026flashrtpai,
  title={FlashRT-PAI: Fast Kernels Are Not Fast Pipelines for Physical AI Inference},
  author={Liang Su and David Chen and Jason Li and Kevin Guo and Teresa Lv},
  year={2026},
  eprint={2610.24349},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2610.24349},
  url={https://arxiv.org/abs/2610.24349},
}
```

Also link to [llama.cpp](https://github.com/ggml-org/llama.cpp) as the host
engine. Model weights and model architectures remain the work of their
respective authors.
