# Supported platforms

| Host | Resident target and attention | Routed experts |
|---|---|---|
| Apple Silicon macOS | MPS | Metal, with native CPU fallback |
| NVIDIA Linux x86-64/aarch64 | CUDA | qualified native CUDA MXFP4, with native CPU fallback |
| AMD Linux x86-64 (ROCm/HIP) | HIP, reported as CUDA devices | native CPU MXFP4 by default; opt-in HIP MXFP4 kernels with no hardware evidence |
| Linux x86-64 | CPU | SSSE3/AVX/FMA3 compatibility kernel, with runtime-selected AVX2 acceleration |
| Linux aarch64 | CPU | native NEON MXFP4 |
| Windows x86-64 (CUDA) | CUDA | native CUDA MXFP4, with native CPU fallback |
| Windows x86-64 (Intel XPU) | XPU (`at::kXPU`) | native SYCL MXFP4 expert kernel, with native CPU fallback |
| Windows x86-64 (CPU) | CPU | SSSE3/AVX/FMA3 compatibility kernel, with runtime-selected AVX2 acceleration |

ROCm builds of PyTorch keep the `torch_cuda`/`c10_cuda` library names and the `at::kCUDA` device type, so Deltafin identifies them by their HIP ELF dependency and reports them as CUDA devices. Resident work and the int8 spine run on the GPU through ATen and need no Deltafin kernel.

The device kernels compile for HIP. The MXFP4 expert kernels reduce through shared memory and never read a wavefront lane, so their FP32 addition order is identical at 32 and 64 lanes; the original-BF16 spine kernel reduces against the code object's own `warpSize`, which changes its addition tree between wave32 and wave64 while keeping every 32-lane device bit-for-bit unchanged. Neither is bit-identical to NVIDIA in any case, because the SiTU activation's `tanhf`/`expf` resolve to AMD's OCML rather than CUDA libdevice — the same cross-vendor difference the CPU and Metal paths already carry, which the token oracle rather than bit equality is there to catch.

What does not exist is evidence. No AMD device has executed these kernels: no token-oracle run, no timing arm, no physical residency or fault trace. `DELTAFIN_CUDA_MOE=ON` builds them on a ROCm root for whoever runs those gates; an unset or `AUTO` setting keeps the exact CPU MXFP4 path. Because the reproducible upgrade profile describes an NVCC build, `deltafin upgrade` refuses a HIP kernel build rather than rebuilding it as a different configuration.

The Intel XPU provider runs the resident target and attention on the integrated GPU through PyTorch's `at::kXPU` backend, and the routed experts through a self-contained SYCL kernel DLL (`tools/xpu_moe_kernels.cpp`, built by the DPC++ toolchain into `deltafin_xpu_kernels.dll`). The MXFP4 expert GEMV and the row-quantized int8 projections are SYCL kernels; the expert arena is allocated as USM shared so the storage layer reads disk straight into device-visible memory with no host->device copy. It is selected with `--device xpu` (or `K3_DEV=xpu`) and needs `DELTAFIN_TORCH_ROOT` (an XPU LibTorch root), `DELTAFIN_SYCL_ROOT`, and `DELTAFIN_XPU_RUNTIME`. On an integrated GPU that shares system memory the provider is bandwidth-bound and, on the current dev box, decodes about 1.3x slower than the CPU; its value is keeping the working set out of a discrete GPU's 16 GB VRAM. There is maintainer hardware evidence for the dev-box iGPU only (Intel Graphics DEV_7D67, Xe2); no discrete Intel Arc evidence.

Platform support describes implemented selection and safety gates, not equal benchmark coverage. Apple Silicon has maintainer-run full-model evidence; a DGX Spark community run established Linux/CUDA viability; additional physical CUDA and native Linux performance reports remain welcome. The compiled CPU fallback stays available when a GPU-specific expert provider does not qualify. Windows x86-64 is implemented for CPU and CUDA, but has no maintainer hardware evidence yet: its CUDA LibTorch root is supplied explicitly through `DELTAFIN_TORCH_ROOT` (or `LIBTORCH`) with a matching CUDA 12.6/13.x toolkit, and the bootstrap pins only the CPU wheel.
