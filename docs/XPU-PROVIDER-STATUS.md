# XPU provider — status / handoff

Branch: `feat/xpu-provider` (based on `feat/windows-cuda`, PR #35). Goal: run K3
on the Intel iGPU via PyTorch's `at::kXPU` backend so the working set stays in
shared system memory instead of streaming over PCIe from a 16 GB discrete GPU.

## Where it stands

Committed (HEAD = `e38c0d0`):

- `4b619cf` Phase 1 — XPU device plumbing + ATen-XPU attention. `doctor`
  reports `xpu_devices=1`, every XPU canary PASS, and `run --device xpu`
  generates text. Experts used the exact CPU MXFP4 fallback.
- `99877a5` — SYCL kernel DLL pipeline. `clang++ -fsycl -shared` builds
  `deltafin_xpu_kernels.dll`, which self-registers its device image on load, so
  rustc/`link.exe` can link it (a bare `-c` object does not register).
- `ff43e55` — MXFP4 GEMV ported to SYCL on USM **shared** weights (host fills,
  kernel reads the same pointer; zero copy). Verified vs a CPU reference
  (max relative error 7.2e-07).
- `21f5c28` — `MoeExpertBackend::XpuMxfp4` + `execute_xpu`, wired into
  `select_backend` and the three routed-MoE switches.
- `e38c0d0` — crash fix: the prefill position path is a **scattered** tile
  (`expert_span_pointers`, empty `expert_major_bytes`); `execute_xpu_positions`
  assumed contiguous, so pointers became bare `slot*span` and the null slot
  crashed. Prefill now also goes through the XPU backend.

Uncommitted working-tree changes (2026-10-10) — the speed pass:

- `tools/xpu_moe_kernels.cpp`: the batched MXFP4 expert kernel now runs **one
  work-group per output row** (32 work-items, each striding the 32-wide scale
  groups), takes the group scale with a single `ldexp` per 32 weights, and
  decodes e2m1 from a table. The stashed serial-per-row loop called `ldexp` per
  element. Standalone: **57 → 13 ms/dispatch (4.4×)**, max rel error 1.3e-4 vs a
  scalar reference.
- New `k3_xpu_int8_gemv`: row-quantized int8 GEMV that runs on **torch_xpu's own
  stream** (the `sycl::queue*` is passed in, so it is ordered with the ATen ops
  around it) and does **not** wait on the host — the in-order stream keeps it
  ordered and the GPU pipelines. ATen's `_weight_int8pack_mm` has a **~6 ms
  fixed per-op overhead** on the iGPU regardless of shape; the SYCL kernel is
  **0.5 ms for 1024×7168 (12×)** and reads the torch_xpu USM allocations in
  place, with no copy.
- Wired into every resident int8 projection via `xpu_int8_linear` in
  `provider_xpu.cpp`: `provider_mla.cpp`, `provider_kda.cpp`,
  `provider_kda_batch.cpp`, `provider_target.cpp`, `provider_moe.cpp`. The ATen
  op stays the fallback off XPU or for non-contiguous tensors.
- Zero-copy expert arena (see the dedicated section below): the reader fills a
  USM shared slab straight from disk and the expert kernel reads it in place,
  removing the 280 MB/layer host→USM memcpy.

## Measured (dev box: Intel Graphics DEV_7D67, ~30 GB/s host memory)

Decode steady state, `run --device xpu --stats`, Qwen + int8 spine, France
prompt, 93 layers:

- Before the speed pass: first token ~497 s, steady ~25 s/token
  (attention ~16.5, expert_kernel ~6.1, expert_read ~3.2).
- After the int8 pass: first token ~100 s, steady ~14.3 s/token
  (attention ~4.8, expert_kernel ~6.1, expert_read ~3.2).
- After the zero-copy pass: first token ~82 s, steady **~12.5 s/token**
  (attention ~5.4, expert_read ~3.2, expert_kernel ~3.85).
- After the async-int8 pass: first token ~84 s, steady **~11.5 s/token**
  (attention ~7.0, expert_read ~2.0, expert_kernel ~2.35). Submitting the int8
  projections on torch_xpu's in-order stream without a host wait lets the
  torch-stream attention overlap the internal expert-kernel queue; the phase
  split shifts but the wall drops ~1 s.
- Output parity with CPU: `" Paris. The E"` → `" Paris. The Eiffel Tower"`.

CPU baseline on the same box and request: first token ~63 s, steady
**~9 s/token** (attention ~2.9, expert_read ~3.1, expert_kernel ~2.8).

Both paths are phase-serial: CPU phases sum to 8.8 s and wall is 8.8 s; XPU
phases sum to 12.45 s and wall is 12.5 s. **There is no separate "overlap" lever
left:** the disk already overlaps compute — 64.7 GB/token of reads at 2.16 GB/s
is ~30 s of I/O inside a 12.5 s/token wall, hidden by the prefetch reader. The
remaining gap is pure per-phase throughput: attention (5.4 vs 2.9) and the
expert kernel (3.85 vs 2.8), both memory-streaming-bound at ~13 GB/s against a
~33 GB/s copy ceiling.

## Zero-copy expert arena (2026-10-10)

The reader's arena slabs are now allocated through the provider as USM shared on
XPU, so the storage layer reads disk straight into device-visible memory and the
SYCL expert kernel reads the same pointer. The 280 MB/layer host→USM memcpy is
gone (measured 2.2 s/token).

- `tools/xpu_moe_kernels.cpp`: `k3_xpu_alloc_shared_aligned`.
- Provider ABI `deltafin_provider_alloc_unified_v1` / `_free_unified_v1`
  (`provider_abi.h`, `provider_runtime.cpp`, `provider_xpu.cpp`).
- Rust: `SessionInner::allocate_unified`/`release_unified` (`provider.rs`);
  `UnifiedBufferHooks` threaded through `AlignedBuffer`/`SharedBuffers`/
  `BufferArena`/`Reader` (`storage.rs`), default `None` so every other device is
  byte-for-byte unchanged; enabled in `engine.rs` for XPU on the **demand and
  prefetch** readers.
- The host **pin tier is disabled on XPU**: every expert byte source (demand,
  prefetch, pin) must be USM shared, and the pin tier allocates host pages.
  Mixing them crashed the kernel with `UR_RESULT_ERROR_DEVICE_LOST` on the
  scattered prefetch-hit path.

## Next steps

1. **Attention throughput.** Still the largest phase. Instrumentation shows
   **~1000 int8 GEMVs/token and ~44 GB of int8 weights/token**; at the measured
   ~12 GB/s that is ~3.5 s/token, so roughly half the attention phase is int8
   streaming and half is the MLA/KDA elementwise, einsum, softmax and recurrent
   ops. The int8 GEMV plateaus at ~10-12 GB/s across work-group sizes (8..256)
   and both scalar and 4-wide-unrolled inner loops, against a ~33 GB/s copy
   ceiling — it is not load-width-bound, so it is memory-latency/occupancy-bound
   on this iGPU. Candidates: a transposed/batched weight layout, and fusing the
   MLA/KDA decode into fewer SYCL kernels.
2. **Expert-kernel throughput.** ~13 GB/s against the same ceiling. A
   rows-per-work-item variant that reuses the shared activation across rows,
   and vectorised nibble decode, are the candidates. Under sustained load the
   iGPU also throttles the kernel ~2× versus a burst harness.
3. Phase 3: SYCL RAW_BF16 spine kernel so `--spine bf16` runs on XPU.
4. Evidence: a native test spec for XPU, and a `PLATFORMS.md` update.

## Environment (Windows, not admin)

- `DELTAFIN_TORCH_ROOT` = extracted `torch-2.13.0+xpu` wheel
  (`...\Temp\opencode\xpu\extracted\torch`).
- `DELTAFIN_SYCL_ROOT` = `E:\deltafin\.deltafin\sycl-dpcpp` (portable DPC++
  nightly-2026-10-07, `bin/clang++.exe -fsycl`).
- `DELTAFIN_XPU_RUNTIME` = dir with the Intel runtime DLLs (sycl9, mkl_sycl,
  ur_*, from `intel-sycl-rt` / `onemkl-sycl-*` / `intel-openmp` wheels).
- Build: set the three vars, then `cargo build --locked -p deltafin --release`.
  `doctor --runtime-only` shows `xpu_devices=1 xpu_moe_compiled=yes`.

Two invariants that must hold (both handled in the build graph / provider):
1. The nightly SYCL runtime is newer than the pip 2026.0 `sycl9.dll` and is
   backward compatible with torch_xpu — one runtime (the nightly's) must win the
   DLL search.
2. `at::xpu::getCurrentDeviceProperties` must be referenced (anchor) so the
   linker keeps the `torch_xpu.dll` import and the XPU ATen kernels register.

## Local iteration harness

`C:\Users\mpathy\AppData\Local\Temp\opencode\sycl\xpu_expert_bench.cpp` links
`tools/xpu_moe_kernels.cpp` and benchmarks the expert kernel and the int8 GEMV
against a scalar reference without loading the 1.3 TB model. Build it with the
same `clang++ -fsycl` flags the build graph uses (`lib.rs:3852`).
