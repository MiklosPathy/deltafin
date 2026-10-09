# XPU provider — status / handoff

Branch: `feat/xpu-provider` (based on `feat/windows-cuda`, PR #35). Goal: run K3
on the Intel iGPU via PyTorch's `at::kXPU` backend so the working set stays in
shared system memory instead of streaming over PCIe from a 16 GB discrete GPU.

## Where it stands

Working and committed (HEAD = `e38c0d0`):

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

## Measured (dev box: Intel Graphics DEV_7D67, ~30 GB/s host memory)

- ATen XPU ops are slow: `_weight_int8pack_mm` 12.5 ms, fp32 GEMV 14.6 ms for
  1x7168x7168. Attention is ~18–20 s/token.
- The SYCL MXFP4 kernel is compute-bound (e2m1 `switch` + `ldexp` in the inner
  loop): ~2.1 GB/s, ~20 s/token for the experts.
- End to end ~35–54 s/token. Not yet faster than the CUDA build; both
  attention and the expert kernel need work.

## Instability (why we paused)

The uncommitted batched-kernel work caused `UR_RESULT_ERROR_DEVICE_LOST` and a
GPU TDR (screen blank + driver restart). Stashed as:

    git stash list   ->  stash@{0}: XPU batched kernel WIP (TDR-unstable)

The WIP batches the expert GEMVs into one launch (pointer arrays in USM shared,
one work-item per output) and adds an e2m1 lookup table to drop `ldexp` from the
inner loop — but it was not fully stabilised. Restore with `git stash pop` when
resuming.

## Environment (Windows, not admin)

- `DELTAFIN_TORCH_ROOT` = extracted `torch-2.13.0+xpu` wheel
  (`...\Temp\opencode\xpu\extracted\torch`).
- `DELTAFIN_SYCL_ROOT` = `E:\deltafin\.deltafin\sycl-dpcpp` (portable DPC++
  nightly-2026-10-07, `bin/clang++.exe -fsycl`).
- `DELTAFIN_XPU_RUNTIME` = dir with the Intel runtime DLLs (sycl9, mkl_sycl,
  ur_*, from `intel-sycl-rt` / `onemkl-sycl-*` / `intel-openmp` wheels).
- Build: set the three vars, then `cargo build --locked -p deltafin`.
  `doctor --runtime-only` shows `xpu_devices=1 xpu_moe_compiled=yes`.

Two invariants that must hold (both handled in the build graph / provider):
1. The nightly SYCL runtime is newer than the pip 2026.0 `sycl9.dll` and is
   backward compatible with torch_xpu — one runtime (the nightly's) must win the
   DLL search.
2. `at::xpu::getCurrentDeviceProperties` must be referenced (anchor) so the
   linker keeps the `torch_xpu.dll` import and the XPU ATen kernels register.

## Next steps

1. Stabilise the batched expert kernel (TDR), then optimise it (lookup table,
   no `ldexp`, vectorised loads, better occupancy) toward the ~30 GB/s memory
   floor.
2. The attention path is the bigger cost; ATen XPU GEMV/int8-mm is the ceiling.
   Consider SYCL attention kernels (or a faster ATen path) after the experts.
3. Phase 3: SYCL RAW_BF16 spine kernel so `--spine bf16` runs on XPU.
4. Evidence: a native test spec for XPU, and a `PLATFORMS.md` update.
