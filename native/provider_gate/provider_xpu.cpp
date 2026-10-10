#include "provider_cuda_moe.h"
#include "provider_device.h"

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <vector>

#if defined(DELTAFIN_HAVE_XPU_PROVIDER_V1)
#include <ATen/Context.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#endif

// `getCurrentXPUStream` is needed by the int8 linear helper, which is available
// whenever the SYCL expert kernel DLL is linked.
#if defined(DELTAFIN_HAVE_XPU_PROVIDER_V1) || defined(DELTAFIN_HAVE_XPU_MOE_V1)
#include <c10/xpu/XPUStream.h>
#endif

namespace deltafin::provider_internal {

#if defined(DELTAFIN_HAVE_XPU_PROVIDER_V1)
// A second torch_xpu anchor, in the translation unit the int8/arena helpers
// already pull in. An isolated provider test that only calls `xpu_int8_linear`
// would otherwise dead-strip the `torch_xpu.dll` import (and with it the XPU
// ATen kernel registrations) that `provider_abi.cpp`'s anchor keeps in the
// main executable.
decltype(&at::xpu::getCurrentDeviceProperties) k3_xpu_anchor =
    &at::xpu::getCurrentDeviceProperties;

CudaProviderMemorySnapshot xpu_provider_memory_snapshot(
    const at::Device& device, const bool trim_unused) {
  if (!device.is_xpu() || !device.has_index()) {
    throw std::invalid_argument(
        "XPU memory snapshot requires an indexed XPU device");
  }
  // The allocator's per-device table is only sized by the lazy XPU init a
  // tensor operation triggers; a snapshot can be a run's first XPU touch.
  at::globalContext().lazyInitDevice(at::kXPU);
  const c10::DeviceGuard guard(device);
  if (trim_unused) {
    // Synchronize first so stream-deferred frees become allocator-owned
    // inactive blocks; emptyCache then releases only those unused blocks.
    c10::xpu::syncStreamsOnDevice(device.index());
    c10::xpu::XPUCachingAllocator::emptyCache();
  }
  const auto device_index = device.index();
  const auto stats =
      c10::xpu::XPUCachingAllocator::getDeviceStats(device_index);
  const auto memory =
      c10::xpu::XPUCachingAllocator::get()->getMemoryInfo(device_index);
  const auto nonnegative = [](const std::int64_t value) -> std::uint64_t {
    return value <= 0 ? 0 : static_cast<std::uint64_t>(value);
  };
  constexpr auto aggregate = static_cast<std::size_t>(
      c10::CachingAllocator::StatType::AGGREGATE);
  CudaProviderMemorySnapshot result;
  result.active_valid = true;
  result.reserved_valid = true;
  result.total_valid = true;
  result.available_valid = true;
  result.cache_trimmed = trim_unused;
  result.active_bytes =
      nonnegative(stats.allocated_bytes[aggregate].current);
  result.reserved_bytes =
      nonnegative(stats.reserved_bytes[aggregate].current);
  result.available_bytes = static_cast<std::uint64_t>(memory.first);
  result.total_bytes = static_cast<std::uint64_t>(memory.second);
  if (result.available_bytes > result.total_bytes) {
    throw std::runtime_error(
        "XPU allocator reported free memory larger than total memory");
  }
  return result;
}

#else

CudaProviderMemorySnapshot xpu_provider_memory_snapshot(
    const at::Device& device, const bool trim_unused) {
  if (!device.is_xpu()) {
    throw std::invalid_argument("XPU memory snapshot requires an XPU device");
  }
  if (trim_unused) {
    throw std::runtime_error(
        "XPU allocator trimming is unavailable in this provider build");
  }
  return {};
}

#endif

bool xpu_moe_compiled() noexcept {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  return true;
#else
  return false;
#endif
}

bool xpu_spine_bf16_compiled() noexcept {
  // The BF16 spine GEMV lives in the same SYCL kernel DLL as the MXFP4 expert
  // kernel, so it is present exactly when that DLL is linked.
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  return true;
#else
  return false;
#endif
}

bool xpu_int8_linear_compiled() noexcept {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  return true;
#else
  return false;
#endif
}

#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
// The SYCL int8 GEMV entry point from `deltafin_xpu_kernels.dll`. The first
// argument is a `sycl::queue*` (torch_xpu's current stream) so the kernel is
// ordered with the ATen ops around it.
extern "C" int k3_xpu_int8_gemv(void* queue_pointer, const float* x,
                                const std::int8_t* weight, const float* scale,
                                float* output, int rows, int cols, int batches,
                                int row_group);

// USM shared arena allocation from the same DLL, on the internal expert-kernel
// queue/context so the expert kernel can read the arena directly.
extern "C" void* k3_xpu_alloc_shared_aligned(std::size_t alignment,
                                             std::size_t bytes);
extern "C" void k3_xpu_free_shared(void* pointer);
#endif

#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
namespace {
std::atomic<bool> g_unified_expert_arena{false};
}  // namespace
#endif

bool xpu_unified_arena_available() noexcept {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  return true;
#else
  return false;
#endif
}

bool xpu_unified_expert_arena_active() noexcept {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  return g_unified_expert_arena.load(std::memory_order_relaxed);
#else
  return false;
#endif
}

void* xpu_alloc_unified_aligned(std::size_t alignment, std::size_t bytes) {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  void* pointer = k3_xpu_alloc_shared_aligned(alignment, bytes);
  if (pointer != nullptr) {
    g_unified_expert_arena.store(true, std::memory_order_relaxed);
  }
  return pointer;
#else
  static_cast<void>(alignment);
  static_cast<void>(bytes);
  return nullptr;
#endif
}

void xpu_free_unified(void* pointer) noexcept {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  if (pointer != nullptr) {
    k3_xpu_free_shared(pointer);
  }
#else
  static_cast<void>(pointer);
#endif
}

at::Tensor xpu_int8_linear(const at::Tensor& input, const at::Tensor& weight,
                           const at::Tensor& scale) {
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
  const std::int64_t cols = input.size(-1);
  const std::int64_t rows = weight.size(0);
  if (!input.is_xpu() || !weight.is_xpu() || !scale.is_xpu() ||
      input.scalar_type() != at::kFloat || weight.scalar_type() != at::kChar ||
      scale.scalar_type() != at::kFloat || weight.dim() != 2 ||
      weight.size(1) != cols || scale.dim() != 1 || scale.size(0) != rows ||
      !input.is_contiguous() || !weight.is_contiguous() ||
      !scale.is_contiguous()) {
    throw std::invalid_argument(
        "XPU int8 linear received incompatible tensors");
  }
  std::vector<std::int64_t> output_shape(input.sizes().begin(),
                                         input.sizes().end());
  output_shape.back() = rows;
  const at::Tensor flat = input.reshape({-1, cols});
  const std::int64_t batches = flat.size(0);
  at::Tensor output = at::empty({batches, rows}, input.options());
  auto& queue = c10::xpu::getCurrentXPUStream(input.device().index()).queue();
  if (k3_xpu_int8_gemv(static_cast<void*>(&queue),
                       flat.const_data_ptr<float>(),
                       weight.const_data_ptr<std::int8_t>(),
                       scale.const_data_ptr<float>(), output.data_ptr<float>(),
                       static_cast<int>(rows), static_cast<int>(cols),
                       static_cast<int>(batches), 128) != 0) {
    throw std::runtime_error("XPU int8 GEMV kernel failed");
  }
  return output.view(output_shape);
#else
  static_cast<void>(input);
  static_cast<void>(weight);
  static_cast<void>(scale);
  throw std::runtime_error(
      "XPU int8 linear is unavailable in this provider build");
#endif
}

#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
// The SYCL kernel DLL's probe entry point. Referencing it keeps the
// `deltafin_xpu_kernels.dll` import so its device image registers on load.
extern "C" std::int32_t k3_xpu_probe();
decltype(&k3_xpu_probe) k3_xpu_probe_anchor = &k3_xpu_probe;

std::int32_t xpu_probe_checksum() { return k3_xpu_probe(); }
#else
std::int32_t xpu_probe_checksum() { return -1; }
#endif

}  // namespace deltafin::provider_internal
