#include "provider_cuda_moe.h"
#include "provider_device.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#if defined(DELTAFIN_HAVE_XPU_PROVIDER_V1)
#include <ATen/Context.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#endif

namespace deltafin::provider_internal {

#if defined(DELTAFIN_HAVE_XPU_PROVIDER_V1)

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
#if defined(DELTAFIN_HAVE_XPU_SPINE_BF16_V1)
  return true;
#else
  return false;
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
