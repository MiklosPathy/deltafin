// SYCL device kernels for the Intel XPU provider.
//
// Compiled into a self-contained `deltafin_xpu_kernels.dll` by the DPC++
// toolchain (`clang++ -fsycl -shared`), which bundles and registers the device
// image on load. The provider links the import library and calls the exported
// `extern "C"` entry points; no other translation unit needs `-fsycl`.
//
// The streamed expert weights live in USM *shared* memory so the host fills
// them straight from disk and the kernel reads the same pointer: on an
// integrated GPU that removes the redundant host->device copy that ATen's
// device-only allocations force.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>

#if defined(_WIN32)
#define K3_XPU_EXPORT __declspec(dllexport)
#else
#define K3_XPU_EXPORT
#endif

namespace {

constexpr int kThreads = 128;
constexpr int kHidden = 3584;
constexpr int kIntermediate = 3072;
constexpr std::int64_t kExpertSpan = 17'547'264;

// One process-wide in-order queue. Kernel calls are synchronous (they wait), so
// the provider sees the same ordering contract it expects from a stream. A
// mutex serializes submission, since the provider may call from several threads.
sycl::queue& device_queue() {
  static sycl::queue queue{sycl::property::queue::in_order()};
  return queue;
}

std::mutex& queue_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline float e2m1_value(std::uint8_t code) {
  float magnitude;
  switch (code & 7u) {
    case 0: magnitude = 0.0f; break;
    case 1: magnitude = 0.5f; break;
    case 2: magnitude = 1.0f; break;
    case 3: magnitude = 1.5f; break;
    case 4: magnitude = 2.0f; break;
    case 5: magnitude = 3.0f; break;
    case 6: magnitude = 4.0f; break;
    default: magnitude = 6.0f; break;
  }
  return (code & 8u) ? -magnitude : magnitude;
}

}  // namespace

extern "C" {

// A deterministic probe whose result `doctor` can check.
K3_XPU_EXPORT std::int32_t k3_xpu_probe();

// USM shared allocation: host-writable and device-readable, so a disk read and
// the device kernel share one pointer with no copy.
K3_XPU_EXPORT void* k3_xpu_alloc_shared(std::size_t bytes);
K3_XPU_EXPORT void k3_xpu_free_shared(void* pointer);

// Expert MXFP4 GEMV. `packed` is rows x cols/2 nibbles, `scales` is rows x
// cols/32 e8m0 bytes, `x` is batches x cols fp32, `output` is batches x rows.
K3_XPU_EXPORT int k3_xpu_mxfp4_gemv(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    const float* x,
    float* output,
    int rows,
    int cols,
    int batches);

K3_XPU_EXPORT void k3_xpu_moe_shapes(int* hidden, int* intermediate, std::int64_t* expert_span);

}  // extern "C"

K3_XPU_EXPORT std::int32_t k3_xpu_probe() {
  std::lock_guard<std::mutex> lock(queue_mutex());
  auto& queue = device_queue();
  constexpr int kCount = 1024;
  std::int32_t* buffer = sycl::malloc_shared<std::int32_t>(kCount, queue);
  for (int i = 0; i < kCount; ++i) {
    buffer[i] = i;
  }
  queue.parallel_for(kCount, [=](sycl::id<1> i) { buffer[i] *= 2; }).wait();
  std::int32_t sum = 0;
  for (int i = 0; i < kCount; ++i) {
    sum += buffer[i];
  }
  sycl::free(buffer, queue);
  return sum;
}

K3_XPU_EXPORT void* k3_xpu_alloc_shared(std::size_t bytes) {
  std::lock_guard<std::mutex> lock(queue_mutex());
  return sycl::malloc_shared(bytes, device_queue());
}

K3_XPU_EXPORT void k3_xpu_free_shared(void* pointer) {
  if (pointer != nullptr) {
    std::lock_guard<std::mutex> lock(queue_mutex());
    sycl::free(pointer, device_queue());
  }
}

K3_XPU_EXPORT int k3_xpu_mxfp4_gemv(
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    const float* x,
    float* output,
    int rows,
    int cols,
    int batches) {
  if (packed == nullptr || scales == nullptr || x == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || batches <= 0 ||
      (cols % 32) != 0) {
    return -1;
  }
  const int groups = cols / 32;
  const std::int64_t total =
      static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(batches);
  std::lock_guard<std::mutex> lock(queue_mutex());
  auto& queue = device_queue();
  queue
      .parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<std::size_t>(total) * kThreads),
              sycl::range<1>(kThreads)),
          [=](sycl::nd_item<1> item) {
            const int block = static_cast<int>(item.get_group(0));
            const int tid = static_cast<int>(item.get_local_id(0));
            const int row = block % rows;
            const int batch = block / rows;
            const std::uint8_t* row_packed =
                packed + static_cast<std::int64_t>(row) * (cols / 2);
            const std::uint8_t* row_scales =
                scales + static_cast<std::int64_t>(row) * groups;
            const float* batch_x = x + static_cast<std::int64_t>(batch) * cols;
            float local = 0.0f;
            for (int group = tid; group < groups; group += kThreads) {
              const std::uint8_t scale_byte = row_scales[group];
              if (scale_byte == 255u) {
                local = std::numeric_limits<float>::quiet_NaN();
                break;
              }
              const int exponent = static_cast<int>(scale_byte) - 127;
              const int column_base = group * 32;
              const int packed_base = group * 16;
              for (int lane = 0; lane < 32; ++lane) {
                const std::uint8_t byte =
                    row_packed[packed_base + (lane >> 1)];
                const std::uint8_t code = static_cast<std::uint8_t>(
                    (byte >> ((lane & 1) * 4)) & 0x0f);
                const float weight = std::ldexp(e2m1_value(code), exponent);
                local = std::fma(weight, batch_x[column_base + lane], local);
              }
            }
            // Work-group reduction over the whole group.
            const float total = sycl::reduce_over_group(
                item.get_group(), local, sycl::plus<float>());
            if (tid == 0) {
              output[block] = total;
            }
          })
      .wait();
  return 0;
}

K3_XPU_EXPORT void k3_xpu_moe_shapes(
    int* hidden, int* intermediate, std::int64_t* expert_span) {
  if (hidden != nullptr) {
    *hidden = kHidden;
  }
  if (intermediate != nullptr) {
    *intermediate = kIntermediate;
  }
  if (expert_span != nullptr) {
    *expert_span = kExpertSpan;
  }
}
