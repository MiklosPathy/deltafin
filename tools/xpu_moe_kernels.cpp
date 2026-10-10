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

// e2m1 decode for every 4-bit code, so the inner loop is a table lookup rather
// than a switch. The sign bit is bit 3; the magnitude table is indexed by the
// low three bits and negated for negative codes.
constexpr float kE2M1Magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                     2.0f, 3.0f,  4.0f, 6.0f};

inline float e2m1_lookup(std::uint8_t code) {
  const float magnitude = kE2M1Magnitude[code & 7u];
  return (code & 8u) ? -magnitude : magnitude;
}

// One output row is computed by a work-group of this many work-items, each
// striding over the 32-element scale groups. Small groups keep the iGPU's few
// hardware threads busy without paying a wide cross-subgroup reduction.
constexpr int kRowGroup = 32;

}  // namespace

extern "C" {

// A deterministic probe whose result `doctor` can check.
K3_XPU_EXPORT std::int32_t k3_xpu_probe();

// USM shared allocation: host-writable and device-readable, so a disk read and
// the device kernel share one pointer with no copy.
K3_XPU_EXPORT void* k3_xpu_alloc_shared(std::size_t bytes);
// Aligned sibling used by the storage arena (the reader requires the buffer
// alignment its host allocator guaranteed).
K3_XPU_EXPORT void* k3_xpu_alloc_shared_aligned(std::size_t alignment,
                                                std::size_t bytes);
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

// Batched form: `n_mats` independent rows[m] x cols[m] matrices, each with its
// own packed/scales/x/output. One kernel launch covers every matrix, which
// matters because a per-expert launch is otherwise launch-overhead bound.
K3_XPU_EXPORT int k3_xpu_mxfp4_gemm_batch(
    const std::uint8_t* const* packed,
    const std::uint8_t* const* scales,
    const float* const* x,
    float* const* output,
    const int* rows,
    const int* cols,
    int n_mats);

// Row-quantized int8 GEMV with a per-output-channel fp32 scale:
//   output[b,n] = scale[n] * sum_k x[b,k] * weight[n,k].
// `queue_pointer` is a `sycl::queue*` supplied by the caller (torch_xpu's
// current stream) so the kernel is ordered with the ATen ops that produce the
// input; a null pointer selects the internal default queue. `x` and `weight`
// may be torch_xpu USM allocations: the iGPU shares host memory, so the kernel
// reads them without a copy.
K3_XPU_EXPORT int k3_xpu_int8_gemv(
    void* queue_pointer, const float* x, const std::int8_t* weight,
    const float* scale, float* output, int rows, int cols, int batches,
    int row_group);

// Original-BF16 spine GEMV: `output[b,n] = sum_k bf16(weight[n,k]) * x[b,k]`.
// The weight is the raw BF16 bit pattern (`uint16`), decoded as the top 16 bits
// of an fp32 (`bits << 16`), so the original weights are never expanded to
// FP32. Same queue/stream contract as the int8 GEMV.
K3_XPU_EXPORT int k3_xpu_bf16_gemv(
    void* queue_pointer, const std::uint16_t* weight, const float* x,
    float* output, int rows, int cols, int batches, int row_group);

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

K3_XPU_EXPORT void* k3_xpu_alloc_shared_aligned(std::size_t alignment,
                                                std::size_t bytes) {
  std::lock_guard<std::mutex> lock(queue_mutex());
  if (alignment == 0) {
    alignment = 64;
  }
  return sycl::aligned_alloc_shared(alignment, bytes, device_queue());
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
                const float weight = std::ldexp(e2m1_lookup(code), exponent);
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

K3_XPU_EXPORT int k3_xpu_mxfp4_gemm_batch(
    const std::uint8_t* const* packed,
    const std::uint8_t* const* scales,
    const float* const* x,
    float* const* output,
    const int* rows,
    const int* cols,
    int n_mats) {
  if (packed == nullptr || scales == nullptr || x == nullptr ||
      output == nullptr || rows == nullptr || cols == nullptr || n_mats <= 0) {
    return -1;
  }
  int total = 0;
  for (int m = 0; m < n_mats; ++m) {
    if (rows[m] <= 0 || cols[m] <= 0 || (cols[m] % 32) != 0) {
      return -1;
    }
    total += rows[m];
  }
  std::lock_guard<std::mutex> lock(queue_mutex());
  auto& queue = device_queue();
  // One work-group per output row. Each work-item strides over the row's 32-wide
  // scale groups and folds in one group at a time; the group scale is a single
  // `ldexp` per 32 weights rather than one per element, and the e2m1 decode is a
  // table lookup. The per-row partials are tree-reduced across the work-group.
  queue
      .parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<std::size_t>(total) *
                             static_cast<std::size_t>(kRowGroup)),
              sycl::range<1>(static_cast<std::size_t>(kRowGroup))),
          [=](sycl::nd_item<1> item) {
            const int row_index = static_cast<int>(item.get_group(0));
            const int tid = static_cast<int>(item.get_local_id(0));
            int matrix = 0;
            int base = 0;
            for (; matrix < n_mats; ++matrix) {
              if (row_index < base + rows[matrix]) {
                break;
              }
              base += rows[matrix];
            }
            const int row = row_index - base;
            const int cols_m = cols[matrix];
            const int groups = cols_m / 32;
            const std::uint8_t* row_packed =
                packed[matrix] + static_cast<std::int64_t>(row) * (cols_m / 2);
            const std::uint8_t* row_scales =
                scales[matrix] + static_cast<std::int64_t>(row) * groups;
            const float* batch_x = x[matrix];
            float local = 0.0f;
            for (int group = tid; group < groups; group += kRowGroup) {
              const std::uint8_t scale_byte = row_scales[group];
              if (scale_byte == 255u) {
                local = std::numeric_limits<float>::quiet_NaN();
                break;
              }
              const float scale =
                  sycl::ldexp(1.0f, static_cast<int>(scale_byte) - 127);
              const std::uint8_t* packed_group = row_packed + group * 16;
              const float* x_group = batch_x + group * 32;
              for (int lane = 0; lane < 32; ++lane) {
                const std::uint8_t byte = packed_group[lane >> 1];
                const std::uint8_t code = static_cast<std::uint8_t>(
                    (byte >> ((lane & 1) * 4)) & 0x0f);
                local = sycl::fma(e2m1_lookup(code) * scale, x_group[lane],
                                  local);
              }
            }
            const float row_sum = sycl::reduce_over_group(
                item.get_group(), local, sycl::plus<float>());
            if (tid == 0) {
              output[matrix][row] = row_sum;
            }
          })
      .wait();
  return 0;
}

K3_XPU_EXPORT int k3_xpu_int8_gemv(
    void* queue_pointer, const float* x, const std::int8_t* weight,
    const float* scale, float* output, int rows, int cols, int batches,
    int row_group) {
  if (x == nullptr || weight == nullptr || scale == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || batches <= 0) {
    return -1;
  }
  if (row_group <= 0) {
    row_group = 128;
  }
  const bool external = queue_pointer != nullptr;
  sycl::queue* queue =
      external ? static_cast<sycl::queue*>(queue_pointer) : &device_queue();
  std::lock_guard<std::mutex> lock(queue_mutex());
  auto event = queue->parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(rows) *
                         static_cast<std::size_t>(batches) *
                         static_cast<std::size_t>(row_group)),
          sycl::range<1>(static_cast<std::size_t>(row_group))),
      [=](sycl::nd_item<1> item) {
        const int block = static_cast<int>(item.get_group(0));
        const int tid = static_cast<int>(item.get_local_id(0));
        const int n = block % rows;
        const int batch = block / rows;
        const std::int8_t* weight_row =
            weight + static_cast<std::int64_t>(n) * cols;
        const float* x_row = x + static_cast<std::int64_t>(batch) * cols;
        float local = 0.0f;
        for (int k = tid; k < cols; k += row_group) {
          local =
              sycl::fma(x_row[k], static_cast<float>(weight_row[k]), local);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), local,
                                                  sycl::plus<float>());
        if (tid == 0) {
          output[static_cast<std::int64_t>(batch) * rows + n] = sum * scale[n];
        }
      });
  // On the caller's (torch_xpu) in-order stream the kernel is already ordered
  // with the ATen ops around it, so waiting here would only stall the host and
  // break the GPU's pipelining. The internal default queue is not ordered with
  // any caller, so it must complete before returning.
  if (!external) {
    event.wait();
  }
  return 0;
}

K3_XPU_EXPORT int k3_xpu_bf16_gemv(
    void* queue_pointer, const std::uint16_t* weight, const float* x,
    float* output, int rows, int cols, int batches, int row_group) {
  if (weight == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || batches <= 0) {
    return -1;
  }
  if (row_group <= 0) {
    row_group = 128;
  }
  const bool external = queue_pointer != nullptr;
  sycl::queue* queue =
      external ? static_cast<sycl::queue*>(queue_pointer) : &device_queue();
  std::lock_guard<std::mutex> lock(queue_mutex());
  auto event = queue->parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(rows) *
                         static_cast<std::size_t>(batches) *
                         static_cast<std::size_t>(row_group)),
          sycl::range<1>(static_cast<std::size_t>(row_group))),
      [=](sycl::nd_item<1> item) {
        const int block = static_cast<int>(item.get_group(0));
        const int tid = static_cast<int>(item.get_local_id(0));
        const int n = block % rows;
        const int batch = block / rows;
        const std::uint16_t* weight_row =
            weight + static_cast<std::int64_t>(n) * cols;
        const float* x_row = x + static_cast<std::int64_t>(batch) * cols;
        float local = 0.0f;
        for (int k = tid; k < cols; k += row_group) {
          const float decoded = sycl::bit_cast<float>(
              static_cast<std::uint32_t>(weight_row[k]) << 16);
          local = sycl::fma(decoded, x_row[k], local);
        }
        const float sum = sycl::reduce_over_group(item.get_group(), local,
                                                  sycl::plus<float>());
        if (tid == 0) {
          output[static_cast<std::int64_t>(batch) * rows + n] = sum;
        }
      });
  if (!external) {
    event.wait();
  }
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
