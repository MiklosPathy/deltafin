// SYCL device kernels for the Intel XPU provider.
//
// Compiled into a self-contained `deltafin_xpu_kernels.dll` by the DPC++
// toolchain (`clang++ -fsycl -shared`), which bundles and registers the device
// image on load. The provider links the import library and calls the exported
// `extern "C"` entry points; no other translation unit needs `-fsycl`.

#include <sycl/sycl.hpp>

#include <cstdint>

#if defined(_WIN32)
#define K3_XPU_EXPORT __declspec(dllexport)
#else
#define K3_XPU_EXPORT
#endif

extern "C" {

// A deterministic probe: a trivial device kernel whose result `doctor` can
// check, proving the SYCL device image registered and executed.
K3_XPU_EXPORT std::int32_t k3_xpu_probe();

}  // extern "C"

K3_XPU_EXPORT std::int32_t k3_xpu_probe() {
  sycl::queue queue;
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
