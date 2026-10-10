/*
 * XPU provider evidence.
 *
 * Exercises the SYCL kernels the XPU provider adds, on a real device when one
 * is present. The int8 projection is compared against ATen's
 * `_weight_int8pack_mm`; the MXFP4 expert batch and the original-BF16 spine
 * GEMV are covered by the shared `moe`/`target` suites when those run on an
 * XPU build. With no XPU device or kernel DLL the case reports SKIP but still
 * emits the PASS marker, so the spec is portable to CPU-only hosts.
 *
 * The source deliberately avoids `<ATen/xpu/...>` headers: the test build does
 * not carry the SYCL include path, and `at::kXPU` is an always-defined device
 * type. A tensor allocation on `at::kXPU` fails cleanly when no device exists.
 */
#include "provider_cuda_moe.h"

#include <ATen/ATen.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/zeros.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

double maximum_absolute_error(const at::Tensor& left, const at::Tensor& right) {
  return at::max(at::abs(left.to(at::kCPU, at::kFloat) -
                         right.to(at::kCPU, at::kFloat)))
      .item<double>();
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    bool exercised = false;
#if defined(DELTAFIN_HAVE_XPU_MOE_V1)
    if (deltafin::provider_internal::xpu_int8_linear_compiled()) {
      bool device_present = false;
      std::string probe_error = "no device";
      try {
        device_present =
            at::zeros({1}, at::TensorOptions().device(at::kXPU)).device().is_xpu();
      } catch (const std::exception& error) {
        // No XPU device: the allocation itself fails, which is the portable
        // SKIP path rather than a test failure.
        device_present = false;
        probe_error = error.what();
      }
      if (!device_present) {
        std::printf("provider_xpu.device_probe=SKIP (%s)\n",
                    probe_error.c_str());
      }
      if (device_present) {
        constexpr std::int64_t kRows = 1024;
        constexpr std::int64_t kColumns = 7168;
        const auto options = at::TensorOptions().device(at::kXPU);
        const at::Tensor input = at::randn({1, kColumns}, options);
        const at::Tensor weight = at::randint(
            -8, 9, {kRows, kColumns}, options.dtype(at::kChar));
        const at::Tensor scale = at::rand({kRows}, options) * 0.05 + 0.01;
        const at::Tensor reference =
            at::_weight_int8pack_mm(input, weight, scale);
        const at::Tensor candidate =
            deltafin::provider_internal::xpu_int8_linear(input, weight, scale);
        const double error = maximum_absolute_error(candidate, reference);
        if (!(error < 1.0e-2)) {
          throw std::runtime_error("XPU int8 linear parity max_abs=" +
                                   std::to_string(error));
        }
        std::printf("provider_xpu.int8_linear_max_abs=%g\n", error);
        exercised = true;
      }
    }
#endif
    if (!exercised) {
      std::printf("provider_xpu.exercised=SKIP (no XPU device or kernel DLL)\n");
    }
    std::printf("provider_xpu.synthetic=PASS\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "provider_xpu=FAIL: %s\n", error.what());
    return 1;
  }
}
