#include <cassert>
#include <cstdint>
#include <type_traits>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "gemmini.h"


static_assert(DIM == 8, "custom-mv targets the DIM=8 Gemmini bitstream");
static_assert(BANK_ROWS == 2048, "custom-mv targets BANK_ROWS=2048");
static_assert(ACC_ROWS == 2048, "custom-mv targets ACC_ROWS=2048");
static_assert(std::is_same_v<elem_t, float>, "custom-mv requires FP32 elem_t");
static_assert(std::is_same_v<acc_t, float>, "custom-mv requires FP32 acc_t");


extern "C" void pytorch_chipyard_mv_f32(
    int64_t matrix_rank,
    void* matrix_descriptor,
    int64_t vector_rank,
    void* vector_descriptor,
    int64_t output_rank,
    void* output_descriptor) {
  assert(matrix_rank == 2);
  assert(vector_rank == 1);
  assert(output_rank == 1);
  assert(matrix_descriptor != nullptr);
  assert(vector_descriptor != nullptr);
  assert(output_descriptor != nullptr);

  const auto& matrix =
      *static_cast<StridedMemRefType<float, 2>*>(matrix_descriptor);
  const auto& vector =
      *static_cast<StridedMemRefType<float, 1>*>(vector_descriptor);
  auto& output = *static_cast<StridedMemRefType<float, 1>*>(output_descriptor);

  assert(matrix.sizes[1] == vector.sizes[0]);
  assert(matrix.sizes[0] == output.sizes[0]);
  assert(matrix.strides[1] == 1);
  assert(vector.strides[0] == 1);
  assert(output.strides[0] == 1);

  const auto* matrix_data = matrix.data + matrix.offset;
  const auto* vector_data = vector.data + vector.offset;
  auto* output_data = output.data + output.offset;

  gemmini_flush(0);
  tiled_matmul_auto(
      static_cast<size_t>(matrix.sizes[0]),
      1,
      static_cast<size_t>(matrix.sizes[1]),
      matrix_data,
      vector_data,
      nullptr,
      output_data,
      static_cast<size_t>(matrix.strides[0]),
      static_cast<size_t>(vector.strides[0]),
      1,
      static_cast<size_t>(output.strides[0]),
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      NO_ACTIVATION,
      ACC_SCALE_IDENTITY,
      0,
      false,
      false,
      false,
      false,
      false,
      0,
      WS);
  gemmini_fence();
}
