#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "gemmini.h"


static_assert(DIM == 8, "custom-linalg-matmul targets DIM=8 Gemmini");
static_assert(BANK_ROWS == 2048, "custom-linalg-matmul targets BANK_ROWS=2048");
static_assert(ACC_ROWS == 2048, "custom-linalg-matmul targets ACC_ROWS=2048");
static_assert(std::is_same_v<elem_t, float>, "FP32 elem_t is required");
static_assert(std::is_same_v<acc_t, float>, "FP32 acc_t is required");


extern "C" void triton_chipyard_gemv_f32(
    int64_t vector_rank,
    void* vector_descriptor,
    int64_t matrix_rank,
    void* matrix_descriptor,
    int64_t output_rank,
    void* output_descriptor) {
  assert(vector_rank == 1);
  assert(matrix_rank == 2);
  assert(output_rank == 1);
  assert(vector_descriptor != nullptr);
  assert(matrix_descriptor != nullptr);
  assert(output_descriptor != nullptr);

  const auto& vector =
      *static_cast<StridedMemRefType<float, 1>*>(vector_descriptor);
  const auto& matrix =
      *static_cast<StridedMemRefType<float, 2>*>(matrix_descriptor);
  auto& output =
      *static_cast<StridedMemRefType<float, 1>*>(output_descriptor);
  assert(vector.sizes[0] == matrix.sizes[0]);
  assert(output.sizes[0] == matrix.sizes[1]);
  assert(vector.strides[0] == 1);
  assert(output.strides[0] == 1);

  bool transpose_matrix = false;
  int64_t matrix_stride = matrix.strides[0];
  if (matrix.strides[1] != 1) {
    assert(matrix.strides[0] == 1);
    transpose_matrix = true;
    matrix_stride = matrix.strides[1];
  }

  const auto* vector_data = vector.data + vector.offset;
  const auto* matrix_data = matrix.data + matrix.offset;
  auto* output_data = output.data + output.offset;
  const size_t k = static_cast<size_t>(vector.sizes[0]);
  const size_t n = static_cast<size_t>(output.sizes[0]);

  gemmini_flush(0);
  tiled_matmul_auto(
      1,
      n,
      k,
      vector_data,
      matrix_data,
      nullptr,
      output_data,
      k,
      static_cast<size_t>(matrix_stride),
      n,
      n,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      NO_ACTIVATION,
      ACC_SCALE_IDENTITY,
      0,
      false,
      false,
      transpose_matrix,
      true,
      false,
      0,
      WS);
  gemmini_fence();
}


extern "C" void triton_chipyard_matmul_f32(
    int64_t lhs_rank,
    void* lhs_descriptor,
    int64_t rhs_rank,
    void* rhs_descriptor,
    int64_t output_rank,
    void* output_descriptor) {
  assert(lhs_rank == 2);
  assert(rhs_rank == 2);
  assert(output_rank == 2);
  assert(lhs_descriptor != nullptr);
  assert(rhs_descriptor != nullptr);
  assert(output_descriptor != nullptr);

  const auto& lhs = *static_cast<StridedMemRefType<float, 2>*>(lhs_descriptor);
  const auto& rhs = *static_cast<StridedMemRefType<float, 2>*>(rhs_descriptor);
  auto& output =
      *static_cast<StridedMemRefType<float, 2>*>(output_descriptor);

  assert(lhs.sizes[1] == rhs.sizes[0]);
  assert(lhs.sizes[0] == output.sizes[0]);
  assert(rhs.sizes[1] == output.sizes[1]);
  assert(lhs.strides[1] == 1);
  assert(output.strides[1] == 1);

  bool transpose_rhs = false;
  int64_t rhs_stride = rhs.strides[0];
  if (rhs.strides[1] != 1) {
    assert(rhs.strides[0] == 1);
    transpose_rhs = true;
    rhs_stride = rhs.strides[1];
  }

  const auto* lhs_data = lhs.data + lhs.offset;
  const auto* rhs_data = rhs.data + rhs.offset;
  auto* output_data = output.data + output.offset;

  // Triton's tl.dot accumulator is zero-initialized in this example, so the
  // linalg.matmul output operand does not need a separate D/bias input.
  gemmini_flush(0);
  tiled_matmul_auto(
      static_cast<size_t>(lhs.sizes[0]),
      static_cast<size_t>(rhs.sizes[1]),
      static_cast<size_t>(lhs.sizes[1]),
      lhs_data,
      rhs_data,
      nullptr,
      output_data,
      static_cast<size_t>(lhs.strides[0]),
      static_cast<size_t>(rhs_stride),
      static_cast<size_t>(output.strides[0]),
      static_cast<size_t>(output.strides[0]),
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      MVIN_SCALE_IDENTITY,
      NO_ACTIVATION,
      ACC_SCALE_IDENTITY,
      0,
      false,
      false,
      transpose_rhs,
      true,
      false,
      0,
      WS);
  gemmini_fence();
}
