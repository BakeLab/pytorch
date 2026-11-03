#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>


// These tests use PrivateUse1 to exercise extension backend dispatching
// without requiring an in-tree accelerator backend.

using namespace at;

static int test_int;

Tensor empty_override(SymIntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout,
                      std::optional<Device> device, std::optional<bool> pin_memory, std::optional<MemoryFormat> optional_memory_format) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          Storage::use_byte_size_t(),
          0,
          at::DataPtr(nullptr, Device(DeviceType::PrivateUse1, 1)),
          nullptr,
          false),
      DispatchKey::PrivateUse1,
      caffe2::TypeMeta::Make<float>());
  return Tensor(std::move(tensor_impl));
}

Tensor add_override(const Tensor & a, const Tensor & b , const Scalar& c) {
  auto out = empty({5, 5}, at::kPrivateUse1);  // Don't return self as-is
  test_int = 2;
  return out;
}

Tensor empty_strided_override(
  IntArrayRef size,
  IntArrayRef stride,
  std::optional<c10::ScalarType> dtype,
  std::optional<c10::Layout> layout,
  std::optional<c10::Device> device,
  std::optional<bool> pin_memory) {

  return empty_override(fromIntArrayRefSlow(size), dtype, layout, device, pin_memory, std::nullopt);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("aten::empty.memory_format",  empty_override);
  m.impl("aten::empty_strided",        empty_strided_override);
  m.impl("aten::add.Tensor",           add_override);
}

TEST(BackendExtensionTest, TestRegisterOp) {
  Tensor a = empty({5, 5}, at::kPrivateUse1);
  ASSERT_EQ(a.device().type(), at::kPrivateUse1);
  ASSERT_EQ(a.device().index(), 1);
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  ASSERT_EQ(test_int, 1);

  Tensor b = empty_like(a, at::kPrivateUse1);
  ASSERT_EQ(b.device().type(), at::kPrivateUse1);
  ASSERT_EQ(b.device().index(), 1);
  ASSERT_EQ(b.dtype(), caffe2::TypeMeta::Make<float>());

  add(a, b);
  ASSERT_EQ(test_int, 2);

  // Ensure that non-PrivateUse1 operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);
}
