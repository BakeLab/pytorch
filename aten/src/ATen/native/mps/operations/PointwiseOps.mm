//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>

#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
namespace at::native {
// scope the MPS's internal methods to not expose them to at::native
namespace mps {

static void addc_mul_div_out_mps(const Tensor& self,
                                 const Tensor& tensor1,
                                 const Tensor& tensor2,
                                 const Scalar& value_opt, // default value = 1.0
                                 const Tensor& output,
                                 const bool is_div,
                                 const std::string& op_name) {
  if (value_opt.toDouble() == 0.0) {
    output.copy_(self);
    return;
  }

  if (output.numel() == 0) {
    return;
  }

  MPSStream* mpsStream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor *firstTensor = nil, *secondTensor = nil, *valueTensor = nil;
  };

  @autoreleasepool {
    bool contiguousOutput = !needsGather(output);
    Tensor output_ = output;
    if (!contiguousOutput) {
      output_ = at::empty_like(self, MemoryFormat::Contiguous);
    }

    std::string key = op_name + getTensorsStringKey({self, tensor1, tensor2});

// `value` is handed over at opmath precision, matching the CPU kernels: a
// half/bfloat16 alpha would both lose accuracy and be unable to represent the
// values `addcmul` accepts.
static void addcmul_mps_kernel(TensorIteratorBase& iter, const Scalar& value) {
  TORCH_CHECK_NOT_IMPLEMENTED(iter.common_dtype() != kBool, "addcmul_mps not implemented for 'Bool'");
  lib.exec_ternary_kernel(iter, "addcmul", value, at::toOpMathType(iter.common_dtype()));
}

static void addcdiv_mps_kernel(TensorIteratorBase& iter, const Scalar& value) {
  lib.exec_ternary_kernel(iter, "addcdiv", value, at::toOpMathType(iter.common_dtype()));
}

REGISTER_DISPATCH(addcmul_stub, &addcmul_mps_kernel)
REGISTER_DISPATCH(addcdiv_stub, &addcdiv_mps_kernel)

} // namespace at::native
