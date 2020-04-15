/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
// using dnnl::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
// // using dnnl::stream;
using platform::GetMKLDNNFormat;
using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
/*using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::stream;
using mkldnn::prop_kind;*/
using platform::MKLDNNGetDataType;

template <typename T_in, typename T_w, typename T_out>
class FCPrimitiveFactory {
 public:
  explicit FCPrimitiveFactory(const mkldnn::engine& engine) : engine_(engine) {}

  static framework::DDim RowMatrixDimsFromVector(const framework::DDim& x_dim) {
    return x_dim.size() > 1 ? x_dim : framework::make_ddim({1, x_dim[0]});
  }

  static framework::DDim ColumnMatrixDimsFromVector(
      const framework::DDim& y_dim) {
    return y_dim.size() > 1 ? y_dim : framework::make_ddim({1, y_dim[0]});
  }

  template <typename T>
  dnnl::memory CreateMemory(const memory::dims& dims,
                            const memory::dims& strides, const T* data) {
    auto md = memory::desc(dims, MKLDNNGetDataType<T>(), strides);
    return dnnl::memory(md, engine_, to_void_cast(data));
  }
  /*
  memory::dims InitMD() {
    memory::dims init_dims;
    switch (tag) {
      case tag::ab:
        init_dims = {};
        break;
      case tag::ba:
        init_dims = {};
        break;
      case tag::abc:
        init_dims = {};
        break;
      case tag::acb:
        init_dims = {};
        break;
      default:
        throw std::invalid_argument("Tag doesn't support custom operator");
    }
    return tag;
  } */
  void ExecuteFcPrimitive(const Tensor* input, const Tensor* weights,
                          const Tensor* bias, Tensor* output,
                          const ExecutionContext& ctx) {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();
    auto weight_dim = weights->dims();
    auto input_dims = input->dims();
    auto input_descriptor = math::CreateMatrixDescriptor(
        RowMatrixDimsFromVector(input_dims), 0, false);
    auto weight_descriptor = math::CreateMatrixDescriptor(
        ColumnMatrixDimsFromVector(weight_dim), 0, false);

    std::cout << "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD \n ";
    // memory dims DNNL
    const memory::dim MB =
        input_descriptor.batch_size_ || weight_descriptor.batch_size_
            ? std::max(input_descriptor.batch_size_,
                       weight_descriptor.batch_size_)
            : 1;
    const memory::dim M = input_descriptor.height_;
    const memory::dim K = input_descriptor.width_;
    const memory::dim N = weight_descriptor.width_;

    std::cout << MB << " " << M << " " << K << " " << N << std::endl;
    // memory descriptor
    memory::dims src_dim = {M, K};
    memory::dims weights_dim = {K, N};
    memory::dims bias_dim = {M, N};
    memory::dims out_dim = {M, N};

    // The Matmul Stride
    memory::dims input_strides = {M * K, K, 1};
    memory::dims weight_strides = {N * K, N, 1};
    memory::dims bias_stirdes = {M * N, N, 1};
    memory::dims output_strides = {M * N, N, 1};

    std::cout << "1111111111111111111111111111\n";
    auto x_memory_descriptor =
        memory::desc(src_dim, memory::data_type::f32, memory::format_tag::ab);
    auto w_memory_descriptor = memory::desc(weights_dim, memory::data_type::f32,
                                            memory::format_tag::ab);
    auto b_memory_descriptor =
        memory::desc(bias_dim, memory::data_type::f32, memory::format_tag::ab);
    auto d_memory_descriptor =
        memory::desc(out_dim, memory::data_type::f32, memory::format_tag::ab);

    auto x_dnnl_memory = dnnl::memory(x_memory_descriptor, engine,
                                      to_void_cast(input->data<T_in>()));
    auto w_dnnl_memory = dnnl::memory(w_memory_descriptor, engine,
                                      to_void_cast(weights->data<T_w>()));
    auto b_dnnl_memory = dnnl::memory(b_memory_descriptor, engine,
                                      to_void_cast(bias->data<float>()));
    auto d_dnnl_memory =
        dnnl::memory(d_memory_descriptor, engine,
                     to_void_cast(output->mutable_data<T_out>(ctx.GetPlace())));

    // Create memory descriptori
    auto matmul_d = dnnl::matmul::desc(x_memory_descriptor, w_memory_descriptor,
                                       d_memory_descriptor);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, engine);
    auto matmul_prim = dnnl::matmul(matmul_pd);
    dnnl::stream astream(engine);
    //   if (bias_) {
    //matmul_prim.execute(astream, {{MKLDNN_ARG_SRC, x_dnnl_memory},
    //                              {MKLDNN_ARG_WEIGHTS, w_dnnl_memory},
    //                              {MKLDNN_ARG_BIAS, b_dnnl_memory},
    //                              {MKLDNN_ARG_DST, d_dnnl_memory}});
    //  } else {

    //std::cout << "22222222222222222222222\n";
     matmul_prim.execute(astream, {{MKLDNN_ARG_SRC, x_dnnl_memory},
                                 {MKLDNN_ARG_WEIGHTS, w_dnnl_memory},
                                 {MKLDNN_ARG_DST, d_dnnl_memory}});
   //   }
    astream.wait();

    //std::cout << "33333333333333333333333\n";
    //output->set_layout(DataLayout::kMKLDNN);
    //output->set_format(GetMKLDNNFormat(d_dnnl_memory));
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
};

// Attempt to fetch cached primitive factory based on provided parameters
// of input format, weight dimensions and output name.
// If not cached, create a new one.
template <typename T_in, typename T_w, typename T_out>
static std::shared_ptr<FCPrimitiveFactory<T_in, T_w, T_out>>
GetPrimitiveFactory(const MKLDNNDeviceContext& dev_ctx,
                    const ExecutionContext& ctx, const Tensor* input,
                    const Tensor* weights,
                    const mkldnn::engine& mkldnn_engine) {
  const std::string key = platform::CreateKey(
      platform::ThreadIDasStr(), input->format(), input->dims()[0],
      framework::vectorize<int>(weights->dims()), ctx.OutputName("Out"));

  auto prim_creator =
      std::static_pointer_cast<FCPrimitiveFactory<T_in, T_w, T_out>>(
          dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator =
        std::make_shared<FCPrimitiveFactory<T_in, T_w, T_out>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename T_in, typename T_w>
static void ExecuteFc(const MKLDNNDeviceContext& dev_ctx,
                      const ExecutionContext& ctx, const Tensor* input,
                      const Tensor* w, const Tensor* bias, Tensor* output,
                      const mkldnn::engine& mkldnn_engine, bool fuse_relu,
                      bool force_fp32_output) {
  //  constexpr bool is_int8 =
  //      std::is_same<T_in, int8_t>::value || std::is_same<T_in,
  //      uint8_t>::value;
  //  if (!is_int8 || force_fp32_output) {
  GetPrimitiveFactory<T_in, T_w, float>(dev_ctx, ctx, input, w, mkldnn_engine)
      ->ExecuteFcPrimitive(input, w, bias, output, ctx);
  //  } else if (fuse_relu) {
  //    GetPrimitiveFactory<T_in, T_w, uint8_t>(dev_ctx, ctx, input, w,
  //                                            mkldnn_engine)
  //        ->ExecuteFcPrimitive(input, w, bias, output, ctx);
  //  } else {
  //   GetPrimitiveFactory<T_in, T_w, int8_t>(dev_ctx, ctx, input, w,
  //                                          mkldnn_engine)
  //       ->ExecuteFcPrimitive(input, w, bias, output, ctx);
  // }
}

template <typename T_in, typename T_w>
class FCMKLDNNOpKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("FC MKL-DNN must use CPUPlace."));
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<Tensor>("Out");

    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    ExecuteFc<T_in, T_w>(dev_ctx, ctx, input, w, bias, output, mkldnn_engine,
                         fuse_relu, force_fp32_output);

    output->set_layout(DataLayout::kMKLDNN);
  }
};
}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t, int8_t>);
