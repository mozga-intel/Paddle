/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "mkldnn.hpp"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/variant.h"
namespace paddle {
namespace operators {
using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;
using dnnl::memory;
using dnnl::primitive;
using platform::to_void_cast;
using framework::DataLayout;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;
using platform::MKLDNNDeviceContext;
using framework::ExecutionContext;
using Tensor = framework::Tensor;
// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
static framework::DDim RowMatrixDimsFromVector(const framework::DDim& x_dim) {
  return x_dim.size() > 1 ? x_dim : framework::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static framework::DDim ColumnMatrixDimsFromVector(
    const framework::DDim& y_dim) {
  return y_dim.size() > 1 ? y_dim : framework::make_ddim({y_dim[0], 1});
}
/*
#define FWD(...) ::std::forward<decltype>(__VARGS__)>(__VA__ARGS__)
template <typename FThen>
auto then(FThen&& then) {
  return ::node {[
      parent = std::move(*this),
      f_then = FWD(f_then)
  ]() mutable->decltype(auto) {
      return f_then(static_cast<F&>(parent)());
    }
  }
}

template<typename >
class FFcFactory {
    struct Dim : D {
        int e;
        Dim(D & d, int w) : D(d), e(w) { }
    };
    struct Ve : V, ::std::vevtor<Dim> { }
    }
};
*/

template <typename XT, typename YT, typename OT>
class FcFactory {
 public:
  void CreateAndExecute(const ExecutionContext& ctx) {
    RecomputeOutputDims(ctx);
    SetDNNLEngine(ctx);
    CreateMemories(ctx);
    CreatePrimitive(ctx);
    Execute();
    SetOutputFormat(ctx);
    SetInitialized();
  }

 private:
  struct MatMulDims {
    const memory::dim BS, M, N, K;
  };

  void RecomputeOutputDims(const ExecutionContext& ctx) {
    auto input = ctx.Input<LoDTensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto output = ctx.Output<LoDTensor>("Out");
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights, false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w->dims(), output_dims, in_num_col_dims,
                 padding_weights);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());
  }

  void SetOutputFormat(const ExecutionContext& ctx) {
    using platform::MKLDNNFormatForSize;
    auto* out = ctx.Output<LoDTensor>("Out");
    auto format =
        MKLDNNFormatForSize(out->dims().size(), MKLDNNMemoryFormat::nchw);
    out->set_format(format);
    out->set_layout(DataLayout::kMKLDNN);
  }

  void SetDNNLEngine(const ExecutionContext& ctx) {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    engine_ = dev_ctx.GetEngine();
  }

  MatMulDims GetMatmulDims(const ExecutionContext& ctx) {
    auto mat_dim_x = math::CreateMatrixDescriptor(
        RowMatrixDimsFromVector(ctx.Input<LoDTensor>("Input")->dims()), 0,
        false);
    auto mat_dim_y = math::CreateMatrixDescriptor(
        ColumnMatrixDimsFromVector(ctx.Input<Tensor>("W")->dims()), 0, false);
    const auto x_bs = mat_dim_x.batch_size_;
    const auto y_bs = mat_dim_y.batch_size_;
    PADDLE_ENFORCE_EQ(x_bs > 0 && y_bs > 0 && x_bs != y_bs, false,
                      platform::errors::InvalidArgument(
                          "If batch sizes of X and Y are positive,"
                          "they have to be equal."));

    memory::dim BS = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    memory::dim M = mat_dim_x.height_;
    memory::dim N = mat_dim_y.width_;
    memory::dim K = mat_dim_x.width_;

    const auto x_dims =
        framework::vectorize<int>(ctx.Input<LoDTensor>("Input")->dims());
    if (x_dims.size() == 4) {
      BS = 1;
      M = x_dims[0];
      K = x_dims[1] * x_dims[2] * x_dims[3];
    }
    return {BS, M, N, K};
  }

  void CreateMemories(const ExecutionContext& ctx) {
    auto matmul_dims = GetMatmulDims(ctx);
    auto BS = matmul_dims.BS;
    auto M = matmul_dims.M;
    auto N = matmul_dims.N;
    auto K = matmul_dims.K;

    typedef memory::dims dims;
    dims x_dims = {BS, M, K};
    dims y_dims = {BS, K, N};
    dims b_dims = {1, 1, N};
    dims out_dims = {BS, M, N};

    auto x_md =
        memory::desc(x_dims, memory::data_type::f32, memory::format_tag::abc);
    auto w_md =
        memory::desc(y_dims, memory::data_type::f32, memory::format_tag::abc);
    auto o_md =
        memory::desc(out_dims, memory::data_type::f32, memory::format_tag::abc);

    x_mem_ = dnnl::memory(
        x_md, engine_, to_void_cast(ctx.Input<LoDTensor>("Input")->data<XT>()));
    y_mem_ = dnnl::memory(w_md, engine_,
                          to_void_cast(ctx.Input<Tensor>("W")->data<XT>()));
    auto* bias = ctx.Input<Tensor>("Bias");
    if (bias) {
      auto b_md =
          memory::desc(b_dims, memory::data_type::f32, memory::format_tag::abc);
      b_mem_ = dnnl::memory(
          b_md, engine_, to_void_cast(ctx.Input<Tensor>("Bias")->data<XT>()));
    }
    out_mem_ = dnnl::memory(
        o_md, engine_,
        to_void_cast(
            ctx.Output<LoDTensor>("Out")->mutable_data<OT>(ctx.GetPlace())));
  }

  void CreatePrimitive(const ExecutionContext& ctx) {
    dnnl::primitive_attr attr;
    dnnl::post_ops post_operations;

    if (ctx.Attr<std::string>("activation_type") == "relu") {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
      attr.set_post_ops(post_operations);
    }
    auto matmul_d = dnnl::matmul::desc(x_mem_.get_desc(), y_mem_.get_desc(),
                                       out_mem_.get_desc());
    if (b_mem_) {
      matmul_d = dnnl::matmul::desc(x_mem_.get_desc(), y_mem_.get_desc(),
                                    b_mem_.get_desc(), out_mem_.get_desc());
    }
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine_);
    matmul_prim_ = dnnl::matmul(matmul_pd);
  }

  void Execute() {
    dnnl::stream stream(engine_);
    if (b_mem_) {
      matmul_prim_.execute(stream, {
                                       {MKLDNN_ARG_SRC, x_mem_},
                                       {MKLDNN_ARG_WEIGHTS, y_mem_},
                                       {MKLDNN_ARG_BIAS, b_mem_},
                                       {MKLDNN_ARG_DST, out_mem_},
                                   });
    } else {
      matmul_prim_.execute(stream, {
                                       {MKLDNN_ARG_SRC, x_mem_},
                                       {MKLDNN_ARG_WEIGHTS, y_mem_},
                                       {MKLDNN_ARG_DST, out_mem_},
                                   });
    }
    stream.wait();
  }

  // If initialized, x memory should've been already initialized
  bool IsInitialized() { return initialized_; }

  void SetInitialized() { initialized_ = true; }

 private:
  dnnl::engine engine_;
  dnnl::memory x_mem_;
  dnnl::memory y_mem_;
  dnnl::memory b_mem_;
  dnnl::memory out_mem_;
  dnnl::matmul matmul_prim_;
  bool initialized_ = false;
};

template <typename XT, typename YT, typename OT>
static std::shared_ptr<FcFactory<XT, YT, OT>> GetFcPrimitiveFactory(
    const ExecutionContext& ctx) {
  const auto x_dims =
      framework::vectorize<int>(ctx.Input<LoDTensor>("Input")->dims());
  const auto y_dims = framework::vectorize<int>(ctx.Input<Tensor>("W")->dims());
  //  const auto b_dims =
  ///      framework::vectorize<int>(ctx.Input<Tensor>("Bias")->dims());
  const auto& out_name = ctx.OutputName("Out");
  const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

  const std::string key =
      platform::CreateKey(platform::ThreadIDasStr(), x_dims, y_dims, out_name);

  auto factory =
      std::static_pointer_cast<FcFactory<XT, YT, OT>>(dev_ctx.GetBlob(key));
  if (factory == nullptr) {
    factory = std::make_shared<FcFactory<XT, YT, OT>>();
    dev_ctx.SetBlob(key, factory);
  }

  return factory;
}

template <typename T>
constexpr bool IsInt8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}
// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename XT, typename YT>
static void ExecuteFc(const ExecutionContext& ctx) {
  constexpr bool is_int8 = IsInt8<XT>();
  const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
  constexpr bool fuse_relu = false;  // TODO(intel): Enable eltwise fuses
  if (!is_int8 || force_fp32_output) {
    GetFcPrimitiveFactory<XT, YT, float>(ctx)->CreateAndExecute(ctx);
  } else if (fuse_relu) {
    GetFcPrimitiveFactory<XT, YT, uint8_t>(ctx)->CreateAndExecute(ctx);
  } else {
    GetFcPrimitiveFactory<XT, YT, int8_t>(ctx)->CreateAndExecute(ctx);
  }
}

template <typename T>
class FCMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ExecuteFc<T, T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t>);
