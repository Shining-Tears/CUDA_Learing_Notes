#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>



#include <ATen/ops/special_xlogy_ops.h>

namespace at {


// aten::special_xlogy(Tensor self, Tensor other) -> Tensor
inline at::Tensor special_xlogy(const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::special_xlogy::call(self, other);
}

// aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor
inline at::Tensor special_xlogy(const at::Scalar & self, const at::Tensor & other) {
    return at::_ops::special_xlogy_self_scalar::call(self, other);
}

// aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor
inline at::Tensor special_xlogy(const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::special_xlogy_other_scalar::call(self, other);
}

// aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_xlogy_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::special_xlogy_out::call(self, other, out);
}
// aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_xlogy_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::special_xlogy_out::call(self, other, out);
}

// aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_xlogy_out(at::Tensor & out, const at::Scalar & self, const at::Tensor & other) {
    return at::_ops::special_xlogy_self_scalar_out::call(self, other, out);
}
// aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_xlogy_outf(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::special_xlogy_self_scalar_out::call(self, other, out);
}

// aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_xlogy_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::special_xlogy_other_scalar_out::call(self, other, out);
}
// aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_xlogy_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    return at::_ops::special_xlogy_other_scalar_out::call(self, other, out);
}

}
