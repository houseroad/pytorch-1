#pragma once

#include "torch/csrc/autograd/ir.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct Map : public Function {
  std::shared_ptr<PExpr> fn;
  Map(std::shared_ptr<PExpr> fn)
    : fn(fn)
    {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct MapBackward : public Function {
  std::vector<SavedVariable> saved_inputs;

  MapBackward(FunctionFlags&& flags,
              std::vector<SavedVariable>&& saved_inputs)
    : Function(std::move(flags))
    , saved_inputs(std::move(saved_inputs)) {}

  virtual variable_list apply(const variable_list& gradOutputs) override;
};

}}
