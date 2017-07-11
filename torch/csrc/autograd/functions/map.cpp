#include "map.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

#include "THPP/Type.hpp"
#include "THP.h"

#include <THPP/THPP.h>
#include <sstream>

// TODO ifdef me
extern THCState* state;

extern "C"
bool THCudaTensor_pointwiseApply2(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  const char* op_string);

extern "C"
bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const char* op_string);

namespace torch { namespace autograd {

// TODO: this is not init'ed yet

auto Map::apply(const variable_list& inputs) -> variable_list {

  auto num_inputs = inputs.size();
  if (num_inputs < 1) {
    throw std::logic_error("cannot map over no inputs");
  }
  auto& arg0 = inputs[0]->data;
  AutoGPU guard(arg0->getDevice());
  // NB: This assumes that all the dimensions are the same
  auto output = arg0->newTensor();
  output->resizeAs(*arg0);

  // TODO: The following code is CUDA ONLY

  // TODO: sanity check: make sure that free variables of expression
  // line up with number of inputs. (Better: have a function.)
  std::stringstream ss;
  ss << "x = ";
  printPExpr(fn, ss);

  bool r;
  switch (num_inputs) {
    case 1:
      r = THCudaTensor_pointwiseApply2(
              state,
              (THCudaTensor*)(output->cdata()),
              (THCudaTensor*)(inputs[0]->data->cdata()),
              ss.str().c_str());
      break;
    case 2:
      r = THCudaTensor_pointwiseApply3(
              state,
              (THCudaTensor*)(output->cdata()),
              (THCudaTensor*)(inputs[0]->data->cdata()),
              (THCudaTensor*)(inputs[1]->data->cdata()),
              ss.str().c_str());
      break;
    default:
      throw std::logic_error("mapping over more than 2 inputs not supported yet");
  }
  if (!r) {
    throw std::logic_error("unspecified failure running fused op");
  }
  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<MapBackward>(std::move(f));
  });


};

auto MapBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("AddBackward", grad_outputs, 1);
  return {grad_outputs[0], grad_outputs[0]};
};

}}
