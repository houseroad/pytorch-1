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

// Needs to be done on functions, not expressions!
// static std::shared_ptr<Expr> derivative(std::shared_ptr<Expr> e) {
//}

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
  /*
  for (int i = 0; i < num_inputs; i++) {
    ss << "auto __t" << i << " = y;" << std::endl;
  }
  */
  ss << "float result0;" << std::endl;
  if (num_inputs > 0) ss << "float __t0 = y;" << std::endl;
  if (num_inputs > 1) ss << "float __t1 = z;" << std::endl;
  if (num_inputs > 2) throw std::logic_error("too many inputs");
  printCudaExpr(fn, ss);
  // NB: one output only atm!
  ss << "x = result0;" << std::endl;

  bool r;
  switch (num_inputs) {
    case 1:
      // TODO: This only works with Floats at the moment!
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
  std::vector<SavedVariable> saved_inputs;
  saved_inputs.reserve(inputs.size());
  for (auto& i : inputs) {
    saved_inputs.emplace_back(std::move(i->save(this)));
  }
  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<MapBackward>(std::move(f), std::move(saved_inputs));
  });


};

// Note [Fused backwards]
// ~~~~~~~~~~~~~~~~~~~~~~
// Here is the strategy we are taking for running fusion on backwards:
//  - UNCONDITIONALLY save all inputs
//  - Always use derivative formulas in terms of inputs, recomputing
//    intermediate results (no longer available due to fusion)
//
// However, there are some missed opportunities here:
//  - Sometimes an input becomes dead in the gradient pass, in which case we
//    shouldn't save it (a dead input can be deallocated sooner.)  It would
//    be a simple matter to check free variables of the gradient computation
//    never reference an input and avoid saving it.
//  - It may be profitable to write out an intermediate value to avoid
//    recomputing, but it is unclear when this is profitable.

auto MapBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("MapBackward", grad_outputs, 1);
  return {grad_outputs[0], grad_outputs[0]};
};

}}
