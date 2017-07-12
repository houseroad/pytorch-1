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

extern "C"
bool THCudaTensor_pointwiseApplyMany(THCState* state,
                                  THCudaTensor** ts,
                                  int n_ts,
                                  const char* op_string);


namespace torch { namespace autograd {

// TODO: this is not init'ed yet

/*
struct Linearize
  : public ExprVisitor<Differentiate, std::shared_ptr<Graph>>
{
}

struct Differentiate
  : public ExprVisitor<Differentiate, std::shared_ptr<Graph>>
  , public OperatorVisitor<Differentiate>
{
  // Expr
  std::shared_ptr<Graph> visitTuple(std::shared_ptr<Tuple> e, std::shared_ptr<Expr> r) {
    return std::make_shared<Graph>(e->locals, r);
  }
  std::shared_ptr<Graph> visitLet(std::shared_ptr<Let> e, std::shared_ptr<Expr> r) {
    e
    // TODO
  }
};
*/

static std::shared_ptr<Graph> differentiate(std::shared_ptr<Graph> e) {
  return e;
}

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
  if (num_inputs > 0) ss << "float __t0 = x1;" << std::endl;
  if (num_inputs > 1) ss << "float __t1 = x2;" << std::endl;
  if (num_inputs > 2) throw std::logic_error("too many inputs");
  printCudaExpr(fn, ss);
  // NB: one output only atm!
  ss << "x0 = result0;" << std::endl;

  bool r;
  std::vector<THCudaTensor*> ts;
  ts.push_back((THCudaTensor*)(output->cdata()));
  for (auto& input : inputs) {
    ts.push_back((THCudaTensor*)(input->data->cdata()));
  }

  r = THCudaTensor_pointwiseApplyMany(
          state,
          ts.data(),
          num_inputs + 1,
          ss.str().c_str());
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
