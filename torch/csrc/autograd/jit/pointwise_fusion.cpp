#include "torch/csrc/autograd/jit/pointwise_fusion.h"

namespace torch { namespace autograd {

std::shared_ptr<Expr> pointwise_fusion(std::shared_ptr<Expr> e) {
    return e;
}

}}
