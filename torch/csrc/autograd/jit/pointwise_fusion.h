#include "torch/csrc/autograd/ir.h"

namespace torch { namespace autograd {

// This is a very simple pointwise fusion pass, which I wrote to get
// a sense of the adequacy of the IR.  It has the following constraints:
//
//    - Pointwise (map) fusion only
//    - Maps must only be used once (syntactic fusion criteria)

std::shared_ptr<Expr> pointwise_fusion(std::shared_ptr<Expr>, int& unique_supply);

}}
