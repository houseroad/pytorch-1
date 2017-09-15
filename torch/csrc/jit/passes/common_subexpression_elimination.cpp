#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/interned_strings.h"

namespace torch { namespace jit {

void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph) {
  // Keep iterating until reach the fixed point.
  bool reach_fixed = false;
  while (!reach_fixed) {
    reach_fixed = true;
    auto nodes = graph->nodes();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      if (node->kind() == kSelect)
        continue;

      std::cout << node->unique() << "," << symbolToString(node->kind()) << std::endl;
    }
  }
}

}}
