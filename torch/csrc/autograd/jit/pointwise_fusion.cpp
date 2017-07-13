#include "torch/csrc/autograd/jit/pointwise_fusion.h"

#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <iostream>

namespace torch { namespace autograd {

using output_num = int;
using unique = int;
using def_uses = std::unordered_map<unique, int>;

// Computes the number of times a variable is used
struct DefUses
  : public ExprVisitor<DefUses, void>
{
  // last int is uses
  def_uses env;
  void visitTuple(std::shared_ptr<Tuple> e) {
    for (auto l : e->locals) {
      env[l->unique]++;
    }
  }
  void visitLet(std::shared_ptr<Let> e) {
    int i = 0;
    for (auto l : e->bind.lvals) {
      env.insert({l->unique, 0});
      i++;
    }
    for (auto l : e->bind.rval->args) {
      env[l->unique]++;
    }
    visitExpr(e->expr);
  }
};

// A renaming environment is used to remap local variables to fresh ones
// when we are joining to environments.
struct RnEnv {
  std::unordered_map<unique, unique> env;
  unique& unique_supply;

  RnEnv(unique& unique_supply)
    : unique_supply(unique_supply)
    {};

  // Remap a local number to a previously allocated fresh one
  std::shared_ptr<Local> rename(std::shared_ptr<Local> l) {
    auto r = env.find(l->unique);
    if (r != env.end()) {
      return std::make_shared<Local>(r->second);
    } else {
      std::cout << "looking for " << l->unique << "\nCurrent contents:\n";
      for (auto pair : env) {
        std::cout << pair.first << " -> " << pair.second << "\n";
      }
      throw std::logic_error("could not find unique");
    }
  }
  local_list rename(local_list locals) {
    local_list new_locals;
    new_locals.reserve(locals.size());
    for (auto l : locals) {
      new_locals.push_back(rename(l));
    }
    return new_locals;
  }

  // Make a fresh variable for this local
  std::shared_ptr<Local> fresh(std::shared_ptr<Local> l) {
    auto u = unique_supply++;
    env.insert({l->unique, u});
    return std::make_shared<Local>(u);
  }
  local_list fresh(local_list locals) {
    local_list new_locals;
    new_locals.reserve(locals.size());
    for (auto l : locals) {
      new_locals.push_back(fresh(l));
    }
    return new_locals;
  }
};

// Single edge fuser.  Only works for SINGLE RETURN things (NOT CHECKED)
//
// Given
//  g2 = graph y0 ... ym { ... ret z }
//  g2_output = j = 0 (always ZERO today)
//  g1 = graph x0 ... xi ... xn { ... }
//  g1_input = i
//
// fuse them into a single graph
//
//  graph x0 ... (y0 ... ym) ... xn
//    ...g2 body...
//    xi = z
//    ...g1 body...
//
// Up to alpha-equivalence.
//
// This can generalize to take multiple return g2, in which case the extra
// returns need to be returned
struct FuseEdge2 : public ExprVisitor<FuseEdge2, std::shared_ptr<Expr>> {
  local_list ret_inputs;
  // we rename all of the locals in g2, this keeps track of it
  RnEnv rn_env;
  unique& unique_supply;
  std::shared_ptr<Graph> g1;
  int g1_input;
  std::shared_ptr<Graph> g2;
  int g2_output;
  FuseEdge2(unique& unique_supply, std::shared_ptr<Graph> g1, int g1_input, std::shared_ptr<Graph> g2, int g2_output)
    : rn_env(unique_supply)
    , unique_supply(unique_supply)
    , g1(g1)
    , g1_input(g1_input)
    , g2(g2)
    , g2_output(g2_output)
    {}
  std::shared_ptr<Graph> run() {
    // Add the parameters to the environment, so subsequent renames catch them
    for (auto l : g2->params) {
      rn_env.fresh(l);
    }
    auto ret_e = visitExpr(g2->body);
    // NB: ret_inputs got updated via visitTuple
    return std::make_shared<Graph>(ret_inputs, ret_e);
  }
  std::shared_ptr<Expr> visitTuple(std::shared_ptr<Tuple> e) {
    printExpr(e, std::cerr);
    int i = 0;
    // setup the final inputs
    for (auto l : g1->params) {
      if (i != g1_input) {
        // no renaming!
        ret_inputs.push_back(l);
      } else {
        for (auto l : g2->params) {
          ret_inputs.push_back(rn_env.rename(l));
        }
      }
      i++;
    }
    // TODO: stop dropping extra returns from g2.  Need a second recursion
    // to handle that.
    return std::make_shared<Let>(
      Bind({g1->params[g1_input]},
           std::make_shared<Instruction>(
              std::make_shared<PrimOp>(PrimOp::Op::Id),
              local_list{rn_env.rename(e->locals[g2_output])}
           )
         ),
      g1->body);
  }
  std::shared_ptr<Expr> visitLet(std::shared_ptr<Let> e) {
    printExpr(e, std::cerr);
    auto ret_args = rn_env.rename(e->bind.rval->args);
    auto ret_lvals = rn_env.fresh(e->bind.lvals);
    auto ret_e = visitExpr(e->expr);
    return std::make_shared<Let>(Bind(ret_lvals, std::make_shared<Instruction>(e->bind.rval->op, ret_args)), ret_e);
  }
};

std::shared_ptr<Graph> fuse_edge(std::shared_ptr<Graph> g1, int g1_input, std::shared_ptr<Graph> g2, int g2_output, unique& unique_supply) {
  std::cout << "fuse_edge\ng1 = ";
  printGraph(g1, std::cout);
  std::cout << "\ng1_input = " << g1_input << "\ng2 = ";
  printGraph(g2, std::cout);
  std::cout << "\ng2_output = " << g2_output << "\n";
  return FuseEdge2(unique_supply, g1, g1_input, g2, g2_output).run();
}


// Fuse single-use map nodes.  Basically, anywhere a map node with one output is
// used exactly once by another map node, we fuse them together.
struct Fuser
  : public ExprVisitor<Fuser, std::shared_ptr<Expr>>
  , public OperatorVisitor<Fuser, std::shared_ptr<Graph>>
{
  def_uses uses;
  std::unordered_map<unique, std::pair<std::shared_ptr<Instruction>, output_num>> env;
  std::unordered_set<unique> killed;
  unique& unique_supply;
  Fuser(def_uses uses, unique& unique_supply)
    : uses(uses)
    , unique_supply(unique_supply)
    {}
  std::shared_ptr<Expr> visitTuple(std::shared_ptr<Tuple> e) {
    return e;
  }
  std::shared_ptr<Expr> visitLet(std::shared_ptr<Let> e) {
    auto g = visitOperator(e->bind.rval->op);
    std::shared_ptr<Instruction> inst;
    if (g) {
      // g is a map and so fusion could be
      // profitable.  There may be multiple
      // fusions available: we'll apply them
      // one by one (fuse_edge.)
      printGraph(g, std::cout);
      std::cout << "\n";
      local_list new_args = e->bind.rval->args;
      for (size_t i = 0; i < g->params.size(); i++) {
        if (g->params.size() != new_args.size()) throw std::logic_error("A");
        auto l = new_args[i];
        auto sub_uses = uses[l->unique];
        // TODO: the def uses here gets old!  That's awful!  Better to
        // recompute it as we go.
        auto r = env.find(l->unique);
        if (r != env.end()) {
          auto sub_insn   = r->second.first;
          auto sub_output = r->second.second;
          auto sub_g = visitOperator(sub_insn->op);
          // TODO: add check it's single output
          std::cout << l->unique << " sub_uses=" << sub_uses << "\n";
          if (sub_uses == 1 && sub_g != nullptr) {
            // TODO: arguably, we should kill the rest of the outputs
            // too when we know that they get folded into the map
            killed.insert(l->unique);
            // This variable was used only once and we've removed
            // it's only usage, so kill it!
            // killed.insert(l->unique);
            std::cout << "fusing!\n";
            g = fuse_edge(g, i, sub_g, sub_output, unique_supply);
            local_list new_new_args;
            std::cout << "done fusing\n";
            printGraph(g, std::cout);
            std::cout << "\n";
            // TODO: this is awful!  Would be better to splice,
            // or use a different calling convention for the merged
            // thing.
            for (size_t j = 0; j < new_args.size(); j++) {
              if (i == j) {
                for (auto l : sub_insn->args) {
                  new_new_args.push_back(l);
                }
              } else {
                new_new_args.push_back(new_args.at(j));
              }
            }
            new_args = new_new_args;
            i--; // reprocess the new arguments!!!
            // Would be better to skip over them...
            //i += sub_g->params.size() - 1;
          }
        }
      }
      inst = std::make_shared<Instruction>(
        std::make_shared<MapOp>(g),
        new_args
        );
    } else {
      inst = e->bind.rval;
    }
    int i = 0;
    for (auto l : e->bind.lvals) {
      env.insert({l->unique, {inst, i}});
      i++;
    }
    auto r = visitExpr(e->expr);
    bool all_killed = true;
    for (auto l : e->bind.lvals) {
      if (killed.count(l->unique) == 0) {
        all_killed = false;
        break;
      }
    }
    if (all_killed) {
      return r;
    } else {
      return std::make_shared<Let>(Bind(e->bind.lvals, inst), r);
    }
  }

  std::shared_ptr<Graph> visitMapOp(std::shared_ptr<MapOp> o) {
    return o->fn;
  }
  std::shared_ptr<Graph> visitPrimOp(std::shared_ptr<PrimOp> o) {return nullptr;}
  std::shared_ptr<Graph> visitPythonOp(std::shared_ptr<PythonOp> o) {return nullptr;}
};




std::shared_ptr<Expr> pointwise_fusion(std::shared_ptr<Expr> e, int& unique_supply) {
  DefUses uses;
  uses.visitExpr(e);
  /*
  for (auto it : uses.env) {
    std::cout << "%" << it.first << " usage count: " << std::get<2>(it.second) << std::endl;
  }
  */
  return Fuser(uses.env, unique_supply).visitExpr(e);
}

}}
