#include "torch/csrc/autograd/jit/pointwise_fusion.h"

#include <unordered_map>
#include <tuple>
#include <iostream>

namespace torch { namespace autograd {

// Syntactic way to do it:
//    - Inline everything as much as you can
//    - Pattern match for map f (map g x) and fuse them
//
// Our strategy:
//    - An "inlining" pass which simply marks if a single-output
//    expression is used only once, and sets up a def-use link between
//    a local and its definition (stored externally for now...)
//
// Interesting thing about hog-wild ANFing: it becomes really obvious when
// you move let statements around. (In contrast, expression erases the
// ordering, so you don't have to worry about it.)
//
// If it's single use, we can unconditionally remove the let-binding.
// This highly smells of pointer manipulation (otherwise have to rebuild
// the entire front of the singly linked list).  We can instead remove
// let bindings in a dead-code elimination pass.
//
// Does it make sense to figure out nesting?  That may make certain
// optimization passes easier
//
// NO MUTATION!!!
//
//
// Dead variable elimination
//
//    let z, zz = map (\x -> (x + x, x * 2)) x
//    in z
//
//    NB: zz is dead
//
// Map is equivalent to a "Fusion group", but with its own binding structure.
// Benefit of "fusion group": you can run other optimizations as long as they
// "preserve" the fusion group.
//
//
//
// At the decision point, we have something like (NB shadowing!):
//
//      %1 = map [graph %0, %1, %2 { ... }] %2, %3, %4
//
// where %2 and %4 are eligible for inlining.  For simplicity, we want to inline
// one argument at a time.  Invariant is that inputs %2 refer to are not
// inlineable (because we already processed them to saturation.)  Inlining of
// %2 can introduce other arguments, which we want to deduplicate.

using output_num = int;
using unique = int;
using def_uses = std::unordered_map<unique, std::tuple<std::shared_ptr<Instruction>, output_num, int>>;

struct DefUses
  : public ExprVisitor<DefUses, void>
{
  // last int is uses
  def_uses env;
  void visitTuple(std::shared_ptr<Tuple> e) {
    for (auto l : e->locals) {
      std::get<2>(env[l->unique])++;
    }
  }
  void visitLet(std::shared_ptr<Let> e) {
    int i = 0;
    for (auto l : e->bind.lvals) {
      env.insert({l->unique, std::make_tuple(e->bind.rval, i, 0)});
      i++;
    }
    for (auto l : e->bind.rval->args) {
      std::get<2>(env[l->unique])++;
    }
    visitExpr(e->expr);
  }
};

struct RnEnv {
  std::unordered_map<unique, unique> env;
  unique& unique_supply;

  RnEnv(unique& unique_supply)
    : unique_supply(unique_supply)
    {};

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

struct FuseEdge2 : public ExprVisitor<FuseEdge2, std::shared_ptr<Expr>> {
  local_list ret_inputs;
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
    std::cout << "CA0\n";
    for (auto l : g2->params) {
      ret_inputs.push_back(rn_env.fresh(l));
    }
    std::cout << "CA1\n";
    auto ret_e = visitExpr(g2->body);
    std::cout << "CA2\n";
    // NB: ret_inputs got updated with other inputs needed
    return std::make_shared<Graph>(ret_inputs, ret_e);
  }
  std::shared_ptr<Expr> visitTuple(std::shared_ptr<Tuple> e) {
    printExpr(e, std::cerr);
    int i = 0;
    for (auto l : g1->params) {
      if (i != g1_input) {
        // no renaming!
        ret_inputs.push_back(l);
      }
      i++;
    }
    // TODO: stop dropping returns
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



struct Fuser
  : public ExprVisitor<Fuser, std::shared_ptr<Expr>>
  , public OperatorVisitor<Fuser, std::shared_ptr<Graph>>
{
  def_uses env;
  unique& unique_supply;
  Fuser(def_uses env, unique& unique_supply)
    : env(env)
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
        auto r = env[l->unique];
        auto sub_insn   = std::get<0>(r);
        auto sub_output = std::get<1>(r);
        auto sub_uses   = std::get<2>(r);
        auto sub_g = sub_insn ? visitOperator(sub_insn->op) : nullptr;
        // TODO: add check it's single output
        std::cout << l->unique << " sub_uses=" << sub_uses << "\n";
        if (sub_uses == 1 && sub_g != nullptr) {
          std::cout << "fusing!\n";
          g = fuse_edge(g, i, sub_g, sub_output, unique_supply);
          local_list new_new_args;
          std::cout << "done fusing\n";
          printGraph(g, std::cout);
          std::cout << "\n";
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
          i--; // redos everything!
          //i += sub_g->params.size() - 1;
        }
      }
      inst = std::make_shared<Instruction>(
        std::make_shared<MapOp>(g),
        new_args
        );
    } else {
      inst = e->bind.rval;
    }
    auto r = visitExpr(e->expr);
    return std::make_shared<Let>(Bind(e->bind.lvals, inst), r);
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
  for (auto it : uses.env) {
    std::cout << "%" << it.first << " usage count: " << std::get<2>(it.second) << std::endl;
  }
  return Fuser(uses.env, unique_supply).visitExpr(e);
}

}}
