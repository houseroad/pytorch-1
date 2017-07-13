#include "torch/csrc/autograd/jit/pointwise_fusion.h"

#include <unordered_map>
#include <tuple>

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
  void visitTuple(std::shared_ptr<Tuple> e) {}
  void visitLet(std::shared_ptr<Let> e) {
    int i = 0;
    for (auto& l : e->bind.lvals) {
      env.insert({l->unique, std::make_tuple(e->bind.rval, i, 0)});
      i++;
    }
    for (auto& l : e->bind.rval->args) {
      std::get<2>(env[l->unique])++;
    }
    visitExpr(e->expr);
  }
};

/*
// Given:
//    g = \x_1 ... x_n ->
//          g_body
//          ret (r_1 ... r_m)
//    y_1 ... y_n   (in scope)
//    g2 = \s_1 ... s_m ->
//            g2_body
//
// Return:
//    s1 ... sm <- g y1 ... y_n
//    g2_body
//
// with g inlined (internal variables renamed
// to avoid conflict )
std::shared_ptr<Expr> inline_graph(
  std::shared_ptr<Graph> g,
  local_list inputs,
  std::shared_ptr<Graph> g2
  unique& unique_supply,
) {
  
}

// Given g1 = \x_1 ... x_i ... x_n -> ...
//       g2 = \y_1 ... y_m -> ...
//       arg = i
//       out = j
// computes the expression
//       \x_1 ..(omit i).. x_n y_1 ... y_m ->
//          r1 ... rj ... rs = g2 y1 ... y_m
//          s1 ... st = g1 x_1 ... rj ... x_n
//          ret r1 ..(omit j).. rs s1 ... st
// but with g1 and g2 inlined
void compose_at(std::shared_ptr<Graph> g1, std::shared_ptr<Graph> g2, int arg) {
}
*/

struct Fuser
  : public ExprVisitor<Fuser, void>
  , public OperatorVisitor<Fuser, std::shared_ptr<Graph>>
{
  def_uses env;
  Fuser(def_uses env)
    : env(env)
    {}
  void visitTuple(std::shared_ptr<Tuple> e) {}
  void visitLet(std::shared_ptr<Let> e) {
    auto g = visitOperator(e->bind.rval->op);
    if (g) {
      auto insn = e->bind.rval;
      // g will get updated with new graph
      for (auto& l : e->bind.rval->args) {
        // is single use
        auto r = env[l->unique];
        auto sub_insn   = std::get<0>(r);
        auto sub_output = std::get<1>(r);
        auto sub_uses   = std::get<2>(r);
        auto sub_g = visitOperator(sub_insn->op);
        if (sub_uses == 1 && sub_g != nullptr) {
          // then inline it!
          //
          // f x y = ...
          // g z = ...
          //
          // ...f (g z) y...

          // generate mappings of g and sub_g renumberings.
          // E.g., if we have
          //    %2 = map (fn %1 => ret %1) %3
          // Then we map %1 ==> %3
          std::unordered_map<unique, unique> g_inner_to_outer;
          std::unordered_map<unique, unique> sub_g_inner_to_outer;
          for (auto& l : e->bind.rval->args) {
          }
        }
      }
    }
  }

  std::shared_ptr<Graph> visitMapOp(std::shared_ptr<MapOp> o) {
    return o->fn;
  }
  std::shared_ptr<Graph> visitPrimOp(std::shared_ptr<PrimOp> o) {return nullptr;}
  std::shared_ptr<Graph> visitPythonOp(std::shared_ptr<PythonOp> o) {return nullptr;}
};

std::shared_ptr<Expr> pointwise_fusion(std::shared_ptr<Expr> e) {
  DefUses uses;
  uses.visitExpr(e);
  return e;
}

}}
