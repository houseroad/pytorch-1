#include "torch/csrc/autograd/jit/pointwise_fusion.h"

namespace torch { namespace autograd {

std::shared_ptr<Expr> pointwise_fusion(std::shared_ptr<Expr> e) {
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

    return e;
}

}}
