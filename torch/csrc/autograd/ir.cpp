#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <sstream>

namespace torch { namespace autograd {

std::string getPythonName(const PyObject* obj, bool is_legacy) {
  AutoGIL gil;
  if (is_legacy) {
    return std::string(obj->ob_type->tp_name);
  } else {
    // NB: hypothetically __name__ could mutate the Python
    // object in a externally visible way. Please don't!
    auto wobj = const_cast<PyObject*>(obj);
    THPObjectPtr name{PyObject_GetAttrString(wobj, "__name__")};
    return THPUtils_unpackString(name.get());
  }
}

// Calling convention:
//    - Arguments are loaded into __t0, __t1, ... (to be changed soon)
//    - Results are loaded into result0, result1, etc.
class CudaPrinter : public ExprVisitor<CudaPrinter>, public OperatorVisitor<CudaPrinter> {
  std::ostream& s;

public:
  CudaPrinter(std::ostream& s) : s(s) {}

  void visitLocal(std::shared_ptr<Local> a) {
    s << "__t" << a->unique;
  }

  // Operator
  void visitPythonOp(std::shared_ptr<PythonOp> e) {
    throw std::logic_error("cannot print PythonOp to CUDA");
  }

  void visitMapOp(std::shared_ptr<MapOp> e) {
    throw std::logic_error("cannot print MapOp to CUDA");
  }

  void visitPrimOp(std::shared_ptr<PrimOp> e) {
    switch (e->op) {
      case PrimOp::Op::Add:
        s << "prim_add";
        break;
      case PrimOp::Op::Mul:
        s << "prim_mul";
        break;
      case PrimOp::Op::Sigmoid:
        s << "prim_sigmoid";
        break;
      case PrimOp::Op::Tanh:
        s << "prim_tanh";
        break;
      default:
        __builtin_unreachable();
    }
  }

  // Instruction
  void visitInstruction(std::shared_ptr<Instruction> i) {
    visitOperator(i->op);
    s << "(";
    bool first = true;
    for (auto& l : i->args) {
      if (!first) {
        s << ", ";
      } else {
        first = false;
      }
      visitLocal(l);
    }
    s << ")";
  }

  // Expr
  void visitLet(std::shared_ptr<Let> e) {
    // Instruction
    for (auto l : e->bind.lvals) {
      // This is a special-case, needs to be generalized
      s << "float ";
      visitLocal(l);
      s << ";" << std::endl;
    }
    visitOperator(e->bind.rval->op);
    s << "(";
    bool first = true;
    for (auto r : e->bind.rval->args) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(r);
    }
    for (auto l : e->bind.lvals) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << ");" << std::endl;
    visitExpr(e->expr);
  }
  void visitTuple(std::shared_ptr<Tuple> e) {
    int i = 0;
    for (auto l : e->locals) {
      s << "float output" << i << " = ";
      visitLocal(l);
      s << ";" << std::endl;
      i++;
    }
  }

  // Graph
  void visitGraph(std::shared_ptr<Graph> g) {
    int i = 0;
    for (auto l : g->params) {
      s << "float ";
      visitLocal(l);
      s << " = " << "input" << i << ";" << std::endl;
      i++;
    }
    visitExpr(g->body);
  }
};

// TODO: proper pretty-printer

class Printer : public ExprVisitor<Printer>, public OperatorVisitor<Printer> {
  std::ostream& s;

public:
  Printer(std::ostream& s) : s(s) {}

  void printPyObject(THPObjectPtr& obj) {
    THPObjectPtr repr { PyObject_Repr(obj.get()) };
    s << THPUtils_unpackString(repr.get());
  }

  void visitLocal(std::shared_ptr<Local> a) {
    s << "%" << a->unique;
  }

  // Operator
  void visitPythonOp(std::shared_ptr<PythonOp> e) {
    s << getPythonName(e->pyobj.get(), e->is_legacy);
    if (e->is_legacy) {
      s << " (legacy)";
    }
    for (auto& scalar : e->scalar_args) {
      s << " ";
      printPyObject(scalar);
    }
  }

  void visitMapOp(std::shared_ptr<MapOp> e) {
    s << "map [";
    // TODO: increase indentation
    //visitExpr(e->fn);
    CudaPrinter(s).visitGraph(e->fn);
    s << "]";
  }

  void visitPrimOp(std::shared_ptr<PrimOp> e) {
    s << "prim ";
    switch (e->op) {
      case PrimOp::Op::Add:
        s << "Add";
        break;
      case PrimOp::Op::Mul:
        s << "Mul";
        break;
      case PrimOp::Op::Sigmoid:
        s << "Sigmoid";
        break;
      case PrimOp::Op::Tanh:
        s << "Tanh";
        break;
      default:
        __builtin_unreachable();
    }
  }

  // Instruction
  void visitInstruction(std::shared_ptr<Instruction> i) {
    visitOperator(i->op);
    bool first = true;
    for (auto& l : i->args) {
      if (first) {
        s << " ";
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
  }

  // Expr
  void visitLet(std::shared_ptr<Let> e) {
    bool first = true;
    for (auto l : e->bind.lvals) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << " = ";
    visitInstruction(e->bind.rval);
    s << std::endl;
    visitExpr(e->expr);
  }
  void visitTuple(std::shared_ptr<Tuple> e) {
    s << "ret (";
    bool first = true;
    for (auto l : e->locals) {
      if (first) {
        first = false;
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << ")";
  }

  // Graph
  void visitGraph(std::shared_ptr<Graph> g) {
    s << "graph";
    bool first = true;
    for (auto& l : g->params) {
      if (first) {
        s << " ";
      } else {
        s << ", ";
      }
      visitLocal(l);
    }
    s << " {" << std::endl;
    visitExpr(g->body);
    s << std::endl;
    s << "}";
  }
};

void printExpr(std::shared_ptr<Expr> e) {
  Printer(std::cout).visitExpr(e);
}

void printExpr(std::shared_ptr<Expr> e, std::ostream& s) {
  Printer(s).visitExpr(e);
}

void printCudaGraph(std::shared_ptr<Graph> e, std::ostream& s) {
  CudaPrinter(s).visitGraph(e);
}

void printGraph(std::shared_ptr<Graph> e, std::ostream& s) {
  Printer(s).visitGraph(e);
}

}}
