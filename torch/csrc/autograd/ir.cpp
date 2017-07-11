#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <sstream>

namespace torch { namespace autograd {

std::string getOperatorName(const Operator& o) {
  switch (o._id) {
    case Operator::Id::PythonOp: return "PythonOp";
    case Operator::Id::MapOp: return "MapOp";
  }
  __builtin_unreachable();
}

std::string getExprName(const Expr& expr) {
  switch (expr._id) {
    case Expr::Id::Let: return "Let";
    case Expr::Id::Tuple: return "Tuple";
  }
  __builtin_unreachable();
}

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

// TODO: proper pretty-printer

class PExprPrinter : public PExprVisitor<PExprPrinter> {
  std::ostream& s;

public:
  PExprPrinter(std::ostream& s) : s(s) {}

  void visitPVar(std::shared_ptr<PVar> e) {
    switch (e->var) {
      /*
      case PVar::Var::X:
        s << "x";
        break;
        */
      case PVar::Var::Y:
        s << "y";
        break;
      case PVar::Var::Z:
        s << "z";
        break;
      default:
        __builtin_unreachable();
    }
  }
  void visitPBinOp(std::shared_ptr<PBinOp> e) {
    // TODO: elide this based on precedence
    s << "(";
    visitPExpr(e->lhs);
    switch (e->op) {
      case PBinOp::Op::Add:
        s << " + ";
        break;
      case PBinOp::Op::Mul:
        s << " * ";
        break;
      default:
        __builtin_unreachable();
    };
    visitPExpr(e->rhs);
    s << ")";
  }
  void visitPUnaryOp(std::shared_ptr<PUnaryOp> e) {
    switch (e->op) {
      case PUnaryOp::Op::Tanh:
        s << "tanh";
        break;
      case PUnaryOp::Op::Sigmoid:
        s << "sigmoid";
        break;
    }
    s << "( ";
    visitPExpr(e->expr);
    s << " )";
  }
};

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
    PExprPrinter(s).visitPExpr(e->fn);
    s << "]";
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
    s << "(";
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
};

void printExpr(std::shared_ptr<Expr> e) {
  Printer(std::cout).visitExpr(e);
}

void printExpr(std::shared_ptr<Expr> e, std::ostream& s) {
  Printer(s).visitExpr(e);
}

void printPExpr(std::shared_ptr<PExpr> e, std::ostream& s) {
  PExprPrinter(s).visitPExpr(e);
}

}}
