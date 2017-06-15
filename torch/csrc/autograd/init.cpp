#include <Python.h>

#include "THP.h"


PyObject * THPAutograd_initExtension(PyObject *_unused)
{
  PyObject *autograd_module = PyImport_ImportModule("torch.autograd");
  THPUtils_assert(autograd_module, "class loader couldn't access "
          "torch.autograd module");
  PyObject *autograd_dict = PyModule_GetDict(autograd_module);

  THPVariableClass      = PyMapping_GetItemString(autograd_dict,(char*)"Variable");
  THPFunctionClass      = PyMapping_GetItemString(autograd_dict,(char*)"Function");
  THPStochasticFunctionClass = PyMapping_GetItemString(autograd_dict,(char*)"StochasticFunction");
  THPNodeClass          = PyMapping_GetItemString(autograd_dict,(char*)"Node");
  THPUtils_assert(THPVariableClass, "couldn't find Variable class in "
          "torch.autograd module");
  THPUtils_assert(THPFunctionClass, "couldn't find Function class in "
          "torch.autograd module");
  THPUtils_assert(THPStochasticFunctionClass, "couldn't find "
          "StochasticFunction class in torch.autograd module");
  THPUtils_assert(THPNodeClass, "couldn't find "
          "Node class in torch.autograd module");

  Py_RETURN_TRUE;
}
