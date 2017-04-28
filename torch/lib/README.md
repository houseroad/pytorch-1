This directory contains the low-level tensor libraries for PyTorch.
This code traces its lineage from the original Torch.  There are
multiple variants of the library, summarized here:

* TH = TorcH
* THC = TorcH Cuda
* THCS = TorcH Cuda Sparse
* THCUNN = TorcH CUda Neural Network (see cunn)
* THD = TorcH Distributed
* THNN = TorcH Neural Network
* THPP = TorcH C++ (see thpp)
* THS = TorcH Sparse

(You'll also see these abbreviations show up in symbol names.)

One important thing to know when working with this code is that
we use preprocessor tricks in order to define multiple versions of
tensor functions at different types.  Take addition for example:

```
TH_API void THTensor_(add)(THTensor *r_, THTensor *t, real value);
```

The `THTensor_` macro mangles "add" into a name for the particular type
we are specializing the function for.  For example, if we are defining
add for Float, `THTensor_(add)` will expand into `THFloatTensor_add`;
similarly, `THTensor` will expand into `THFloatTensor`.  `THGenerateAllTypes.h`
is responsible for repeatedly including the header file which use these
macros, redefining these macros appropriately so we generate multiple
versions.

Code that is "generic" over types is defined in `generic while.
