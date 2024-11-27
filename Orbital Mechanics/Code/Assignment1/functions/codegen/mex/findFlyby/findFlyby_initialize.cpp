//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// findFlyby_initialize.cpp
//
// Code generation for function 'findFlyby_initialize'
//

// Include files
#include "findFlyby_initialize.h"
#include "_coder_findFlyby_mex.h"
#include "findFlyby_data.h"
#include "rt_nonfinite.h"

// Function Declarations
static void findFlyby_once();

// Function Definitions
static void findFlyby_once()
{
  mex_InitInfAndNan();
}

void findFlyby_initialize()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2022b(&st);
  emlrtClearAllocCountR2012b(&st, false, 0U, nullptr);
  emlrtEnterRtStackR2012b(&st);
  if (emlrtFirstTimeR2012b(emlrtRootTLSGlobal)) {
    findFlyby_once();
  }
}

// End of code generation (findFlyby_initialize.cpp)
