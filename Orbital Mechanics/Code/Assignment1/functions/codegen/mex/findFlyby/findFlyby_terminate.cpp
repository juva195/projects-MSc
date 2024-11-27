//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// findFlyby_terminate.cpp
//
// Code generation for function 'findFlyby_terminate'
//

// Include files
#include "findFlyby_terminate.h"
#include "_coder_findFlyby_mex.h"
#include "findFlyby_data.h"
#include "rt_nonfinite.h"

// Function Definitions
void findFlyby_atexit()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void findFlyby_terminate()
{
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

// End of code generation (findFlyby_terminate.cpp)
