//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// orbitalToCar.h
//
// Code generation for function 'orbitalToCar'
//

#pragma once

// Include files
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Function Declarations
void orbitalToCar(const emlrtStack &sp, real_T a, real_T e, real_T i,
                  real_T Omega, real_T omega, real_T theta, real_T r[3],
                  real_T v[3]);

// End of code generation (orbitalToCar.h)
