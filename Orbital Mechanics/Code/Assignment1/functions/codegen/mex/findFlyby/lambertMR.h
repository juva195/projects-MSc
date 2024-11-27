//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lambertMR.h
//
// Code generation for function 'lambertMR'
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
real_T lambertMR(const emlrtStack &sp, const real_T RI[3], const real_T RF[3],
                 real_T TOF, real_T &P, real_T &E, real_T &ERROR, real_T VI[3],
                 real_T VF[3], real_T &TPAR, real_T &THETA);

// End of code generation (lambertMR.h)
