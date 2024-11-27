//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// findFlyby.h
//
// Code generation for function 'findFlyby'
//

#pragma once

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "emlrt.h"
#include "mex.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Function Declarations
emlrtCTX emlrtGetRootTLSGlobal();

void emlrtLockerFunction(EmlrtLockeeFunction aLockee, emlrtConstCTX aTLS,
                         void *aData);

void findFlyby(
    const emlrtStack *sp, const real_T firstLaunchDate[6],
    const real_T lastLaunchDate[6], real_T timeStep, real_T flybyWindow,
    real_T interceptWindow, coder::array<real_T, 2U> &DV,
    coder::array<real_T, 2U> &optDepDV, coder::array<real_T, 2U> &optIntDV,
    coder::array<real_T, 2U> &optGraDV, coder::array<real_T, 2U> &optFlyDV,
    coder::array<real_T, 2U> &mjd2000, coder::array<real_T, 2U> &departureV,
    coder::array<real_T, 2U> &flybyDepartureV,
    coder::array<real_T, 2U> &Vinfplus, coder::array<real_T, 2U> &Vinfminus,
    coder::array<real_T, 2U> &rpFlyby, coder::array<real_T, 2U> &delta,
    coder::array<real_T, 2U> &VPminus, coder::array<real_T, 2U> &VPplus);

real_T findFlyby_anonFcn1(const emlrtStack &sp, real_T Delta,
                          const real_T VinfMinus[3], const real_T VinfPlus[3],
                          real_T x);

// End of code generation (findFlyby.h)
