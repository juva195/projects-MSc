//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// findFlyby_internal_types.h
//
// Code generation for function 'findFlyby'
//

#pragma once

// Include files
#include "findFlyby_types.h"
#include "rtwtypes.h"
#include "emlrt.h"

// Type Definitions
struct struct_T {
  real_T Delta;
  real_T VinfMinus[3];
  real_T VinfPlus[3];
};

struct rtDesignRangeCheckInfo {
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
};

struct rtDoubleCheckInfo {
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
  int32_T checkKind;
};

struct rtEqualityCheckInfo {
  int32_T nDims;
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
};

struct rtRunTimeErrorInfo {
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
};

// End of code generation (findFlyby_internal_types.h)
