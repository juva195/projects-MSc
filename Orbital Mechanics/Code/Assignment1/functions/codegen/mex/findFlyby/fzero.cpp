//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// fzero.cpp
//
// Code generation for function 'fzero'
//

// Include files
#include "fzero.h"
#include "anonymous_function.h"
#include "findFlyby.h"
#include "findFlyby_data.h"
#include "findFlyby_internal_types.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo uc_emlrtRSI{
    131,     // lineNo
    "fzero", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\optimfun\\fzero.m" // pathName
};

static emlrtRSInfo vc_emlrtRSI{
    135,     // lineNo
    "fzero", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\optimfun\\fzero.m" // pathName
};

static emlrtRSInfo wc_emlrtRSI{
    217,     // lineNo
    "fzero", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\optimfun\\fzero.m" // pathName
};

static emlrtRTEInfo g_emlrtRTEI{
    137,     // lineNo
    9,       // colNo
    "fzero", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\optimfun\\fzero.m" // pName
};

static emlrtRTEInfo h_emlrtRTEI{
    149,     // lineNo
    9,       // colNo
    "fzero", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\optimfun\\fzero.m" // pName
};

// Function Definitions
namespace coder {
real_T fzero(const emlrtStack &sp, const anonymous_function FunFcn)
{
  emlrtStack b_st;
  emlrtStack st;
  real_T a;
  real_T b;
  real_T fa;
  real_T fb;
  st.prev = &sp;
  st.tls = sp.tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  a = 6671.01;
  b = 923796.45000000007;
  st.site = &uc_emlrtRSI;
  b_st.site = &sc_emlrtRSI;
  fa = findFlyby_anonFcn1(b_st, FunFcn.workspace.Delta,
                          FunFcn.workspace.VinfMinus, FunFcn.workspace.VinfPlus,
                          6671.01);
  st.site = &vc_emlrtRSI;
  b_st.site = &sc_emlrtRSI;
  fb = findFlyby_anonFcn1(b_st, FunFcn.workspace.Delta,
                          FunFcn.workspace.VinfMinus, FunFcn.workspace.VinfPlus,
                          923796.45000000007);
  if (muDoubleScalarIsInf(fa) || muDoubleScalarIsNaN(fa) ||
      (muDoubleScalarIsInf(fb) || muDoubleScalarIsNaN(fb))) {
    emlrtErrorWithMessageIdR2018a(
        &sp, &g_emlrtRTEI,
        "MATLAB:optimfun:fzero:ValuesAtEndPtsComplexOrNotFinite",
        "MATLAB:optimfun:fzero:ValuesAtEndPtsComplexOrNotFinite", 0);
  }
  if (fa == 0.0) {
    b = 6671.01;
  } else if (!(fb == 0.0)) {
    real_T c;
    real_T d;
    real_T e;
    real_T fc;
    boolean_T exitg1;
    if ((fa > 0.0) == (fb > 0.0)) {
      emlrtErrorWithMessageIdR2018a(
          &sp, &h_emlrtRTEI, "MATLAB:optimfun:fzero:ValuesAtEndPtsSameSign",
          "MATLAB:optimfun:fzero:ValuesAtEndPtsSameSign", 0);
    }
    fc = fb;
    c = 923796.45000000007;
    e = 0.0;
    d = 0.0;
    exitg1 = false;
    while ((!exitg1) && ((fb != 0.0) && (a != b))) {
      real_T m;
      real_T toler;
      if ((fb > 0.0) == (fc > 0.0)) {
        c = a;
        fc = fa;
        d = b - a;
        e = d;
      }
      if (muDoubleScalarAbs(fc) < muDoubleScalarAbs(fb)) {
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
      }
      m = 0.5 * (c - b);
      toler =
          4.4408920985006262E-16 * muDoubleScalarMax(muDoubleScalarAbs(b), 1.0);
      if ((muDoubleScalarAbs(m) <= toler) || (fb == 0.0)) {
        exitg1 = true;
      } else {
        if ((muDoubleScalarAbs(e) < toler) ||
            (muDoubleScalarAbs(fa) <= muDoubleScalarAbs(fb))) {
          d = m;
          e = m;
        } else {
          real_T q;
          real_T s;
          s = fb / fa;
          if (a == c) {
            fa = 2.0 * m * s;
            q = 1.0 - s;
          } else {
            real_T r;
            q = fa / fc;
            r = fb / fc;
            fa = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
            q = (q - 1.0) * (r - 1.0) * (s - 1.0);
          }
          if (fa > 0.0) {
            q = -q;
          } else {
            fa = -fa;
          }
          if ((2.0 * fa < 3.0 * m * q - muDoubleScalarAbs(toler * q)) &&
              (fa < muDoubleScalarAbs(0.5 * e * q))) {
            e = d;
            d = fa / q;
          } else {
            d = m;
            e = m;
          }
        }
        a = b;
        fa = fb;
        if (muDoubleScalarAbs(d) > toler) {
          b += d;
        } else if (b > c) {
          b -= toler;
        } else {
          b += toler;
        }
        st.site = &wc_emlrtRSI;
        b_st.site = &sc_emlrtRSI;
        fb = findFlyby_anonFcn1(b_st, FunFcn.workspace.Delta,
                                FunFcn.workspace.VinfMinus,
                                FunFcn.workspace.VinfPlus, b);
      }
    }
  }
  return b;
}

} // namespace coder

// End of code generation (fzero.cpp)
