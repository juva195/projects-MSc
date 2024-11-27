//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_findFlyby_api.cpp
//
// Code generation for function '_coder_findFlyby_api'
//

// Include files
#include "_coder_findFlyby_api.h"
#include "findFlyby.h"
#include "findFlyby_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Declarations
static real_T b_emlrt_marshallIn(const emlrtStack &sp, const mxArray *b_nullptr,
                                 const char_T *identifier);

static real_T b_emlrt_marshallIn(const emlrtStack &sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId);

static real_T (*c_emlrt_marshallIn(const emlrtStack &sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[6];

static real_T d_emlrt_marshallIn(const emlrtStack &sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId);

static real_T (*emlrt_marshallIn(const emlrtStack &sp, const mxArray *b_nullptr,
                                 const char_T *identifier))[6];

static real_T (*emlrt_marshallIn(const emlrtStack &sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId))[6];

static const mxArray *emlrt_marshallOut(const coder::array<real_T, 2U> &u);

// Function Definitions
static real_T b_emlrt_marshallIn(const emlrtStack &sp, const mxArray *b_nullptr,
                                 const char_T *identifier)
{
  emlrtMsgIdentifier thisId;
  real_T y;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(sp, emlrtAlias(b_nullptr), &thisId);
  emlrtDestroyArray(&b_nullptr);
  return y;
}

static real_T b_emlrt_marshallIn(const emlrtStack &sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId)
{
  real_T y;
  y = d_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T (*c_emlrt_marshallIn(const emlrtStack &sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[6]
{
  static const int32_T dims[2]{1, 6};
  real_T(*ret)[6];
  int32_T iv[2];
  boolean_T bv[2]{false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)&sp, msgId, src, "double", false, 2U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (real_T(*)[6])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T d_emlrt_marshallIn(const emlrtStack &sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId)
{
  static const int32_T dims{0};
  real_T ret;
  emlrtCheckBuiltInR2012b((emlrtConstCTX)&sp, msgId, src, "double", false, 0U,
                          (const void *)&dims);
  ret = *static_cast<real_T *>(emlrtMxGetData(src));
  emlrtDestroyArray(&src);
  return ret;
}

static real_T (*emlrt_marshallIn(const emlrtStack &sp, const mxArray *b_nullptr,
                                 const char_T *identifier))[6]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[6];
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  y = emlrt_marshallIn(sp, emlrtAlias(b_nullptr), &thisId);
  emlrtDestroyArray(&b_nullptr);
  return y;
}

static real_T (*emlrt_marshallIn(const emlrtStack &sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId))[6]
{
  real_T(*y)[6];
  y = c_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static const mxArray *emlrt_marshallOut(const coder::array<real_T, 2U> &u)
{
  static const int32_T iv[2]{0, 0};
  const mxArray *m;
  const mxArray *y;
  y = nullptr;
  m = emlrtCreateNumericArray(2, (const void *)&iv[0], mxDOUBLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, &(((coder::array<real_T, 2U> *)&u)->data())[0]);
  emlrtSetDimensions((mxArray *)m, ((coder::array<real_T, 2U> *)&u)->size(), 2);
  emlrtAssign(&y, m);
  return y;
}

void findFlyby_api(const mxArray *const prhs[5], int32_T nlhs,
                   const mxArray *plhs[14])
{
  coder::array<real_T, 2U> DV;
  coder::array<real_T, 2U> VPminus;
  coder::array<real_T, 2U> VPplus;
  coder::array<real_T, 2U> Vinfminus;
  coder::array<real_T, 2U> Vinfplus;
  coder::array<real_T, 2U> delta;
  coder::array<real_T, 2U> departureV;
  coder::array<real_T, 2U> flybyDepartureV;
  coder::array<real_T, 2U> mjd2000;
  coder::array<real_T, 2U> optDepDV;
  coder::array<real_T, 2U> optFlyDV;
  coder::array<real_T, 2U> optGraDV;
  coder::array<real_T, 2U> optIntDV;
  coder::array<real_T, 2U> rpFlyby;
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *prhs_copy_idx_0;
  const mxArray *prhs_copy_idx_1;
  real_T(*firstLaunchDate)[6];
  real_T(*lastLaunchDate)[6];
  real_T flybyWindow;
  real_T interceptWindow;
  real_T timeStep;
  st.tls = emlrtRootTLSGlobal;
  emlrtHeapReferenceStackEnterFcnR2012b(&st);
  prhs_copy_idx_0 = emlrtProtectR2012b(prhs[0], 0, false, -1);
  prhs_copy_idx_1 = emlrtProtectR2012b(prhs[1], 1, false, -1);
  // Marshall function inputs
  firstLaunchDate =
      emlrt_marshallIn(st, emlrtAlias(prhs_copy_idx_0), "firstLaunchDate");
  lastLaunchDate =
      emlrt_marshallIn(st, emlrtAlias(prhs_copy_idx_1), "lastLaunchDate");
  timeStep = b_emlrt_marshallIn(st, emlrtAliasP(prhs[2]), "timeStep");
  flybyWindow = b_emlrt_marshallIn(st, emlrtAliasP(prhs[3]), "flybyWindow");
  interceptWindow =
      b_emlrt_marshallIn(st, emlrtAliasP(prhs[4]), "interceptWindow");
  // Invoke the target function
  findFlyby(&st, *firstLaunchDate, *lastLaunchDate, timeStep, flybyWindow,
            interceptWindow, DV, optDepDV, optIntDV, optGraDV, optFlyDV,
            mjd2000, departureV, flybyDepartureV, Vinfplus, Vinfminus, rpFlyby,
            delta, VPminus, VPplus);
  // Marshall function outputs
  DV.no_free();
  plhs[0] = emlrt_marshallOut(DV);
  if (nlhs > 1) {
    optDepDV.no_free();
    plhs[1] = emlrt_marshallOut(optDepDV);
  }
  if (nlhs > 2) {
    optIntDV.no_free();
    plhs[2] = emlrt_marshallOut(optIntDV);
  }
  if (nlhs > 3) {
    optGraDV.no_free();
    plhs[3] = emlrt_marshallOut(optGraDV);
  }
  if (nlhs > 4) {
    optFlyDV.no_free();
    plhs[4] = emlrt_marshallOut(optFlyDV);
  }
  if (nlhs > 5) {
    mjd2000.no_free();
    plhs[5] = emlrt_marshallOut(mjd2000);
  }
  if (nlhs > 6) {
    departureV.no_free();
    plhs[6] = emlrt_marshallOut(departureV);
  }
  if (nlhs > 7) {
    flybyDepartureV.no_free();
    plhs[7] = emlrt_marshallOut(flybyDepartureV);
  }
  if (nlhs > 8) {
    Vinfplus.no_free();
    plhs[8] = emlrt_marshallOut(Vinfplus);
  }
  if (nlhs > 9) {
    Vinfminus.no_free();
    plhs[9] = emlrt_marshallOut(Vinfminus);
  }
  if (nlhs > 10) {
    rpFlyby.no_free();
    plhs[10] = emlrt_marshallOut(rpFlyby);
  }
  if (nlhs > 11) {
    delta.no_free();
    plhs[11] = emlrt_marshallOut(delta);
  }
  if (nlhs > 12) {
    VPminus.no_free();
    plhs[12] = emlrt_marshallOut(VPminus);
  }
  if (nlhs > 13) {
    VPplus.no_free();
    plhs[13] = emlrt_marshallOut(VPplus);
  }
  emlrtHeapReferenceStackLeaveFcnR2012b(&st);
}

// End of code generation (_coder_findFlyby_api.cpp)
