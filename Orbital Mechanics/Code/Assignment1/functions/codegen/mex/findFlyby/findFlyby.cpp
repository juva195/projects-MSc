//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// findFlyby.cpp
//
// Code generation for function 'findFlyby'
//

// Include files
#include "findFlyby.h"
#include "anonymous_function.h"
#include "date2jd.h"
#include "ephNEO.h"
#include "findFlyby_data.h"
#include "findFlyby_internal_types.h"
#include "fzero.h"
#include "lambertMR.h"
#include "norm.h"
#include "orbitalToCar.h"
#include "rt_nonfinite.h"
#include "uplanet.h"
#include "blas.h"
#include "coder_array.h"
#include "mwmathutil.h"
#include "omp.h"
#include <cstddef>
#include <emmintrin.h>

// Variable Definitions
static emlrtRSInfo emlrtRSI{
    11,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo b_emlrtRSI{
    13,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo c_emlrtRSI{
    15,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo d_emlrtRSI{
    19,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo e_emlrtRSI{
    49,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo f_emlrtRSI{
    51,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo g_emlrtRSI{
    57,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo h_emlrtRSI{
    59,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo i_emlrtRSI{
    63,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo j_emlrtRSI{
    80,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo k_emlrtRSI{
    82,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo l_emlrtRSI{
    86,          // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo m_emlrtRSI{
    103,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo n_emlrtRSI{
    107,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo o_emlrtRSI{
    111,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo p_emlrtRSI{
    113,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo q_emlrtRSI{
    114,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo r_emlrtRSI{
    116,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo s_emlrtRSI{
    117,         // lineNo
    "findFlyby", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRSInfo v_emlrtRSI{
    47,             // lineNo
    "date2mjd2000", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\time\\date2mjd2000.m" // pathName
};

static emlrtRSInfo
    cb_emlrtRSI{
        125,     // lineNo
        "colon", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pathName
    };

static emlrtRSInfo
    db_emlrtRSI{
        319,               // lineNo
        "eml_float_colon", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pathName
    };

static emlrtRSInfo tc_emlrtRSI{
    105, // lineNo
    "@(x)Delta*pi/180-asin(1/(1+(x*(norm(VinfMinus))^2)/muEarth))-asin(1/"
    "(1+(x*(norm(VinfPlus))^2)/muEarth))", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pathName
};

static emlrtRTEInfo b_emlrtRTEI{
    55,          // lineNo
    24,          // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo c_emlrtRTEI{
    78,          // lineNo
    32,          // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtBCInfo emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    137,         // lineNo
    25,          // colNo
    "mjd2000",   // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo b_emlrtBCI{
    -1,           // iFirst
    -1,           // iLast
    138,          // lineNo
    31,           // colNo
    "departureV", // aName
    "findFlyby",  // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo c_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    139,               // lineNo
    36,                // colNo
    "flybyDepartureV", // aName
    "findFlyby",       // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo d_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    129,         // lineNo
    29,          // colNo
    "Vinfplus",  // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo e_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    130,         // lineNo
    30,          // colNo
    "Vinfminus", // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtRTEInfo
    e_emlrtRTEI{
        419,               // lineNo
        15,                // colNo
        "assert_pmaxsize", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pName
    };

static emlrtBCInfo f_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    145,         // lineNo
    4,           // colNo
    "DV",        // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo g_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    146,         // lineNo
    4,           // colNo
    "DV",        // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo h_emlrtBCI{
    -1,                      // iFirst
    -1,                      // iLast
    45,                      // lineNo
    43,                      // colNo
    "possibleLaunchMjd2000", // aName
    "findFlyby",             // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo i_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    125,         // lineNo
    27,          // colNo
    "DV",        // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo j_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    126,         // lineNo
    25,          // colNo
    "VPminus",   // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo k_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    127,         // lineNo
    24,          // colNo
    "VPplus",    // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo l_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    128,         // lineNo
    23,          // colNo
    "delta",     // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo m_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    131,         // lineNo
    25,          // colNo
    "rpFlyby",   // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo n_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    132,         // lineNo
    20,          // colNo
    "DV",        // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo o_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    133,         // lineNo
    26,          // colNo
    "optDepDV",  // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo p_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    134,         // lineNo
    26,          // colNo
    "optIntDV",  // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo q_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    135,         // lineNo
    26,          // colNo
    "optGraDV",  // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtBCInfo r_emlrtBCI{
    -1,          // iFirst
    -1,          // iLast
    136,         // lineNo
    26,          // colNo
    "optFlyDV",  // aName
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m", // pName
    0                     // checkKind
};

static emlrtRTEInfo
    j_emlrtRTEI{
        320,     // lineNo
        20,      // colNo
        "colon", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pName
    };

static emlrtRTEInfo k_emlrtRTEI{
    19,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo l_emlrtRTEI{
    25,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo m_emlrtRTEI{
    26,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo n_emlrtRTEI{
    27,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo o_emlrtRTEI{
    28,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo p_emlrtRTEI{
    29,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo q_emlrtRTEI{
    30,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo r_emlrtRTEI{
    31,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo s_emlrtRTEI{
    32,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo t_emlrtRTEI{
    33,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo u_emlrtRTEI{
    34,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo v_emlrtRTEI{
    35,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo w_emlrtRTEI{
    36,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo x_emlrtRTEI{
    37,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

static emlrtRTEInfo y_emlrtRTEI{
    39,          // lineNo
    1,           // colNo
    "findFlyby", // fName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\findFlyby.m" // pName
};

// Function Definitions
emlrtCTX emlrtGetRootTLSGlobal()
{
  return emlrtRootTLSGlobal;
}

void emlrtLockerFunction(EmlrtLockeeFunction aLockee, emlrtConstCTX aTLS,
                         void *aData)
{
  omp_set_lock(&emlrtLockGlobal);
  emlrtCallLockeeFunction(aLockee, aTLS, aData);
  omp_unset_lock(&emlrtLockGlobal);
}

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
    coder::array<real_T, 2U> &VPminus, coder::array<real_T, 2U> &VPplus)
{
  __m128d r;
  __m128d r1;
  __m128d r2;
  ptrdiff_t incx_t;
  ptrdiff_t incy_t;
  ptrdiff_t n_t;
  coder::anonymous_function fun;
  coder::array<real_T, 2U> possibleLaunchMjd2000;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack st;
  real_T interceptKep[6];
  real_T a_data[3];
  real_T b_data[3];
  real_T flybyR[3];
  real_T flybyTransferArrivalV[3];
  real_T flybyTransferDepartureV[3];
  real_T flybyV[3];
  real_T interceptDepartureV[3];
  real_T interceptR[3];
  real_T launchR[3];
  real_T launchV[3];
  real_T Delta;
  real_T VpMinus;
  real_T VpPlus;
  real_T a;
  real_T a__13;
  real_T apnd;
  real_T auxDV;
  real_T cdiff;
  real_T d;
  real_T departureDV;
  real_T flybyDV;
  real_T flybyMjd2000;
  real_T gravityAsistDV;
  real_T interceptDV;
  real_T interceptMjd2000;
  real_T kd;
  real_T launchMjd2000;
  real_T ndbl;
  real_T rp;
  int32_T b_flybyMjd2000;
  int32_T b_interceptMjd2000;
  int32_T findFlyby_numThreads;
  int32_T i;
  int32_T i1;
  int32_T k;
  int32_T launchIndex;
  int32_T n;
  int32_T nm1d2;
  boolean_T emlrtHadParallelError{false};
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtConstCTX)sp);
  st.site = &emlrtRSI;
  //  astroConstants.m - Returns astrodynamic-related physical constants.
  //
  //  PROTOTYPE:
  //    out = astro_constants(in)
  //
  //  DESCRIPTION:
  //    Returns a row vector of constants, in which there is the corresponding
  //    constant for each element of the input vector.
  //
  //    List of identifiers:
  //        Generic astronomical constants:
  //            1   Universal gravity constant (G) (from DITAN and Horizon)
  //            [km^3/(kg*s^2)] 2   Astronomical Unit (AU) (from DE405) [km]
  //                Note:  The value for 1 au is from the IAU 2012 Resolution
  //                B1.
  //        Sun related:
  //            3   Sun mean radius (from DITAN) [km]
  //            4   Sun planetary constant (mu = mass * G) (from DE405)
  //                [km^3/s^2]
  //            31  Energy flux density of the Sun (from Wertz,SMAD)
  //                [W/m2 at 1 AU]
  //        Other:
  //            5   Speed of light in the vacuum (definition in the SI and
  //            Horizon) [km/s] 6   Standard free fall (the acceleration due to
  //            gravity on the
  //                Earth's surface at sea level) (from Wertz,SMAD) [m/s^2]
  //            7   Mean distance Earth-Moon (from Wertz,SMAD) [km]
  //            8   Obliquity (angle) of the ecliptic at Epoch 2000 (from
  //                Horizon) [rad]
  //            9   Gravitatonal field constant of the Earth (from Wertz,SMAD,
  //                taken from JGM-2). This should be used in conjunction to
  //                Earth radius = 6378.1363 km
  //            32  Days in a Julian year y = 365.25 d  (from Horizon)
  //        Planetary constants of the planets (mu = mass * G) [km^3/s^2]:
  //            11  Me      (from DE405)
  //            12  V       (from DE405)
  //            13  E       (from DE405)
  //            14  Ma      (from DE405)
  //            15  J       (from DE405)
  //            16  S       (from DE405)
  //            17  U       (from DE405)
  //            18  N       (from DE405)
  //            19  P       (from DE405)
  //            20  Moon    (from DE405)
  //        Mean radius of the planets [km]:
  //            21  Me      (from Horizon)
  //            22  V       (from Horizon)
  //            23  E       (from Horizon)
  //            24  Ma      (from Horizon)
  //            25  J       (from Horizon)
  //            26  S       (from Horizon)
  //            27  U       (from Horizon)
  //            28  N       (from Horizon)
  //            29  P       (from Horizon)
  //            30  Moon    (from Horizon)
  //
  //    Notes for upgrading this function:
  //        It is possible to add new constants.
  //        - DO NOT change the structure of the function, as well as its
  //            prototype.
  //        - DO NOT change the identifiers of the constants that have already
  //            been defined in this function. If you want to add a new
  //            constant, use an unused identifier.
  //        - DO NOT add constants that can be easily computed starting form
  //            other ones (avoid redundancy).
  //        Contact the author for modifications.
  //
  //  INPUT:
  //    in      Vector of identifiers of required constants.
  //
  //  OUTPUT:
  //    out     Vector of constants.
  //
  //  EXAMPLE:
  //    astroConstants([2, 4, 26])
  //       Returns a row vector in which there is the value of the AU, the Sun
  //       planetary constant and the mean radius of Saturn.
  //
  //    astroConstants(10 + [1:9])
  //       Returns a row vector with the planetary constant of each planet.
  //
  //  REFERENCES:
  //    - DITAN (Direct Interplanetary Trajectory Analysis), Massimiliano
  //        Vasile, 2006.
  // 	- Wertz J. R., Larson W. J., "Space Mission Analysis and Design", Third
  //        Edition, Space Technology Library 2003.
  //    [A]   DE405 - http://iau-comm4.jpl.nasa.gov/de405iom/de405iom.pdf
  //    [B]   Explanatory Supplement to the Astronomical Almanac. 1992. K. P.
  //          Seidelmann, Ed., p.706 (Table 15.8) and p.316 (Table 5.8.1),
  //          University Science Books, Mill Valley, California.
  //    [C]   Tholen, D.J. and Buie, M.W. 1990. "Further Analysis of
  //          Pluto-Charon Mutual Event Observations" BAAS 22(3):1129.
  //    [D]   Seidelmann, P.K. et al. 2007. "Report of the IAU/IAG Working
  //          Group on cartographic coordinates and rotational elements: 2006"
  //          Celestial Mech. Dyn. Astr. 98:155-180.
  //    [F]   Anderson, J.D., et al. 1987. "The mass, gravity field, and
  //          ephemeris of Mercury" Icarus 71:337-349.
  //    [G]   Konopliv, A.S., et al. 1999. "Venus gravity: 180th degree and
  //          order model" Icarus 139:3-18.
  //    [H]   Folkner, W.M. and Williams, J.G. 2008. "Mass parameters and
  //          uncertainties in planetary ephemeris DE421." Interoffice Memo.
  //          343R-08-004 (internal document), Jet Propulsion Laboratory,
  //          Pasadena, CA.
  //    [I]   Jacobson, R.A. 2008. "Ephemerides of the Martian Satellites -
  //          MAR080" Interoffice Memo. 343R-08-006 (internal document),
  //          Jet Propulsion Laboratory, Pasadena, CA.
  //    [J]   Jacobson, R.A. 2005. "Jovian Satellite ephemeris - JUP230"
  //          private communication.
  //    [K]   Jacobson, R.A., et al. 2006. "The gravity field of the Saturnian
  //          system from satellite observations and spacecraft tracking data"
  //          AJ 132(6):2520-2526.
  //    [L]   Jacobson, R.A. 2007. "The gravity field of the Uranian system and
  //          the orbits of the Uranian satellites and rings" BAAS 39(3):453.
  //    [M]   Jacobson, R.A. 2008. "The orbits of the Neptunian satellites and
  //          the orientation of the pole of Neptune" BAAS 40(2):296.
  //    [N]   Jacobson, R.A. 2007. "The orbits of the satellites of Pluto -
  //          Ephemeris PLU017" private communication.
  //    [W1]  http://ssd.jpl.nasa.gov/?planet_phys_par Last retrieved
  //          20/03/2013
  //    [W2]  http://ssd.jpl.nasa.gov/?sat_phys_par Last retrieved
  //          20/03/2013
  //    [W3]  http://ssd.jpl.nasa.gov/horizons.cgi Last retrieved
  //          20/03/2013
  //    [M1]  Bills, B.G. and Ferrari, A.J. 1977. ``A Harmonic Analysis of
  //          Lunar Topography'', Icarus 31, 244-259.
  //    [M2]  Standish, E. M. 1998. JPL Planetary and Lunar Ephemerides,
  //          DE405/LE405.
  //    [M3]  Lunar Constants and Models Document, Ralph B. Roncoli, 23 Sept
  //    2005,
  //          JPL Technical Document D-32296
  //
  //  CALLED FUNCTIONS:
  //    (none)
  //
  //  AUTHOR:
  //    Matteo Ceriotti, 2006, MATLAB, astroConstants.m
  //
  //  PREVIOUS VERSION:
  //    Matteo Ceriotti, 2006, MATLAB, astro_constants.m, Ver. 1.2
  //        - Header and function name in accordance with guidlines.
  //
  //  CHANGELOG:
  //    26/10/2006, Camilla Colombo: Updated.
  //    22/10/2007, Camilla Colombo: astroConstants(8) added (Obliquity (angle)
  //        of the ecliptic at Epoch 2000).
  //    02/10/2009, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //    12/11/2010, Camilla Colombo: astroConstants(9) added (J2) Note: the
  //        present value of J2 is not consistent with the value of the Earth
  //        radius. This value of J2 should be used in conjunction to Earth
  //        radius = 6378.1363 km
  //    19/03/2013, Camilla Colombo: constants updated to NASA JPL website.
  //        References added.
  //    20/03/2013, REVISION, Francesca Letizia.
  //    22/03/2013, Francesca Letizia: all GM from DE405.
  //
  //  -------------------------------------------------------------------------
  //  9: J2
  //  32: 365.25
  //  out(i)=0.63781600000000E+04; % From DITAN
  //  out(i)=6371.00; % From Horizon [B]
  //  From Horizon [W3]
  st.site = &b_emlrtRSI;
  //  astroConstants.m - Returns astrodynamic-related physical constants.
  //
  //  PROTOTYPE:
  //    out = astro_constants(in)
  //
  //  DESCRIPTION:
  //    Returns a row vector of constants, in which there is the corresponding
  //    constant for each element of the input vector.
  //
  //    List of identifiers:
  //        Generic astronomical constants:
  //            1   Universal gravity constant (G) (from DITAN and Horizon)
  //            [km^3/(kg*s^2)] 2   Astronomical Unit (AU) (from DE405) [km]
  //                Note:  The value for 1 au is from the IAU 2012 Resolution
  //                B1.
  //        Sun related:
  //            3   Sun mean radius (from DITAN) [km]
  //            4   Sun planetary constant (mu = mass * G) (from DE405)
  //                [km^3/s^2]
  //            31  Energy flux density of the Sun (from Wertz,SMAD)
  //                [W/m2 at 1 AU]
  //        Other:
  //            5   Speed of light in the vacuum (definition in the SI and
  //            Horizon) [km/s] 6   Standard free fall (the acceleration due to
  //            gravity on the
  //                Earth's surface at sea level) (from Wertz,SMAD) [m/s^2]
  //            7   Mean distance Earth-Moon (from Wertz,SMAD) [km]
  //            8   Obliquity (angle) of the ecliptic at Epoch 2000 (from
  //                Horizon) [rad]
  //            9   Gravitatonal field constant of the Earth (from Wertz,SMAD,
  //                taken from JGM-2). This should be used in conjunction to
  //                Earth radius = 6378.1363 km
  //            32  Days in a Julian year y = 365.25 d  (from Horizon)
  //        Planetary constants of the planets (mu = mass * G) [km^3/s^2]:
  //            11  Me      (from DE405)
  //            12  V       (from DE405)
  //            13  E       (from DE405)
  //            14  Ma      (from DE405)
  //            15  J       (from DE405)
  //            16  S       (from DE405)
  //            17  U       (from DE405)
  //            18  N       (from DE405)
  //            19  P       (from DE405)
  //            20  Moon    (from DE405)
  //        Mean radius of the planets [km]:
  //            21  Me      (from Horizon)
  //            22  V       (from Horizon)
  //            23  E       (from Horizon)
  //            24  Ma      (from Horizon)
  //            25  J       (from Horizon)
  //            26  S       (from Horizon)
  //            27  U       (from Horizon)
  //            28  N       (from Horizon)
  //            29  P       (from Horizon)
  //            30  Moon    (from Horizon)
  //
  //    Notes for upgrading this function:
  //        It is possible to add new constants.
  //        - DO NOT change the structure of the function, as well as its
  //            prototype.
  //        - DO NOT change the identifiers of the constants that have already
  //            been defined in this function. If you want to add a new
  //            constant, use an unused identifier.
  //        - DO NOT add constants that can be easily computed starting form
  //            other ones (avoid redundancy).
  //        Contact the author for modifications.
  //
  //  INPUT:
  //    in      Vector of identifiers of required constants.
  //
  //  OUTPUT:
  //    out     Vector of constants.
  //
  //  EXAMPLE:
  //    astroConstants([2, 4, 26])
  //       Returns a row vector in which there is the value of the AU, the Sun
  //       planetary constant and the mean radius of Saturn.
  //
  //    astroConstants(10 + [1:9])
  //       Returns a row vector with the planetary constant of each planet.
  //
  //  REFERENCES:
  //    - DITAN (Direct Interplanetary Trajectory Analysis), Massimiliano
  //        Vasile, 2006.
  // 	- Wertz J. R., Larson W. J., "Space Mission Analysis and Design", Third
  //        Edition, Space Technology Library 2003.
  //    [A]   DE405 - http://iau-comm4.jpl.nasa.gov/de405iom/de405iom.pdf
  //    [B]   Explanatory Supplement to the Astronomical Almanac. 1992. K. P.
  //          Seidelmann, Ed., p.706 (Table 15.8) and p.316 (Table 5.8.1),
  //          University Science Books, Mill Valley, California.
  //    [C]   Tholen, D.J. and Buie, M.W. 1990. "Further Analysis of
  //          Pluto-Charon Mutual Event Observations" BAAS 22(3):1129.
  //    [D]   Seidelmann, P.K. et al. 2007. "Report of the IAU/IAG Working
  //          Group on cartographic coordinates and rotational elements: 2006"
  //          Celestial Mech. Dyn. Astr. 98:155-180.
  //    [F]   Anderson, J.D., et al. 1987. "The mass, gravity field, and
  //          ephemeris of Mercury" Icarus 71:337-349.
  //    [G]   Konopliv, A.S., et al. 1999. "Venus gravity: 180th degree and
  //          order model" Icarus 139:3-18.
  //    [H]   Folkner, W.M. and Williams, J.G. 2008. "Mass parameters and
  //          uncertainties in planetary ephemeris DE421." Interoffice Memo.
  //          343R-08-004 (internal document), Jet Propulsion Laboratory,
  //          Pasadena, CA.
  //    [I]   Jacobson, R.A. 2008. "Ephemerides of the Martian Satellites -
  //          MAR080" Interoffice Memo. 343R-08-006 (internal document),
  //          Jet Propulsion Laboratory, Pasadena, CA.
  //    [J]   Jacobson, R.A. 2005. "Jovian Satellite ephemeris - JUP230"
  //          private communication.
  //    [K]   Jacobson, R.A., et al. 2006. "The gravity field of the Saturnian
  //          system from satellite observations and spacecraft tracking data"
  //          AJ 132(6):2520-2526.
  //    [L]   Jacobson, R.A. 2007. "The gravity field of the Uranian system and
  //          the orbits of the Uranian satellites and rings" BAAS 39(3):453.
  //    [M]   Jacobson, R.A. 2008. "The orbits of the Neptunian satellites and
  //          the orientation of the pole of Neptune" BAAS 40(2):296.
  //    [N]   Jacobson, R.A. 2007. "The orbits of the satellites of Pluto -
  //          Ephemeris PLU017" private communication.
  //    [W1]  http://ssd.jpl.nasa.gov/?planet_phys_par Last retrieved
  //          20/03/2013
  //    [W2]  http://ssd.jpl.nasa.gov/?sat_phys_par Last retrieved
  //          20/03/2013
  //    [W3]  http://ssd.jpl.nasa.gov/horizons.cgi Last retrieved
  //          20/03/2013
  //    [M1]  Bills, B.G. and Ferrari, A.J. 1977. ``A Harmonic Analysis of
  //          Lunar Topography'', Icarus 31, 244-259.
  //    [M2]  Standish, E. M. 1998. JPL Planetary and Lunar Ephemerides,
  //          DE405/LE405.
  //    [M3]  Lunar Constants and Models Document, Ralph B. Roncoli, 23 Sept
  //    2005,
  //          JPL Technical Document D-32296
  //
  //  CALLED FUNCTIONS:
  //    (none)
  //
  //  AUTHOR:
  //    Matteo Ceriotti, 2006, MATLAB, astroConstants.m
  //
  //  PREVIOUS VERSION:
  //    Matteo Ceriotti, 2006, MATLAB, astro_constants.m, Ver. 1.2
  //        - Header and function name in accordance with guidlines.
  //
  //  CHANGELOG:
  //    26/10/2006, Camilla Colombo: Updated.
  //    22/10/2007, Camilla Colombo: astroConstants(8) added (Obliquity (angle)
  //        of the ecliptic at Epoch 2000).
  //    02/10/2009, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //    12/11/2010, Camilla Colombo: astroConstants(9) added (J2) Note: the
  //        present value of J2 is not consistent with the value of the Earth
  //        radius. This value of J2 should be used in conjunction to Earth
  //        radius = 6378.1363 km
  //    19/03/2013, Camilla Colombo: constants updated to NASA JPL website.
  //        References added.
  //    20/03/2013, REVISION, Francesca Letizia.
  //    22/03/2013, Francesca Letizia: all GM from DE405.
  //
  //  -------------------------------------------------------------------------
  //  9: J2
  //  32: 365.25
  //  out(i)=0.59736990612667E+25*6.67259e-20; % From DITAN
  //  out(i)=5.97219E+24*6.67259e-20;     % From Horizon [H]
  //  From DE405
  st.site = &c_emlrtRSI;
  //  astroConstants.m - Returns astrodynamic-related physical constants.
  //
  //  PROTOTYPE:
  //    out = astro_constants(in)
  //
  //  DESCRIPTION:
  //    Returns a row vector of constants, in which there is the corresponding
  //    constant for each element of the input vector.
  //
  //    List of identifiers:
  //        Generic astronomical constants:
  //            1   Universal gravity constant (G) (from DITAN and Horizon)
  //            [km^3/(kg*s^2)] 2   Astronomical Unit (AU) (from DE405) [km]
  //                Note:  The value for 1 au is from the IAU 2012 Resolution
  //                B1.
  //        Sun related:
  //            3   Sun mean radius (from DITAN) [km]
  //            4   Sun planetary constant (mu = mass * G) (from DE405)
  //                [km^3/s^2]
  //            31  Energy flux density of the Sun (from Wertz,SMAD)
  //                [W/m2 at 1 AU]
  //        Other:
  //            5   Speed of light in the vacuum (definition in the SI and
  //            Horizon) [km/s] 6   Standard free fall (the acceleration due to
  //            gravity on the
  //                Earth's surface at sea level) (from Wertz,SMAD) [m/s^2]
  //            7   Mean distance Earth-Moon (from Wertz,SMAD) [km]
  //            8   Obliquity (angle) of the ecliptic at Epoch 2000 (from
  //                Horizon) [rad]
  //            9   Gravitatonal field constant of the Earth (from Wertz,SMAD,
  //                taken from JGM-2). This should be used in conjunction to
  //                Earth radius = 6378.1363 km
  //            32  Days in a Julian year y = 365.25 d  (from Horizon)
  //        Planetary constants of the planets (mu = mass * G) [km^3/s^2]:
  //            11  Me      (from DE405)
  //            12  V       (from DE405)
  //            13  E       (from DE405)
  //            14  Ma      (from DE405)
  //            15  J       (from DE405)
  //            16  S       (from DE405)
  //            17  U       (from DE405)
  //            18  N       (from DE405)
  //            19  P       (from DE405)
  //            20  Moon    (from DE405)
  //        Mean radius of the planets [km]:
  //            21  Me      (from Horizon)
  //            22  V       (from Horizon)
  //            23  E       (from Horizon)
  //            24  Ma      (from Horizon)
  //            25  J       (from Horizon)
  //            26  S       (from Horizon)
  //            27  U       (from Horizon)
  //            28  N       (from Horizon)
  //            29  P       (from Horizon)
  //            30  Moon    (from Horizon)
  //
  //    Notes for upgrading this function:
  //        It is possible to add new constants.
  //        - DO NOT change the structure of the function, as well as its
  //            prototype.
  //        - DO NOT change the identifiers of the constants that have already
  //            been defined in this function. If you want to add a new
  //            constant, use an unused identifier.
  //        - DO NOT add constants that can be easily computed starting form
  //            other ones (avoid redundancy).
  //        Contact the author for modifications.
  //
  //  INPUT:
  //    in      Vector of identifiers of required constants.
  //
  //  OUTPUT:
  //    out     Vector of constants.
  //
  //  EXAMPLE:
  //    astroConstants([2, 4, 26])
  //       Returns a row vector in which there is the value of the AU, the Sun
  //       planetary constant and the mean radius of Saturn.
  //
  //    astroConstants(10 + [1:9])
  //       Returns a row vector with the planetary constant of each planet.
  //
  //  REFERENCES:
  //    - DITAN (Direct Interplanetary Trajectory Analysis), Massimiliano
  //        Vasile, 2006.
  // 	- Wertz J. R., Larson W. J., "Space Mission Analysis and Design", Third
  //        Edition, Space Technology Library 2003.
  //    [A]   DE405 - http://iau-comm4.jpl.nasa.gov/de405iom/de405iom.pdf
  //    [B]   Explanatory Supplement to the Astronomical Almanac. 1992. K. P.
  //          Seidelmann, Ed., p.706 (Table 15.8) and p.316 (Table 5.8.1),
  //          University Science Books, Mill Valley, California.
  //    [C]   Tholen, D.J. and Buie, M.W. 1990. "Further Analysis of
  //          Pluto-Charon Mutual Event Observations" BAAS 22(3):1129.
  //    [D]   Seidelmann, P.K. et al. 2007. "Report of the IAU/IAG Working
  //          Group on cartographic coordinates and rotational elements: 2006"
  //          Celestial Mech. Dyn. Astr. 98:155-180.
  //    [F]   Anderson, J.D., et al. 1987. "The mass, gravity field, and
  //          ephemeris of Mercury" Icarus 71:337-349.
  //    [G]   Konopliv, A.S., et al. 1999. "Venus gravity: 180th degree and
  //          order model" Icarus 139:3-18.
  //    [H]   Folkner, W.M. and Williams, J.G. 2008. "Mass parameters and
  //          uncertainties in planetary ephemeris DE421." Interoffice Memo.
  //          343R-08-004 (internal document), Jet Propulsion Laboratory,
  //          Pasadena, CA.
  //    [I]   Jacobson, R.A. 2008. "Ephemerides of the Martian Satellites -
  //          MAR080" Interoffice Memo. 343R-08-006 (internal document),
  //          Jet Propulsion Laboratory, Pasadena, CA.
  //    [J]   Jacobson, R.A. 2005. "Jovian Satellite ephemeris - JUP230"
  //          private communication.
  //    [K]   Jacobson, R.A., et al. 2006. "The gravity field of the Saturnian
  //          system from satellite observations and spacecraft tracking data"
  //          AJ 132(6):2520-2526.
  //    [L]   Jacobson, R.A. 2007. "The gravity field of the Uranian system and
  //          the orbits of the Uranian satellites and rings" BAAS 39(3):453.
  //    [M]   Jacobson, R.A. 2008. "The orbits of the Neptunian satellites and
  //          the orientation of the pole of Neptune" BAAS 40(2):296.
  //    [N]   Jacobson, R.A. 2007. "The orbits of the satellites of Pluto -
  //          Ephemeris PLU017" private communication.
  //    [W1]  http://ssd.jpl.nasa.gov/?planet_phys_par Last retrieved
  //          20/03/2013
  //    [W2]  http://ssd.jpl.nasa.gov/?sat_phys_par Last retrieved
  //          20/03/2013
  //    [W3]  http://ssd.jpl.nasa.gov/horizons.cgi Last retrieved
  //          20/03/2013
  //    [M1]  Bills, B.G. and Ferrari, A.J. 1977. ``A Harmonic Analysis of
  //          Lunar Topography'', Icarus 31, 244-259.
  //    [M2]  Standish, E. M. 1998. JPL Planetary and Lunar Ephemerides,
  //          DE405/LE405.
  //    [M3]  Lunar Constants and Models Document, Ralph B. Roncoli, 23 Sept
  //    2005,
  //          JPL Technical Document D-32296
  //
  //  CALLED FUNCTIONS:
  //    (none)
  //
  //  AUTHOR:
  //    Matteo Ceriotti, 2006, MATLAB, astroConstants.m
  //
  //  PREVIOUS VERSION:
  //    Matteo Ceriotti, 2006, MATLAB, astro_constants.m, Ver. 1.2
  //        - Header and function name in accordance with guidlines.
  //
  //  CHANGELOG:
  //    26/10/2006, Camilla Colombo: Updated.
  //    22/10/2007, Camilla Colombo: astroConstants(8) added (Obliquity (angle)
  //        of the ecliptic at Epoch 2000).
  //    02/10/2009, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //    12/11/2010, Camilla Colombo: astroConstants(9) added (J2) Note: the
  //        present value of J2 is not consistent with the value of the Earth
  //        radius. This value of J2 should be used in conjunction to Earth
  //        radius = 6378.1363 km
  //    19/03/2013, Camilla Colombo: constants updated to NASA JPL website.
  //        References added.
  //    20/03/2013, REVISION, Francesca Letizia.
  //    22/03/2013, Francesca Letizia: all GM from DE405.
  //
  //  -------------------------------------------------------------------------
  //  9: J2
  //  32: 365.25
  //  out(i)=0.19891000000000E+31*6.67259e-20; % From DITAN
  //  From DE405 [A]
  //  Set launch dates and intercept windows
  st.site = &d_emlrtRSI;
  b_st.site = &d_emlrtRSI;
  //  date2mjd2000.m - Modified Julian day 2000 number from Gregorian date.
  //
  //  PROTOTYPE:
  //    mjd2000 = date2mjd2000(date)
  //
  //  DESCRIPTION:
  //  	Returns the modified Julian day 2000 number of the given date
  //    (Gregorian calendar) plus a fractional part depending on the time of
  //    day.
  //    Note: The function is valid for the whole range of dates since 12:00
  //        noon 24 November -4713, Gregorian calendar. (This bound is set in
  //        order to have symmetry with the inverse function jd2date)
  //    Note: The inputs must be feasible (i.e. the date must exist!). If an
  //        unfeasible date is inputed, wrong results are given because no
  //        check is done on that.
  //
  //  INPUT:
  // 	date[6]     Date in the Gregorian calendar, as a 6-element vector
  //                [year, month, day, hour, minute, second]. For dates before
  //                1582, the resulting date components are valid only in the
  //                Gregorian proleptic calendar. This is based on the
  //                Gregorian calendar but extended to cover dates before its
  //                introduction. date must be after 12:00 noon, 24 November
  //                -4713.
  //
  //  OUTPUT:
  //    mjd2000[1]  Date in MJD 2000. MJD2000 is defined as the number of days
  //                since 01-01-2000, 12:00 noon.
  //
  //  See also mjd20002date.
  //
  //  CALLED FUNCTIONS:
  //     date2jd
  //
  //  AUTHOR:
  //    Nicolas Croisard, 16/02/2008, MATLAB, date2mjd2000.m
  //
  //  CHANGELOG:
  //    03/03/2008, REVISION, Camilla Colombo
  //    22/04/2010, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //
  //  -------------------------------------------------------------------------
  c_st.site = &v_emlrtRSI;
  a = (date2jd(c_st, firstLaunchDate) - 2.4000005E+6) - 51544.5;
  b_st.site = &d_emlrtRSI;
  //  date2mjd2000.m - Modified Julian day 2000 number from Gregorian date.
  //
  //  PROTOTYPE:
  //    mjd2000 = date2mjd2000(date)
  //
  //  DESCRIPTION:
  //  	Returns the modified Julian day 2000 number of the given date
  //    (Gregorian calendar) plus a fractional part depending on the time of
  //    day.
  //    Note: The function is valid for the whole range of dates since 12:00
  //        noon 24 November -4713, Gregorian calendar. (This bound is set in
  //        order to have symmetry with the inverse function jd2date)
  //    Note: The inputs must be feasible (i.e. the date must exist!). If an
  //        unfeasible date is inputed, wrong results are given because no
  //        check is done on that.
  //
  //  INPUT:
  // 	date[6]     Date in the Gregorian calendar, as a 6-element vector
  //                [year, month, day, hour, minute, second]. For dates before
  //                1582, the resulting date components are valid only in the
  //                Gregorian proleptic calendar. This is based on the
  //                Gregorian calendar but extended to cover dates before its
  //                introduction. date must be after 12:00 noon, 24 November
  //                -4713.
  //
  //  OUTPUT:
  //    mjd2000[1]  Date in MJD 2000. MJD2000 is defined as the number of days
  //                since 01-01-2000, 12:00 noon.
  //
  //  See also mjd20002date.
  //
  //  CALLED FUNCTIONS:
  //     date2jd
  //
  //  AUTHOR:
  //    Nicolas Croisard, 16/02/2008, MATLAB, date2mjd2000.m
  //
  //  CHANGELOG:
  //    03/03/2008, REVISION, Camilla Colombo
  //    22/04/2010, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //
  //  -------------------------------------------------------------------------
  c_st.site = &v_emlrtRSI;
  kd = (date2jd(c_st, lastLaunchDate) - 2.4000005E+6) - 51544.5;
  if (muDoubleScalarIsNaN(a) || muDoubleScalarIsNaN(timeStep) ||
      muDoubleScalarIsNaN(kd)) {
    possibleLaunchMjd2000.set_size(&k_emlrtRTEI, &st, 1, 1);
    possibleLaunchMjd2000[0] = rtNaN;
  } else if ((timeStep == 0.0) || ((a < kd) && (timeStep < 0.0)) ||
             ((kd < a) && (timeStep > 0.0))) {
    possibleLaunchMjd2000.set_size(&k_emlrtRTEI, &st, 1, 0);
  } else if ((muDoubleScalarIsInf(a) || muDoubleScalarIsInf(kd)) &&
             (muDoubleScalarIsInf(timeStep) || (a == kd))) {
    possibleLaunchMjd2000.set_size(&k_emlrtRTEI, &st, 1, 1);
    possibleLaunchMjd2000[0] = rtNaN;
  } else if (muDoubleScalarIsInf(timeStep)) {
    possibleLaunchMjd2000.set_size(&k_emlrtRTEI, &st, 1, 1);
    possibleLaunchMjd2000[0] = a;
  } else if ((muDoubleScalarFloor(a) == a) &&
             (muDoubleScalarFloor(timeStep) == timeStep)) {
    n = static_cast<int32_T>((kd - a) / timeStep);
    possibleLaunchMjd2000.set_size(&k_emlrtRTEI, &st, 1, n + 1);
    for (k = 0; k <= n; k++) {
      possibleLaunchMjd2000[k] = a + timeStep * static_cast<real_T>(k);
    }
  } else {
    b_st.site = &cb_emlrtRSI;
    ndbl = muDoubleScalarFloor((kd - a) / timeStep + 0.5);
    apnd = a + ndbl * timeStep;
    if (timeStep > 0.0) {
      cdiff = apnd - kd;
    } else {
      cdiff = kd - apnd;
    }
    if (muDoubleScalarAbs(cdiff) <
        4.4408920985006262E-16 *
            muDoubleScalarMax(muDoubleScalarAbs(a), muDoubleScalarAbs(kd))) {
      ndbl++;
      apnd = kd;
    } else if (cdiff > 0.0) {
      apnd = a + (ndbl - 1.0) * timeStep;
    } else {
      ndbl++;
    }
    if (ndbl >= 0.0) {
      n = static_cast<int32_T>(ndbl);
    } else {
      n = 0;
    }
    c_st.site = &db_emlrtRSI;
    if (ndbl > 2.147483647E+9) {
      emlrtErrorWithMessageIdR2018a(&c_st, &e_emlrtRTEI,
                                    "Coder:MATLAB:pmaxsize",
                                    "Coder:MATLAB:pmaxsize", 0);
    }
    possibleLaunchMjd2000.set_size(&j_emlrtRTEI, &b_st, 1, n);
    if (n > 0) {
      possibleLaunchMjd2000[0] = a;
      if (n > 1) {
        possibleLaunchMjd2000[n - 1] = apnd;
        nm1d2 = (n - 1) / 2;
        for (k = 0; k <= nm1d2 - 2; k++) {
          kd = (static_cast<real_T>(k) + 1.0) * timeStep;
          possibleLaunchMjd2000[k + 1] = a + kd;
          possibleLaunchMjd2000[(n - k) - 2] = apnd - kd;
        }
        if (nm1d2 << 1 == n - 1) {
          possibleLaunchMjd2000[nm1d2] = (a + apnd) / 2.0;
        } else {
          kd = static_cast<real_T>(nm1d2) * timeStep;
          possibleLaunchMjd2000[nm1d2] = a + kd;
          possibleLaunchMjd2000[nm1d2 + 1] = apnd - kd;
        }
      }
    }
  }
  //  All posible launch dates in mjd2000
  //  Iterate each possible launch date
  DV.set_size(&l_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  n = possibleLaunchMjd2000.size(1);
  optDepDV.set_size(&m_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  optIntDV.set_size(&n_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  optGraDV.set_size(&o_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  optFlyDV.set_size(&p_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  for (k = 0; k < n; k++) {
    DV[k] = 100.0;
    optDepDV[k] = 0.0;
    optIntDV[k] = 0.0;
    optGraDV[k] = 0.0;
    optFlyDV[k] = 0.0;
  }
  departureV.set_size(&q_emlrtRTEI, sp, 3, possibleLaunchMjd2000.size(1));
  nm1d2 = 3 * possibleLaunchMjd2000.size(1);
  for (k = 0; k < nm1d2; k++) {
    departureV[k] = 0.0;
  }
  flybyDepartureV.set_size(&r_emlrtRTEI, sp, 3, possibleLaunchMjd2000.size(1));
  for (k = 0; k < nm1d2; k++) {
    flybyDepartureV[k] = 0.0;
  }
  Vinfplus.set_size(&s_emlrtRTEI, sp, 3, possibleLaunchMjd2000.size(1));
  for (k = 0; k < nm1d2; k++) {
    Vinfplus[k] = 0.0;
  }
  Vinfminus.set_size(&t_emlrtRTEI, sp, 3, possibleLaunchMjd2000.size(1));
  for (k = 0; k < nm1d2; k++) {
    Vinfminus[k] = 0.0;
  }
  rpFlyby.set_size(&u_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  n = possibleLaunchMjd2000.size(1);
  delta.set_size(&v_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  VPminus.set_size(&w_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  VPplus.set_size(&x_emlrtRTEI, sp, 1, possibleLaunchMjd2000.size(1));
  for (k = 0; k < n; k++) {
    rpFlyby[k] = 0.0;
    delta[k] = 0.0;
    VPminus[k] = 0.0;
    VPplus[k] = 0.0;
  }
  mjd2000.set_size(&y_emlrtRTEI, sp, possibleLaunchMjd2000.size(1), 3);
  for (k = 0; k < nm1d2; k++) {
    mjd2000[k] = 0.0;
  }
  //  Iterate launch dates
  n = possibleLaunchMjd2000.size(1) - 1;
  emlrtEnterParallelRegion((emlrtCTX)sp,
                           static_cast<boolean_T>(omp_in_parallel()));
  findFlyby_numThreads =
      emlrtAllocRegionTLSs(sp->tls, static_cast<boolean_T>(omp_in_parallel()),
                           omp_get_max_threads(), omp_get_num_procs());
#pragma omp parallel num_threads(findFlyby_numThreads) private(                \
    auxDV, flybyDV, gravityAsistDV, VpPlus, VpMinus, rp, fun, Delta,           \
    interceptDV, interceptDepartureV, a__13, interceptR, interceptKep,         \
    interceptMjd2000, departureDV, flybyTransferArrivalV,                      \
    flybyTransferDepartureV, flybyV, flybyR, flybyMjd2000, launchV, launchR,   \
    launchMjd2000, f_st, d, i, b_flybyMjd2000, r, r1, i1, b_interceptMjd2000,  \
    r2, n_t, incx_t, incy_t)                                                   \
    firstprivate(d_st, e_st, a_data, b_data, emlrtHadParallelError)
  {
    try {

      d_st.prev = sp;
      d_st.tls = emlrtAllocTLS((emlrtCTX)sp, omp_get_thread_num());
      d_st.site = nullptr;
      e_st.prev = &d_st;
      e_st.tls = d_st.tls;
      f_st.prev = &e_st;
      f_st.tls = e_st.tls;
    } catch (...) {
      emlrtHadParallelError = true;
    }
#pragma omp for nowait
    for (launchIndex = 0; launchIndex <= n; launchIndex++) {
      if (emlrtHadParallelError) {
        continue;
      }
      try {

        if (launchIndex + 1 > possibleLaunchMjd2000.size(1)) {
          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                        possibleLaunchMjd2000.size(1),
                                        &h_emlrtBCI, &d_st);
        }
        launchMjd2000 = possibleLaunchMjd2000[launchIndex];
        //  Get launch orbit
        d = possibleLaunchMjd2000[launchIndex];
        e_st.site = &e_emlrtRSI;
        uplanet(e_st, d, 2.0, interceptKep);
        //  Keplerian elements at launch
        e_st.site = &f_emlrtRSI;
        orbitalToCar(e_st, interceptKep[0], interceptKep[1], interceptKep[2],
                     interceptKep[3], interceptKep[4], interceptKep[5], launchR,
                     launchV);
        //  Iterate flyby transfer arrival dates
        flybyDV = flybyWindow + d;
        i = static_cast<int32_T>((flybyDV + (timeStep - (d + 1.0))) / timeStep);
        emlrtForLoopVectorCheckR2021a(d + 1.0, timeStep, flybyDV,
                                      mxDOUBLE_CLASS, i, &b_emlrtRTEI, &d_st);
        for (b_flybyMjd2000 = 0; b_flybyMjd2000 < i; b_flybyMjd2000++) {
          flybyMjd2000 = (launchMjd2000 + 1.0) +
                         static_cast<real_T>(b_flybyMjd2000) * timeStep;
          //  Minimum TOF is TPAR
          e_st.site = &g_emlrtRSI;
          uplanet(e_st, flybyMjd2000, 3.0, interceptKep);
          //  Keplerian elements of Earth at flyby
          e_st.site = &h_emlrtRSI;
          orbitalToCar(e_st, interceptKep[0], interceptKep[1], interceptKep[2],
                       interceptKep[3], interceptKep[4], interceptKep[5],
                       flybyR, flybyV);
          auxDV = (flybyMjd2000 - launchMjd2000) * 86400.0;
          e_st.site = &i_emlrtRSI;
          a__13 = lambertMR(e_st, launchR, flybyR, auxDV, VpMinus, VpPlus, rp,
                            flybyTransferDepartureV, flybyTransferArrivalV,
                            flybyDV, gravityAsistDV);
          if (!(rp != 0.0)) {
            r = _mm_loadu_pd(&launchV[0]);
            r1 = _mm_loadu_pd(&flybyTransferDepartureV[0]);
            _mm_storeu_pd(&interceptR[0], _mm_sub_pd(r, r1));
            interceptR[2] = launchV[2] - flybyTransferDepartureV[2];
            departureDV = coder::b_norm(interceptR);
            if (!(departureDV > 12.0)) {
              //  Iterate possible intercept dates
              d = flybyMjd2000 + interceptWindow;
              i1 = static_cast<int32_T>(
                  (d + (timeStep - (flybyMjd2000 + 1.0))) / timeStep);
              emlrtForLoopVectorCheckR2021a(flybyMjd2000 + 1.0, timeStep, d,
                                            mxDOUBLE_CLASS, i1, &c_emlrtRTEI,
                                            &d_st);
              for (b_interceptMjd2000 = 0; b_interceptMjd2000 < i1;
                   b_interceptMjd2000++) {
                interceptMjd2000 =
                    (flybyMjd2000 + 1.0) +
                    static_cast<real_T>(b_interceptMjd2000) * timeStep;
                e_st.site = &j_emlrtRSI;
                ephNEO(e_st, interceptMjd2000, interceptKep);
                //  Keplerian elements of Asteroid at intercept
                e_st.site = &k_emlrtRSI;
                orbitalToCar(e_st, 1.2745738582873201E+8, 0.446,
                             0.41887902047863906, 4.125958351714595,
                             5.599016240397809, interceptKep[5], interceptR,
                             fun.workspace.VinfMinus);
                gravityAsistDV = (interceptMjd2000 - flybyMjd2000) * 86400.0;
                e_st.site = &l_emlrtRSI;
                flybyDV = lambertMR(e_st, flybyR, interceptR, gravityAsistDV,
                                    auxDV, a__13, rp, interceptDepartureV,
                                    fun.workspace.VinfPlus, VpMinus, VpPlus);
                if (!(rp != 0.0)) {
                  r = _mm_loadu_pd(&fun.workspace.VinfMinus[0]);
                  r1 = _mm_loadu_pd(&fun.workspace.VinfPlus[0]);
                  _mm_storeu_pd(&interceptR[0], _mm_sub_pd(r, r1));
                  interceptR[2] =
                      fun.workspace.VinfMinus[2] - fun.workspace.VinfPlus[2];
                  interceptDV = coder::b_norm(interceptR);
                  d = interceptDV + departureDV;
                  if (!(d > 12.0)) {
                    r = _mm_loadu_pd(&flybyTransferArrivalV[0]);
                    r1 = _mm_loadu_pd(&flybyV[0]);
                    r = _mm_sub_pd(r, r1);
                    _mm_storeu_pd(&fun.workspace.VinfMinus[0], r);
                    r2 = _mm_loadu_pd(&interceptDepartureV[0]);
                    r1 = _mm_sub_pd(r2, r1);
                    _mm_storeu_pd(&fun.workspace.VinfPlus[0], r1);
                    _mm_storeu_pd(&a_data[0], r);
                    _mm_storeu_pd(&b_data[0], r1);
                    flybyDV = flybyV[2];
                    gravityAsistDV = flybyTransferArrivalV[2] - flybyDV;
                    fun.workspace.VinfMinus[2] = gravityAsistDV;
                    flybyDV = interceptDepartureV[2] - flybyDV;
                    fun.workspace.VinfPlus[2] = flybyDV;
                    a_data[2] = gravityAsistDV;
                    b_data[2] = flybyDV;
                    n_t = (ptrdiff_t)3;
                    incx_t = (ptrdiff_t)1;
                    incy_t = (ptrdiff_t)1;
                    flybyDV =
                        ddot(&n_t, &a_data[0], &incx_t, &b_data[0], &incy_t);
                    a__13 = coder::b_norm(fun.workspace.VinfMinus);
                    VpPlus = coder::b_norm(fun.workspace.VinfPlus);
                    flybyDV = muDoubleScalarMax(
                        muDoubleScalarMin(flybyDV / (a__13 * VpPlus), 1.0),
                        -1.0);
                    e_st.site = &m_emlrtRSI;
                    Delta = 57.295779513082323 * muDoubleScalarAcos(flybyDV);
                    fun.workspace.Delta = Delta;
                    e_st.site = &n_emlrtRSI;
                    f_st.site = &sc_emlrtRSI;
                    auxDV = findFlyby_anonFcn1(
                        f_st, Delta, fun.workspace.VinfMinus,
                        fun.workspace.VinfPlus, 923796.45000000007);
                    e_st.site = &n_emlrtRSI;
                    f_st.site = &sc_emlrtRSI;
                    flybyDV =
                        findFlyby_anonFcn1(f_st, Delta, fun.workspace.VinfMinus,
                                           fun.workspace.VinfPlus, 6671.01);
                    if (!(auxDV * flybyDV > 0.0)) {
                      e_st.site = &o_emlrtRSI;
                      rp = coder::fzero(e_st, fun);
                      e_st.site = &p_emlrtRSI;
                      gravityAsistDV = -398600.433 / (a__13 * a__13);
                      e_st.site = &q_emlrtRSI;
                      flybyDV = -398600.433 / (VpPlus * VpPlus);
                      e_st.site = &r_emlrtRSI;
                      VpMinus = 398600.433 * (2.0 / rp - 1.0 / gravityAsistDV);
                      if (VpMinus < 0.0) {
                        emlrtErrorWithMessageIdR2018a(
                            &e_st, &d_emlrtRTEI,
                            "Coder:toolbox:ElFunDomainError",
                            "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
                      }
                      VpMinus = muDoubleScalarSqrt(VpMinus);
                      e_st.site = &s_emlrtRSI;
                      VpPlus = 398600.433 * (2.0 / rp - 1.0 / flybyDV);
                      if (VpPlus < 0.0) {
                        emlrtErrorWithMessageIdR2018a(
                            &e_st, &d_emlrtRTEI,
                            "Coder:toolbox:ElFunDomainError",
                            "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
                      }
                      VpPlus = muDoubleScalarSqrt(VpPlus);
                      gravityAsistDV = muDoubleScalarAbs(VpPlus - VpMinus);
                      r = _mm_loadu_pd(&interceptDepartureV[0]);
                      r1 = _mm_loadu_pd(&flybyTransferArrivalV[0]);
                      _mm_storeu_pd(&interceptR[0], _mm_sub_pd(r, r1));
                      interceptR[2] =
                          interceptDepartureV[2] - flybyTransferArrivalV[2];
                      flybyDV = coder::b_norm(interceptR);
                      auxDV = d + gravityAsistDV;
                      if (launchIndex + 1 > DV.size(1)) {
                        emlrtDynamicBoundsCheckR2012b(
                            launchIndex + 1, 1, DV.size(1), &i_emlrtBCI, &d_st);
                      }
                      if (auxDV < DV[launchIndex]) {
                        if (launchIndex + 1 > VPminus.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        VPminus.size(1),
                                                        &j_emlrtBCI, &d_st);
                        }
                        VPminus[launchIndex] = VpMinus;
                        if (launchIndex + 1 > VPplus.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        VPplus.size(1),
                                                        &k_emlrtBCI, &d_st);
                        }
                        VPplus[launchIndex] = VpPlus;
                        if (launchIndex + 1 > delta.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        delta.size(1),
                                                        &l_emlrtBCI, &d_st);
                        }
                        delta[launchIndex] = Delta;
                        if (launchIndex + 1 > Vinfplus.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        Vinfplus.size(1),
                                                        &d_emlrtBCI, &d_st);
                        }
                        Vinfplus[3 * launchIndex] = fun.workspace.VinfPlus[0];
                        Vinfplus[3 * launchIndex + 1] =
                            fun.workspace.VinfPlus[1];
                        Vinfplus[3 * launchIndex + 2] =
                            fun.workspace.VinfPlus[2];
                        if (launchIndex + 1 > Vinfminus.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        Vinfminus.size(1),
                                                        &e_emlrtBCI, &d_st);
                        }
                        Vinfminus[3 * launchIndex] = fun.workspace.VinfMinus[0];
                        Vinfminus[3 * launchIndex + 1] =
                            fun.workspace.VinfMinus[1];
                        Vinfminus[3 * launchIndex + 2] =
                            fun.workspace.VinfMinus[2];
                        if (launchIndex + 1 > rpFlyby.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        rpFlyby.size(1),
                                                        &m_emlrtBCI, &d_st);
                        }
                        rpFlyby[launchIndex] = rp;
                        if (launchIndex + 1 > DV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        DV.size(1), &n_emlrtBCI,
                                                        &d_st);
                        }
                        DV[launchIndex] = auxDV;
                        if (launchIndex + 1 > optDepDV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        optDepDV.size(1),
                                                        &o_emlrtBCI, &d_st);
                        }
                        optDepDV[launchIndex] = departureDV;
                        if (launchIndex + 1 > optIntDV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        optIntDV.size(1),
                                                        &p_emlrtBCI, &d_st);
                        }
                        optIntDV[launchIndex] = interceptDV;
                        if (launchIndex + 1 > optGraDV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        optGraDV.size(1),
                                                        &q_emlrtBCI, &d_st);
                        }
                        optGraDV[launchIndex] = gravityAsistDV;
                        if (launchIndex + 1 > optFlyDV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        optFlyDV.size(1),
                                                        &r_emlrtBCI, &d_st);
                        }
                        optFlyDV[launchIndex] = flybyDV;
                        if (launchIndex + 1 > mjd2000.size(0)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        mjd2000.size(0),
                                                        &emlrtBCI, &d_st);
                        }
                        mjd2000[launchIndex] = launchMjd2000;
                        mjd2000[launchIndex + mjd2000.size(0)] = flybyMjd2000;
                        mjd2000[launchIndex + mjd2000.size(0) * 2] =
                            interceptMjd2000;
                        if (launchIndex + 1 > departureV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        departureV.size(1),
                                                        &b_emlrtBCI, &d_st);
                        }
                        departureV[3 * launchIndex] =
                            flybyTransferDepartureV[0];
                        departureV[3 * launchIndex + 1] =
                            flybyTransferDepartureV[1];
                        departureV[3 * launchIndex + 2] =
                            flybyTransferDepartureV[2];
                        if (launchIndex + 1 > flybyDepartureV.size(1)) {
                          emlrtDynamicBoundsCheckR2012b(launchIndex + 1, 1,
                                                        flybyDepartureV.size(1),
                                                        &c_emlrtBCI, &d_st);
                        }
                        flybyDepartureV[3 * launchIndex] =
                            interceptDepartureV[0];
                        flybyDepartureV[3 * launchIndex + 1] =
                            interceptDepartureV[1];
                        flybyDepartureV[3 * launchIndex + 2] =
                            interceptDepartureV[2];
                      }
                    }
                  }
                }
                if (*emlrtBreakCheckR2012bFlagVar != 0) {
                  emlrtBreakCheckR2012b(&d_st);
                }
              }
            }
          }
          if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b(&d_st);
          }
        }
        if (*emlrtBreakCheckR2012bFlagVar != 0) {
          emlrtBreakCheckR2012b(&d_st);
        }
      } catch (...) {
        emlrtHadParallelError = true;
      }
    }
  }
  emlrtExitParallelRegion((emlrtCTX)sp,
                          static_cast<boolean_T>(omp_in_parallel()));
  n = DV.size(1) - 1;
  for (nm1d2 = 0; nm1d2 <= n; nm1d2++) {
    if (DV[nm1d2] == 0.0) {
      if (nm1d2 > DV.size(1) - 1) {
        emlrtDynamicBoundsCheckR2012b(nm1d2, 0, DV.size(1) - 1, &f_emlrtBCI,
                                      (emlrtConstCTX)sp);
      }
      DV[nm1d2] = rtNaN;
    }
  }
  n = DV.size(1) - 1;
  for (nm1d2 = 0; nm1d2 <= n; nm1d2++) {
    if (DV[nm1d2] == 100.0) {
      if (nm1d2 > DV.size(1) - 1) {
        emlrtDynamicBoundsCheckR2012b(nm1d2, 0, DV.size(1) - 1, &g_emlrtBCI,
                                      (emlrtConstCTX)sp);
      }
      DV[nm1d2] = rtNaN;
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtConstCTX)sp);
}

real_T findFlyby_anonFcn1(const emlrtStack &sp, real_T Delta,
                          const real_T VinfMinus[3], const real_T VinfPlus[3],
                          real_T x)
{
  emlrtStack st;
  real_T a;
  real_T absxk;
  real_T b_a;
  real_T scale;
  real_T t;
  st.prev = &sp;
  st.tls = sp.tls;
  scale = 3.3121686421112381E-170;
  absxk = muDoubleScalarAbs(VinfMinus[0]);
  if (absxk > 3.3121686421112381E-170) {
    a = 1.0;
    scale = absxk;
  } else {
    t = absxk / 3.3121686421112381E-170;
    a = t * t;
  }
  absxk = muDoubleScalarAbs(VinfMinus[1]);
  if (absxk > scale) {
    t = scale / absxk;
    a = a * t * t + 1.0;
    scale = absxk;
  } else {
    t = absxk / scale;
    a += t * t;
  }
  absxk = muDoubleScalarAbs(VinfMinus[2]);
  if (absxk > scale) {
    t = scale / absxk;
    a = a * t * t + 1.0;
    scale = absxk;
  } else {
    t = absxk / scale;
    a += t * t;
  }
  a = scale * muDoubleScalarSqrt(a);
  scale = 3.3121686421112381E-170;
  absxk = muDoubleScalarAbs(VinfPlus[0]);
  if (absxk > 3.3121686421112381E-170) {
    b_a = 1.0;
    scale = absxk;
  } else {
    t = absxk / 3.3121686421112381E-170;
    b_a = t * t;
  }
  absxk = muDoubleScalarAbs(VinfPlus[1]);
  if (absxk > scale) {
    t = scale / absxk;
    b_a = b_a * t * t + 1.0;
    scale = absxk;
  } else {
    t = absxk / scale;
    b_a += t * t;
  }
  absxk = muDoubleScalarAbs(VinfPlus[2]);
  if (absxk > scale) {
    t = scale / absxk;
    b_a = b_a * t * t + 1.0;
    scale = absxk;
  } else {
    t = absxk / scale;
    b_a += t * t;
  }
  b_a = scale * muDoubleScalarSqrt(b_a);
  st.site = &tc_emlrtRSI;
  absxk = 1.0 / (x * (a * a) / 398600.433 + 1.0);
  if ((absxk < -1.0) || (absxk > 1.0)) {
    emlrtErrorWithMessageIdR2018a(
        &st, &i_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 4, "asin");
  }
  absxk = muDoubleScalarAsin(absxk);
  st.site = &tc_emlrtRSI;
  scale = 1.0 / (x * (b_a * b_a) / 398600.433 + 1.0);
  if ((scale < -1.0) || (scale > 1.0)) {
    emlrtErrorWithMessageIdR2018a(
        &st, &i_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 4, "asin");
  }
  scale = muDoubleScalarAsin(scale);
  return (Delta * 3.1415926535897931 / 180.0 - absxk) - scale;
}

// End of code generation (findFlyby.cpp)
