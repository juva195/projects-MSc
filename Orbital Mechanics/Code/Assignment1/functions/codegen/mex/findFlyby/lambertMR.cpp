//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lambertMR.cpp
//
// Code generation for function 'lambertMR'
//

// Include files
#include "lambertMR.h"
#include "dot.h"
#include "findFlyby_data.h"
#include "norm.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"
#include <cmath>
#include <emmintrin.h>

// Variable Definitions
static emlrtRSInfo t_emlrtRSI{
    44,       // lineNo
    "mpower", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\matfun\\mpower.m" // pathName
};

static emlrtRSInfo
    u_emlrtRSI{
        71,      // lineNo
        "power", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\ops\\power.m" // pathName
    };

static emlrtRSInfo ib_emlrtRSI{
    203,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo jb_emlrtRSI{
    205,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo kb_emlrtRSI{
    237,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo lb_emlrtRSI{
    239,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo mb_emlrtRSI{
    240,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo nb_emlrtRSI{
    242,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo ob_emlrtRSI{
    254,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo pb_emlrtRSI{
    255,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo qb_emlrtRSI{
    256,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo rb_emlrtRSI{
    259,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo sb_emlrtRSI{
    262,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo tb_emlrtRSI{
    263,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo ub_emlrtRSI{
    296,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo vb_emlrtRSI{
    311,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo wb_emlrtRSI{
    317,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo xb_emlrtRSI{
    324,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo yb_emlrtRSI{
    327,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo ac_emlrtRSI{
    331,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo bc_emlrtRSI{
    335,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo cc_emlrtRSI{
    336,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo dc_emlrtRSI{
    377,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo ec_emlrtRSI{
    378,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo fc_emlrtRSI{
    380,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo gc_emlrtRSI{
    681,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo hc_emlrtRSI{
    682,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo ic_emlrtRSI{
    684,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo jc_emlrtRSI{
    686,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo kc_emlrtRSI{
    691,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRSInfo lc_emlrtRSI{
    692,         // lineNo
    "lambertMR", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\lambertMR.m" // pathName
};

static emlrtRTEInfo
    f_emlrtRTEI{
        82,         // lineNo
        5,          // colNo
        "fltpower", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2023b\\toolbox\\eml\\lib\\matlab\\ops\\power.m" // pName
    };

// Function Definitions
real_T lambertMR(const emlrtStack &sp, const real_T RI[3], const real_T RF[3],
                 real_T TOF, real_T &P, real_T &E, real_T &ERROR, real_T VI[3],
                 real_T VF[3], real_T &TPAR, real_T &THETA)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T A;
  real_T CTH;
  real_T RFM;
  real_T RFM2;
  real_T RIM;
  real_T RIM2;
  real_T STH;
  int32_T CHECKFEAS;
  st.prev = &sp;
  st.tls = sp.tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  //  lambertMR.m - Lambert's problem solver for all possible transfers
  //    (multi-revolution transfer included).
  //
  //  PROTOTYPE:
  //    [A,P,E,ERROR,VI,VF,TPAR,THETA] =
  //    lambertMR(RI,RF,TOF,MU,orbitType,Nrev,Ncase,optionsLMR)
  //
  //  DESCRIPTION:
  //    Lambert's problem solver for all possible transfers:
  //        1- zero-revolution (for all possible types of orbits: circles,
  //        ellipses, 	parabolas and hyperbolas) 2- multirevolution case 3-
  //        inversion of the motion
  //
  //    1- ZERO-REVOLUTION LAMBERT'S PROBLEM
  //
  //    For the solution of Lambert's problem with number of revolution = 0 the
  //    subroutine by Chris D'Souza is included here.
  //    This subroutine is a Lambert algorithm which given two radius vectors
  //    and the time to get from one to the other, it finds the orbit
  //    connecting the two. It solves the problem using a new algorithm
  //    developed by R. Battin. It solves the Lambert problem for all possible
  //    types of orbits (circles, ellipses, parabolas and hyperbolas).
  //    The only singularity is for the case of a transfer angle of 360 degrees,
  //    which is a rather obscure case.
  //    It computes the velocity vectors corresponding to the given radius
  //    vectors except for the case when the transfer angle is 180 degrees
  //    in which case the orbit plane is ambiguous (an infinite number of
  //    transfer orbits exist).
  //
  //    2- MULTIREVOLUTION LAMBERT'S PROBLEM
  //
  //    For the solution of Lambert's problem with Nrev>0 number of revolution,
  //    Battin's formulation has been extended to accomodate N-revolution
  //    transfer orbits, by following the paper: "Using Battin Mathod to obtain
  //    Multiple-revolution Lambert's Solutions" by Shen and Tsiotras.
  //
  //    When Nrev>0 the possible orbits are just ellipses.
  //    If 0<=Nrev<=Nmax, there are two Nrev-revolution transfer orbits.
  //    These two transfer orbits have different semi-major axis and they may
  //    be all combinations of large-e and small-e transfer orbits.
  //    The Original Successive Substitution Method by Battin converges to one
  //    of the two possible solution with a viable initial guest, however it
  //    diverges from the other one. Then a Reversed Successive Substitution is
  //    used to converge to the second solution.
  //    A procedure is implemented in order to guarantee to provide initial
  //    guesses in the convergence region. If Nrew exceeds the maximum number
  //    of revolution an ERROR is given:
  //    warning('off','lambertMR:SuccessiveSubstitutionDiverged') to take out
  //    the warnings or use optionsLMR(1) = 0.
  //
  //    3- INVERSION OF THE MOTION
  //
  //    Direct or retrograde option can be selected for the transfer.
  //
  //    The algorithm computes the semi-major axis, the parameter (semi-latus
  //    rectum), the eccentricity and the velocity vectors.
  //
  //    NOTE: If ERROR occurs or the 360 or 180 degree transfer case is
  //    encountered.
  //
  //  INPUT:
  // 	RI[3]           Vector containing the initial position in Cartesian
  //                    coordinates [L].
  // 	RF[3]           Vector containing the final position vector in
  //                    Cartesian coordinates [L].
  // 	TOF[1]          Transfer time, time of flight [T].
  //   	MU[1]           Planetary constant of the planet (mu = mass * G)
  //   [L^3/T^2]
  // 	orbitType[1]    Logical variable defining whether transfer is
  //                        0: direct transfer from R1 to R2 (counterclockwise)
  //                        1: retrograde transfer from R1 to R2 (clockwise)
  // 	Nrev[1]         Number of revolutions.
  //                    if Nrev = 0 ZERO-REVOLUTION transfer is calculated
  //                    if Nrev > 0 two transfers are possible. Ncase should be
  //                           defined to select one of the two.
  // 	Ncase[1]        Logical variable defining the small-a or large-a option
  //                    in case of Nrev>0:
  //                        0: small-a option
  //                        1: large-a option
  // 	optionsLMR[1]	lambertMR options:
  //                     optionsLMR(1) = display options:
  //                                     0: no display
  //                                     1: warnings are displayed only when
  //                                        the algorithm does not converge
  //                                     2: full warnings displayed
  //
  //  OUTPUT:
  // 	A[1]        Semi-major axis of the transfer orbit [L].
  //  	P[1]        Semi-latus rectum of the transfer orbit [L].
  //   	E[1]        Eccentricity of the transfer orbit.
  // 	ERROR[1]	Error flag
  //                    0:	No error
  //                    1:	Error, routine failed to converge
  //                    -1:	180 degrees transfer
  //                    2:  360 degrees transfer
  //                    3:  the algorithm doesn't converge because the number
  //                        of revolutions is bigger than Nrevmax for that TOF
  //                    4:  Routine failed to converge, maximum number of
  //                        iterations exceeded.
  // 	VI[3]       Vector containing the initial velocity vector in Cartesian
  //                coordinates [L/T].
  // 	VT[1]		Vector containing the final velocity vector in Cartesian
  //                coordinates [L/T].
  // 	TPAR[1] 	Parabolic flight time between RI and RF [T].
  // 	THETA[1]	Transfer angle [radians].
  //
  //  NOTE: The semi-major axis, positions, times, and gravitational parameter
  //        must be in compatible units.
  //
  //  CALLED FUNCTIONS:
  //    qck, h_E (added at the bottom of this file)
  //
  //  REFERENCES:
  //    - Shen and Tsiotras, "Using Battin method to obtain Multiple-Revolution
  //        Lambert's solutions".
  //    - Battin R., "An Introduction to the Mathematics and Methods of
  //        Astrodynamics, Revised Edition", 1999.
  //
  //  FUTURE DEVELOPMENT:
  //    - 180 degrees transfer indetermination
  //    - 360 degrees transfer singularity
  //    - Nmax number of max revolution for a given TOF:
  //      work in progress - Camilla Colombo
  //
  //  ORIGINAL VERSION:
  //    Chris D'Souza, 20/01/1989, MATLAB, lambert.m
  //        verified by Darrel Monroe, 10/25/90
  //        - Labert.m solved only direct transfer, without multi-revolution
  //          option
  //
  //  AUTHOR:
  //    Camilla Colombo, 10/11/2006, MATLAB, lambertMR.m
  //
  //  CHANGELOG:
  //    13/11/2006, Camilla Colombo: added ERROR = 3 if Nrev > NrevMAX
  // 	21/11/2006, Camilla Colombo: added another case of ERROR = 3 (index
  //    	N3) corresponding to the limit case when small-a solution =
  //    large-a
  //        solution. No solution is given in this case.
  // 	06/08/2007, Camilla Colombo: optionsLMR added as an input
  // 	28/11/2007, Camilla Colombo: minor changes
  //    29/01/2009, Matteo Ceriotti:
  //        - Introduced variable for maximum number of iterations nitermax.
  //        - Corrected final check on maximum number of iterations exceeded,
  //        from
  //            "==" to ">=" (if N1 >= nitermax || N >= nitermax).
  //        - Increased maxumum number of iterations to 2000, not to lose some
  //            solutions.
  //        - In OSS loop, added check for maximum number of iterations
  //        exceeded,
  //            which then sets checkNconvOSS = 0.
  //        - Changed the way of coumputing X given Y1 in RSS. Now the
  //            Newton-Raphson method with initial guess suggested by Shen,
  //            Tsiotras is used. This should guarantee convergence without the
  //            need of an external zero finder (fsolve).
  //        - Changed absolute tolerance into relative tolerance in all loops
  //        X0-X.
  //            Now the condition is: while "abs(X0-X) >= abs(X)*TOL+TOL".
  //        - Added return immediately when any error is detected.
  //        - Moved check on 4*TOF*LAMBDA==0 after computing LAMBDA.
  //        - Moved check on THETA==0 || THETA==2*PI after computing THETA.
  //        - Added error code 4 (number of iterations exceeded).
  //        - Removed variable Nwhile, as not strictly needed.
  //        - Removed variable PIE=pi.
  //    29/01/2009, REVISION: Matteo Ceriotti
  //    21/07/2009, Matteo Ceriotti, Camilla Colombo:
  //        added condition to detect case 180 degrees transfer indetermination
  //    30/01/2010, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //
  //  Note: Please if you have got any changes that you would like to be done,
  //    do not change the function, please contact the author.
  //
  //  -------------------------------------------------------------------------
  //  Check inputs
  //  Maximum number of iterations for loops
  //  Reset
  //  ----------------------------------
  //  Compute the vector magnitudes and various cross and dot products
  RIM2 = coder::dot(RI, RI);
  st.site = &ib_emlrtRSI;
  if (RIM2 < 0.0) {
    emlrtErrorWithMessageIdR2018a(
        &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
  }
  RIM = muDoubleScalarSqrt(RIM2);
  RFM2 = coder::dot(RF, RF);
  st.site = &jb_emlrtRSI;
  if (RFM2 < 0.0) {
    emlrtErrorWithMessageIdR2018a(
        &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
  }
  RFM = muDoubleScalarSqrt(RFM2);
  STH = RIM * RFM;
  CTH = coder::dot(RI, RF) / STH;
  VI[0] = RI[1] * RF[2] - RF[1] * RI[2];
  VI[1] = RF[0] * RI[2] - RI[0] * RF[2];
  VI[2] = RI[0] * RF[1] - RF[0] * RI[1];
  STH = coder::b_norm(VI) / STH;
  //  Choose angle for up angular momentum
  //  direct transfer
  if (VI[2] < 0.0) {
    STH = -STH;
  }
  THETA = muDoubleScalarAtan2(STH, CTH);
  //  -------------------------------------------------------------------------
  //  qck.m - Reduce an angle between 0 and 2*pi
  //
  //  PROTOTYPE:
  //    [angle]=qck(angle)
  //
  //  DESCRIPTION:
  //    This function takes any angle and reduces it, if necessary,
  //  	so that it lies in the range from 0 to 2 PI radians.
  //
  //  INPUTS:
  //    ANGLE[1]    Angle to be reduced (in radians)
  //
  //  OUTPUTS:
  //    QCK[1]      The angle reduced, if necessary, to the range
  //                from 0 to 2 PI radians (in radians)
  //
  //  CALLED FUNCTIONS:
  //    pi (from MATLAB)
  //
  //  AUTHOR:
  //    W.T. Fowler, July, 1978
  //
  //  CHANGELOG:
  //    8/20/90, REVISION: Darrel Monroe
  //
  //  -------------------------------------------------------------------------
  if (muDoubleScalarSign(THETA) < 0.0) {
    CHECKFEAS = -1;
  } else {
    CHECKFEAS = 0;
  }
  THETA -= 6.2831853071795862 * (std::trunc(THETA / 6.2831853071795862) +
                                 static_cast<real_T>(CHECKFEAS));
  //  if abs(THETA - pi) >= 0.01
  if ((THETA == 6.2831853071795862) || (THETA == 0.0)) {
    ERROR = 2.0;
    A = 0.0;
    P = 0.0;
    E = 0.0;
    VI[0] = 0.0;
    VF[0] = 0.0;
    VI[1] = 0.0;
    VF[1] = 0.0;
    VI[2] = 0.0;
    VF[2] = 0.0;
    TPAR = 0.0;
    THETA = 0.0;
  } else {
    real_T B1;
    real_T C_tmp;
    real_T LAMBDA;
    real_T S;
    real_T Y;
    real_T d;
    real_T d1;
    B1 = muDoubleScalarSign(STH);
    if (STH == 0.0) {
      B1 = 1.0;
    }
    //  ----------------------------------
    //  Compute the chord and the semi-perimeter
    C_tmp = 2.0 * RIM * RFM;
    STH = (RIM2 + RFM2) - C_tmp * CTH;
    st.site = &kb_emlrtRSI;
    if (STH < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
    }
    STH = muDoubleScalarSqrt(STH);
    S = ((RIM + RFM) + STH) / 2.0;
    RFM2 = S - STH;
    d = RFM2 / S;
    st.site = &lb_emlrtRSI;
    if (d < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
    }
    Y = muDoubleScalarSqrt(d);
    st.site = &lb_emlrtRSI;
    if ((Y < -1.0) || (Y > 1.0)) {
      emlrtErrorWithMessageIdR2018a(
          &st, &i_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 4, "asin");
    }
    st.site = &mb_emlrtRSI;
    b_st.site = &t_emlrtRSI;
    Y = muDoubleScalarPower(S, 3.0);
    d1 = Y / 1.061699520143896E+12;
    st.site = &mb_emlrtRSI;
    if (d1 < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
    }
    st.site = &nb_emlrtRSI;
    if (d < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
    }
    LAMBDA = B1 * muDoubleScalarSqrt(d);
    d1 = 4.0 * TOF * LAMBDA;
    if ((d1 == 0.0) || (muDoubleScalarAbs(d) < 1.0E-14)) {
      ERROR = -1.0;
      A = 0.0;
      P = 0.0;
      E = 0.0;
      VI[0] = 0.0;
      VF[0] = 0.0;
      VI[1] = 0.0;
      VF[1] = 0.0;
      VI[2] = 0.0;
      VF[2] = 0.0;
      TPAR = 0.0;
      THETA = 0.0;
    } else {
      real_T L;
      real_T L1_tmp;
      real_T M;
      real_T X;
      real_T X0;
      int32_T N;
      int32_T N1;
      //  ----------------------------------
      //  Compute L carefully for transfer angles less than 5 degrees
      if (THETA * 180.0 / 3.1415926535897931 <= 5.0) {
        st.site = &ob_emlrtRSI;
        STH = RFM / RIM;
        b_st.site = &t_emlrtRSI;
        c_st.site = &u_emlrtRSI;
        if (STH < 0.0) {
          emlrtErrorWithMessageIdR2018a(&c_st, &f_emlrtRTEI,
                                        "Coder:toolbox:power_domainError",
                                        "Coder:toolbox:power_domainError", 0);
        }
        st.site = &pb_emlrtRSI;
        RIM2 = muDoubleScalarSin(THETA / 4.0);
        b_st.site = &t_emlrtRSI;
        st.site = &qb_emlrtRSI;
        STH = muDoubleScalarTan(
            2.0 * (muDoubleScalarAtan(muDoubleScalarPower(STH, 0.25)) -
                   0.78539816339744828));
        b_st.site = &t_emlrtRSI;
        STH = RIM2 * RIM2 + STH * STH;
        L = STH / (STH + muDoubleScalarCos(THETA / 2.0));
      } else {
        st.site = &rb_emlrtRSI;
        STH = (1.0 - LAMBDA) / (LAMBDA + 1.0);
        b_st.site = &t_emlrtRSI;
        L = STH * STH;
      }
      st.site = &sb_emlrtRSI;
      b_st.site = &t_emlrtRSI;
      st.site = &sb_emlrtRSI;
      b_st.site = &t_emlrtRSI;
      st.site = &sb_emlrtRSI;
      b_st.site = &t_emlrtRSI;
      M = 1.061699520143896E+12 * (TOF * TOF) /
          (Y * muDoubleScalarPower(LAMBDA + 1.0, 6.0));
      st.site = &tb_emlrtRSI;
      st.site = &tb_emlrtRSI;
      b_st.site = &t_emlrtRSI;
      c_st.site = &u_emlrtRSI;
      if (S < 0.0) {
        emlrtErrorWithMessageIdR2018a(&c_st, &f_emlrtRTEI,
                                      "Coder:toolbox:power_domainError",
                                      "Coder:toolbox:power_domainError", 0);
      }
      st.site = &tb_emlrtRSI;
      b_st.site = &t_emlrtRSI;
      c_st.site = &u_emlrtRSI;
      if (RFM2 < 0.0) {
        emlrtErrorWithMessageIdR2018a(&c_st, &f_emlrtRTEI,
                                      "Coder:toolbox:power_domainError",
                                      "Coder:toolbox:power_domainError", 0);
      }
      TPAR = 1.2940108746136426E-6 * (muDoubleScalarPower(S, 1.5) -
                                      B1 * muDoubleScalarPower(RFM2, 1.5));
      L1_tmp = (1.0 - L) / 2.0;
      CHECKFEAS = 0;
      //  ----------------------------------
      //  Initialize values of y, n, and x
      Y = 1.0;
      N = 0;
      N1 = 0;
      ERROR = 0.0;
      //  CHECKFEAS=0;
      if (TOF - TPAR <= 0.001) {
        X0 = 0.0;
      } else {
        X0 = L;
      }
      X = -1.0E+8;
      //  ----------------------------------
      //  Begin iteration
      //  ---> CL: 26/01/2009, Matteo Ceriotti:
      //        Changed absolute tolerance into relative tolerance here below.
      while ((muDoubleScalarAbs(X0 - X) >=
              muDoubleScalarAbs(X) * 1.0E-14 + 1.0E-14) &&
             (N <= 2000)) {
        int32_T GAMMA_tmp;
        N++;
        X = X0;
        st.site = &ub_emlrtRSI;
        b_st.site = &ub_emlrtRSI;
        if (X0 + 1.0 < 0.0) {
          emlrtErrorWithMessageIdR2018a(
              &b_st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
              "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
        }
        d = muDoubleScalarSqrt(X0 + 1.0);
        b_st.site = &t_emlrtRSI;
        STH = X0 / ((d + 1.0) * (d + 1.0));
        CHECKFEAS = 1;
        //  ----------------------------------
        //  Compute x by means of an algorithm devised by
        //  Gauticci for evaluating continued fractions by the
        //  'Top Down' method
        RFM2 = 1.0;
        B1 = 1.0;
        CTH = 1.0;
        N1 = 3;
        while ((muDoubleScalarAbs(B1) > 1.0E-14) && (N1 - 3 <= 2000)) {
          N1++;
          st.site = &vb_emlrtRSI;
          b_st.site = &t_emlrtRSI;
          st.site = &vb_emlrtRSI;
          b_st.site = &t_emlrtRSI;
          GAMMA_tmp = N1 * N1;
          RFM2 = 1.0 / (static_cast<real_T>(GAMMA_tmp) /
                            (4.0 * static_cast<real_T>(GAMMA_tmp) - 1.0) * STH *
                            RFM2 +
                        1.0);
          B1 *= RFM2 - 1.0;
          CTH += B1;
          if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b((emlrtConstCTX)&sp);
          }
        }
        st.site = &wb_emlrtRSI;
        if (X0 + 1.0 < 0.0) {
          emlrtErrorWithMessageIdR2018a(
              &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
              "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
        }
        RIM2 = 8.0 * (muDoubleScalarSqrt(X0 + 1.0) + 1.0) /
               (1.0 / ((STH + 5.0) + 9.0 * STH / 7.0 * CTH) + 3.0);
        //  ----------------------------------
        //  Compute H1 and H2
        if (N == 1) {
          STH = ((2.0 * X0 + 1.0) + L) * ((3.0 * RIM2 + X0 * RIM2) + 4.0 * X0);
          st.site = &xb_emlrtRSI;
          RFM2 = L + X0;
          b_st.site = &t_emlrtRSI;
          Y = RFM2 * RFM2 * ((RIM2 + 1.0) + 3.0 * X0) / STH;
          STH = M * ((RIM2 + X0) - L) / STH;
        } else {
          st.site = &yb_emlrtRSI;
          b_st.site = &t_emlrtRSI;
          st.site = &yb_emlrtRSI;
          b_st.site = &t_emlrtRSI;
          STH = L1_tmp * L1_tmp + M / (Y * Y);
          st.site = &yb_emlrtRSI;
          if (STH < 0.0) {
            emlrtErrorWithMessageIdR2018a(
                &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
                "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
          }
          STH = muDoubleScalarSqrt(STH);
          RFM2 = STH - L1_tmp;
          STH = 2.0 * STH * ((3.0 * RIM2 + X0 * RIM2) + 4.0 * X0);
          st.site = &ac_emlrtRSI;
          b_st.site = &t_emlrtRSI;
          Y = RFM2 * RFM2 * ((RIM2 + 1.0) + 3.0 * X0) / STH;
          STH = M * ((RIM2 + X0) - L) / STH;
        }
        st.site = &bc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        RIM2 = 27.0 * STH / (4.0 * muDoubleScalarPower(Y + 1.0, 3.0));
        st.site = &cc_emlrtRSI;
        if (RIM2 + 1.0 < 0.0) {
          emlrtErrorWithMessageIdR2018a(
              &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
              "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
        }
        B1 = -RIM2 / (2.0 * (muDoubleScalarSqrt(RIM2 + 1.0) + 1.0));
        //  ----------------------------------
        //  Compute the continued fraction expansion K(u)
        //  by means of the 'Top Down' method
        //  Y can be computed finding the roots of the formula and selecting
        //  the real one:
        //  y^3 - (1+h1)*y^2 - h2 = 0     (7.113) Battin
        //
        //  Ycami_ = roots([1 -1-H1 0 -H2])
        //  kcami = find( abs(imag(Ycami_)) < eps );
        //  Ycami = Ycami_(kcami)
        RFM2 = 1.0;
        STH = 1.0;
        CTH = 1.0;
        N1 = 0;
        while ((N1 < 2000) && (muDoubleScalarAbs(STH) >= 1.0E-14)) {
          if (N1 == 0) {
            RFM2 = 1.0 / (1.0 - 0.14814814814814814 * B1 * RFM2);
            STH *= RFM2 - 1.0;
            CTH += STH;
          } else {
            GAMMA_tmp = N1 << 2;
            RFM2 =
                1.0 /
                (1.0 - static_cast<real_T>(((3 * N1 + 1) << 1) * (6 * N1 - 1)) /
                           static_cast<real_T>(9 * (GAMMA_tmp - 1) *
                                               (GAMMA_tmp + 1)) *
                           B1 * RFM2);
            STH *= RFM2 - 1.0;
            CTH += STH;
            if (*emlrtBreakCheckR2012bFlagVar != 0) {
              emlrtBreakCheckR2012b((emlrtConstCTX)&sp);
            }
            RFM2 =
                1.0 /
                (1.0 - static_cast<real_T>(((3 * N1 + 2) << 1) * (6 * N1 + 1)) /
                           static_cast<real_T>(9 * (GAMMA_tmp + 1) *
                                               (GAMMA_tmp + 3)) *
                           B1 * RFM2);
            STH *= RFM2 - 1.0;
            CTH += STH;
            if (*emlrtBreakCheckR2012bFlagVar != 0) {
              emlrtBreakCheckR2012b((emlrtConstCTX)&sp);
            }
          }
          N1++;
          if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b((emlrtConstCTX)&sp);
          }
        }
        st.site = &dc_emlrtRSI;
        RFM2 = CTH / 3.0;
        b_st.site = &t_emlrtRSI;
        st.site = &ec_emlrtRSI;
        if (RIM2 + 1.0 < 0.0) {
          emlrtErrorWithMessageIdR2018a(
              &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
              "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
        }
        Y = (Y + 1.0) / 3.0 *
            (muDoubleScalarSqrt(RIM2 + 1.0) / (1.0 - 2.0 * B1 * (RFM2 * RFM2)) +
             2.0);
        //  Y = Ycami
        st.site = &fc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        st.site = &fc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        d = L1_tmp * L1_tmp + M / (Y * Y);
        st.site = &fc_emlrtRSI;
        if (d < 0.0) {
          emlrtErrorWithMessageIdR2018a(
              &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
              "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
        }
        d = muDoubleScalarSqrt(d);
        X0 = d - (L + 1.0) / 2.0;
        //  fprintf('n= %d, x0=%.14f\n',N,X0);
        if (*emlrtBreakCheckR2012bFlagVar != 0) {
          emlrtBreakCheckR2012b((emlrtConstCTX)&sp);
        }
      }
      //  MULTIREVOLUTION
      //  ----------------------------------
      //  Compute the velocity vectors
      if (CHECKFEAS == 0) {
        ERROR = 1.0;
        A = 0.0;
        P = 0.0;
        E = 0.0;
        VI[0] = 0.0;
        VF[0] = 0.0;
        VI[1] = 0.0;
        VF[1] = 0.0;
        VI[2] = 0.0;
        VF[2] = 0.0;
        TPAR = 0.0;
        THETA = 0.0;
      } else if ((N1 >= 2000) || (N >= 2000)) {
        ERROR = 4.0;
        A = 0.0;
        P = 0.0;
        E = 0.0;
        VI[0] = 0.0;
        VF[0] = 0.0;
        VI[1] = 0.0;
        VF[1] = 0.0;
        VI[2] = 0.0;
        VF[2] = 0.0;
        TPAR = 0.0;
        THETA = 0.0;
      } else {
        __m128d r;
        __m128d r1;
        __m128d r2;
        __m128d r3;
        __m128d r4;
        st.site = &gc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        STH = (LAMBDA + 1.0) * (LAMBDA + 1.0);
        CTH = M * S * STH;
        st.site = &hc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        B1 = Y * Y;
        A = CTH / (8.0 * X0 * B1);
        st.site = &ic_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        RFM2 = STH / d1;
        STH = Y * (X0 + 1.0);
        st.site = &jc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        RIM2 = CTH / STH;
        r = _mm_loadu_pd(&RI[0]);
        r1 = _mm_loadu_pd(&RF[0]);
        r2 = _mm_mul_pd(_mm_set1_pd(STH), _mm_sub_pd(r, r1));
        r3 = _mm_set1_pd(-RFM2);
        r4 = _mm_set1_pd(RIM2);
        _mm_storeu_pd(
            &VI[0],
            _mm_mul_pd(r3, _mm_sub_pd(r2, _mm_div_pd(_mm_mul_pd(r4, r),
                                                     _mm_set1_pd(RIM)))));
        _mm_storeu_pd(
            &VF[0],
            _mm_mul_pd(r3, _mm_add_pd(r2, _mm_div_pd(_mm_mul_pd(r4, r1),
                                                     _mm_set1_pd(RFM)))));
        d = RI[2];
        Y = RF[2];
        d1 = STH * (d - Y);
        VI[2] = -RFM2 * (d1 - RIM2 * d / RIM);
        d1 = -RFM2 * (d1 + RIM2 * Y / RFM);
        VF[2] = d1;
        st.site = &kc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        st.site = &kc_emlrtRSI;
        b_st.site = &t_emlrtRSI;
        st.site = &kc_emlrtRSI;
        RFM2 = muDoubleScalarSin(THETA / 2.0);
        b_st.site = &t_emlrtRSI;
        P = C_tmp * B1 * ((X0 + 1.0) * (X0 + 1.0)) * (RFM2 * RFM2) / CTH;
        E = 1.0 - P / A;
        st.site = &lc_emlrtRSI;
        if (E < 0.0) {
          emlrtErrorWithMessageIdR2018a(
              &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
              "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
        }
        E = muDoubleScalarSqrt(E);
      }
    }
  }
  return A;
}

// End of code generation (lambertMR.cpp)
