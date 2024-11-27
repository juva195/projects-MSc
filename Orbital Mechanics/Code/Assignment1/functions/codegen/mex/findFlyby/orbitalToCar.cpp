//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// orbitalToCar.cpp
//
// Code generation for function 'orbitalToCar'
//

// Include files
#include "orbitalToCar.h"
#include "findFlyby_data.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"
#include <emmintrin.h>

// Variable Definitions
static emlrtRSInfo hb_emlrtRSI{
    51,             // lineNo
    "orbitalToCar", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\orbitalToCar.m" // pathName
};

// Function Definitions
void orbitalToCar(const emlrtStack &sp, real_T a, real_T e, real_T i,
                  real_T Omega, real_T omega, real_T theta, real_T r[3],
                  real_T v[3])
{
  emlrtStack st;
  real_T R1[9];
  real_T R2[9];
  real_T y[9];
  real_T d_a[3];
  real_T R1_tmp;
  real_T b_a;
  real_T c_a;
  real_T d;
  real_T d1;
  real_T p;
  real_T rNorm_tmp;
  real_T rPF_tmp;
  int32_T b_i;
  int32_T coffset;
  st.prev = &sp;
  st.tls = sp.tls;
  //
  //  Conversion from Keplerian elements to Cartesian coordinates.
  //  Angles must be given in radians.
  //
  //  PROTOTYPE:
  //  [r, v] = orbitalToCar(a,e,i,Omega,omega,theta)
  //
  //  If default values are desired insert an empty vector ([]) as the
  //  corresponding value
  //
  //  INPUT:
  //  a         [1x1]         Semi-major axis [km] e         [1x1]-[3x1]
  //  Eccentricity [-] i         [1x1]         Inclination [rad] Omega     [1x1]
  //  RAAN [rad] omega     [1x1]         Pericentre anomaly [rad] theta [1x1]
  //  True anomaly [rad] mu        [1x1]         Gravitational parameter of the
  //  planet (for default mu = 3.98600433e+05)             [km^3/s^2]
  //
  //  OUTPUT:
  //  r         [3x1]   Position vector [km]
  //  v         [3x1]   Velocity vector [km/s]
  //  VALUE CHECK
  //  R,V PERIFOCAL SYSTEM
  if ((e >= 0.0) && (e < 1.0)) {
    //  Elliptic
    p = a * (1.0 - e * e);
    //  Semilat rett
  } else if (e == 1.0) {
    //  Parabolic
    p = 2.0 * a;
  } else {
    p = a * (e * e - 1.0);
  }
  rNorm_tmp = muDoubleScalarCos(theta);
  b_a = p / (e * rNorm_tmp + 1.0);
  //  Radius
  rPF_tmp = muDoubleScalarSin(theta);
  st.site = &hb_emlrtRSI;
  c_a = 1.32712440017987E+11 / p;
  if (c_a < 0.0) {
    emlrtErrorWithMessageIdR2018a(
        &st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 4, "sqrt");
  }
  c_a = muDoubleScalarSqrt(c_a);
  //  FROM PERIFOCAL TO GEOCENTRIC
  //  Rotation of angle Omega along earth's K axis
  p = muDoubleScalarSin(Omega);
  R1_tmp = muDoubleScalarCos(Omega);
  R1[0] = R1_tmp;
  R1[3] = p;
  R1[6] = 0.0;
  R1[1] = -p;
  R1[4] = R1_tmp;
  R1[7] = 0.0;
  //  Rotation of angle i along our orbit's I axis
  p = muDoubleScalarSin(i);
  R1_tmp = muDoubleScalarCos(i);
  R1[2] = 0.0;
  R2[0] = 1.0;
  R1[5] = 0.0;
  R2[3] = 0.0;
  R1[8] = 1.0;
  R2[6] = 0.0;
  R2[1] = 0.0;
  R2[4] = R1_tmp;
  R2[7] = p;
  R2[2] = 0.0;
  R2[5] = -p;
  R2[8] = R1_tmp;
  //  Rotation of angle omega along our orbit's K axis
  p = muDoubleScalarSin(omega);
  R1_tmp = muDoubleScalarCos(omega);
  //  Rotation
  for (int32_T j{0}; j < 3; j++) {
    coffset = j * 3;
    b_i = static_cast<int32_T>(R2[j]);
    d = R2[j + 3];
    d1 = R2[j + 6];
    for (int32_T c_i{0}; c_i < 3; c_i++) {
      int32_T aoffset;
      aoffset = c_i * 3;
      y[coffset + c_i] =
          (R1[aoffset] * static_cast<real_T>(b_i) + R1[aoffset + 1] * d) +
          R1[aoffset + 2] * d1;
    }
  }
  R2[0] = R1_tmp;
  R2[1] = p;
  R2[2] = 0.0;
  R2[3] = -p;
  R2[4] = R1_tmp;
  R2[5] = 0.0;
  R2[6] = 0.0;
  R2[7] = 0.0;
  R2[8] = 1.0;
  //  From Perifocal to Geocentric Equatorial
  d_a[0] = b_a * rNorm_tmp;
  d_a[1] = b_a * rPF_tmp;
  d_a[2] = b_a * 0.0;
  for (b_i = 0; b_i < 3; b_i++) {
    d = y[b_i];
    d1 = y[b_i + 3];
    p = y[b_i + 6];
    R1_tmp = 0.0;
    for (coffset = 0; coffset < 3; coffset++) {
      b_a = (d * R2[3 * coffset] + d1 * R2[3 * coffset + 1]) +
            p * R2[3 * coffset + 2];
      R1[b_i + 3 * coffset] = b_a;
      R1_tmp += b_a * d_a[coffset];
    }
    r[b_i] = R1_tmp;
  }
  __m128d b_r;
  __m128d r1;
  d = c_a * -rPF_tmp;
  d1 = c_a * (e + rNorm_tmp);
  p = c_a * 0.0;
  b_r = _mm_loadu_pd(&R1[0]);
  b_r = _mm_mul_pd(b_r, _mm_set1_pd(d));
  r1 = _mm_loadu_pd(&R1[3]);
  r1 = _mm_mul_pd(r1, _mm_set1_pd(d1));
  b_r = _mm_add_pd(b_r, r1);
  r1 = _mm_loadu_pd(&R1[6]);
  r1 = _mm_mul_pd(r1, _mm_set1_pd(p));
  b_r = _mm_add_pd(b_r, r1);
  _mm_storeu_pd(&v[0], b_r);
  v[2] = (R1[2] * d + R1[5] * d1) + R1[8] * p;
}

// End of code generation (orbitalToCar.cpp)
