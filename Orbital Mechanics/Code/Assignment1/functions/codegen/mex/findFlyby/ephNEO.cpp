//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// ephNEO.cpp
//
// Code generation for function 'ephNEO'
//

// Include files
#include "ephNEO.h"
#include "date2jd.h"
#include "findFlyby_data.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo mc_emlrtRSI{
    46,         // lineNo
    "date2mjd", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\time\\date2mjd.m" // pathName
};

static emlrtRSInfo nc_emlrtRSI{
    582,      // lineNo
    "ephNEO", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\ephNEO.m" // pathName
};

static emlrtRSInfo oc_emlrtRSI{
    599,      // lineNo
    "ephNEO", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\ephNEO.m" // pathName
};

static emlrtRSInfo pc_emlrtRSI{
    636,      // lineNo
    "ephNEO", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\ephNEO.m" // pathName
};

static emlrtRSInfo qc_emlrtRSI{
    637,      // lineNo
    "ephNEO", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\ephNEO.m" // pathName
};

static emlrtRSInfo rc_emlrtRSI{
    672,      // lineNo
    "ephNEO", // fcnName
    "C:"
    "\\Users\\tomis\\OneDrive\\Documentos\\GitHub\\omProject\\Assignment1\\func"
    "tions\\weBeep\\ephNEO.m" // pathName
};

// Function Definitions
real_T ephNEO(const emlrtStack &sp, real_T b_time, real_T kep[6])
{
  static const real_T b_date[6]{2029.0, 12.0, 3.0,
                                18.0,   30.0, 40.3551757335663};
  static const real_T date[6]{2012.0, 5.0, 19.0, 16.0, 46.0, 32.9053440000004};
  emlrtStack b_st;
  emlrtStack st;
  real_T M;
  real_T phi;
  real_T t;
  st.prev = &sp;
  st.tls = sp.tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  //  ephNEO.m - Ephemerides of Near Earth Objects
  //
  //  PROTOTYPE:
  // 	[kep, mass, M] = ephNEO(time, id)
  //
  //  DESCRIPTION:
  //    This function returns the orbital parameters, the mass, and the mean
  //    anomaly of some NEOs. Each NEO is identified by an id.
  //
  //  INPUT:
  //    time[1]     MJD2000 [d].
  //    id[1]       NEO identifier. It is a number starting from 12 (because
  //                the identifiers from 1 to 11 are reserved for the Solar
  //                System).
  //
  //  OUTPUT:
  //    kep[1,6]    Keplerian parameters. It is a 6 entry vector:
  //                [a e i Om om wom] where:
  //                    a is the semimajor axis [km];
  //                    e is the eccentricity;
  //                    i is the inclination [rad];
  //                    Om is the anomaly of the ascending node [rad];
  //                    om is the anomaly of the pericentre [rad];
  //                    wom is the true anomaly (from the pericentre) [rad].
  //    mass        Mass of the NEO [kg]. It can be read from the database, or,
  //                if not available, estimated by an approximate equation.
  //    M           Mean anomaly at time [rad].
  //
  //  CALLED FUNCTIONS:
  //    astroConstants
  //
  //  AUTHOR:
  //    Paolo De Pascale,  November 2004, MATLAB, ephNEO.m
  //
  //  PREVIOUS VERSION:
  //    Paolo De Pascale,  November 2004, MATLAB, NeoEphemeris.m
  //        - Header and function name in accordance with guidlines.
  //
  //  CHANGELOG:
  //    20/06/2006, Christie Maddock, Camilla Colombo, Pau Sanchez
  //    20/06/2006, Matteo Ceriotti: Ccleaned up code, added references to
  //        astro_constants.
  //  	07/07/2006, Pau Sanchez: Added data on more NEOs.
  //    25/08/2006, Pau Sanchez: Added data on more NEOs.
  //  	12/02/2007, Pau Sanchez: Modified mass Itokawa from 6.01e10 kg to
  //        3.51e10 kg - updated from papers.
  //  	16/05/2007, Camilla Colombo: added new PHO form Paolo and Max database
  //                       Note for Paolo and Max: id = 57 to 72 of your
  //                       code is here id 68 to 83.
  //  	07/06/2008, Camilla Colombo: added 2002AA29.
  //    04/10/2010, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //    12/11/2012, Camilla Colombo: Added body 85 for SysNova.
  //    13/03/2013, Michele Stramacchia: Added body 88 etc.
  //
  //  -------------------------------------------------------------------------
  //  to UPDATE use this layout:
  //  %   name, class
  //  neo(id,1)=semimajor axis [AU]     neo(id,2)=eccentricity
  //  neo(id,3)=inclination [deg]     neo(id,4)=asc. node/raan [deg]
  //  neo(id,5)=arg. perigee [deg]     neo(id,6)=mean anomoly, M at time given
  //  in neo(id,8) [deg] neo(id,7)=abs magnitude (i.e. intrinsic brightness)
  //  neo(id,8)=time at which Mo is given [MJD]   neo(id,9)=mass [kg]
  //    Apollo Apollo
  //    Nereus Apollo
  //    Toutatis Apollo
  //    Adonis Apollo
  //    Hermes Apollo
  //    Oljato Apollo
  //    Icarus Apollo
  //    Geographos
  //    Orpheus Apollo
  //    Dionysus  Apollo
  //    Nyx Amor
  //    Minos Apollo
  //    EY2 Apollo
  //    YE4 Aten
  //    SG286 Apollo
  //    YB5 Apollo
  //    VD35 Apollo
  //    VF32 Aten
  //    WR12 Athen
  //    SG344 Aten
  //    UG Apollo class
  //    GK Apollo
  //    SB45  Apollo class
  //    Florence between apollo and amor
  //    Castalia  Apollo class
  //    Pan Apollo class
  //    1989UQ Aten Class
  //    2001CC21 Apollo
  //    1996FG3 Apollo Class
  //    1999YB Amor Class
  //    1994CN2 Apollo class
  //    2003gg21 Apollo class
  //    1989ML Amor class
  //    1999JU3 Apollo class
  // 315.075562;
  //    1999AO10 Aten class
  //    2000LG6 Aten classs
  //    1999AO10
  //  neo(36,1)=0.91220084; neo(36,2)=0.1102457;
  //  neo(36,3)=2.6257975; neo(36,4)=72.7702;
  //  neo(36,5)=7.4322027; neo(36,6)=283.0520;
  //  neo(36,7)=16.4;
  //  neo(36,8)=53400.5;
  //    2003WP25  Aten class
  //   2002 at4  Amor class
  //   2002MN4 Athen Class
  //   2002VD17 Apollo Class
  //   1997XR2  APOLLO class
  //   2001 TW229
  // --------------------------------------------------------------------------
  // Potentially Hazardous Objects PHO
  // da qui ti aggiungo i nuovi che dobbiamo studiare, ci sono alcune
  // ripetizioni con asteroidi già presenti sopra , ma almeno li hai tutti in
  // fila il che può essere piu' comodo . Ti aggiungo anche la magnitudine
  // degli asteroidi.
  //   2004VD17
  //   Apophis
  //   2005WY5
  // ------------------------------------------------------------------------
  //  da qui iniziano quelli lost o non recentemente osservati
  //  direi che sono interessanti sia per il discorso impatto, che peril
  //  discorso rendezvous...quelli sopra sono stati osservati recentemente, il
  //  che nono vuol dire che una missione di rendezvous non sia comunque
  //  importante visto che comunque c' è incertezza nella conoscenza
  //  orbitale...
  //   1997XR2
  //   1994WR12
  //   1979XB
  //   2000SG344
  //   2000QS7
  //   1998HJ3
  //   2005TU45
  //   2004XK3
  //   1994GK
  //   2000SB45
  //   2001CA21
  //  Aggiunto da Camilla -----------------------------------------------------
  //   1991RB - usato da Conway
  //  from NeoDys
  //  neo(57,1)=1.45448;      neo(57,2)=0.485596;
  //  neo(57,3)=19.592;       neo(57,4)=359.48;
  //  neo(57,5)=68.755;       neo(57,6)=59.111;
  //  neo(57,8)=53800;        neo(57,9)=0;
  //  from http://cfa-www.harvard.edu/iau/lists/Dangerous.html
  //  io usavo questo con l'epoca sbagliata
  //  from Conway paper
  //  neo(57,1)=1.45;        neo(57,2)=0.484;
  //  neo(57,3)=19.5;        neo(57,4)=359.6;
  //  neo(57,5)=68.7;        neo(57,6)=320.1;
  //  neo(57,8)= 48500.0 1/09/1991
  //  neo(57,9)=0;
  //   1996JA1 -usato da Carusi
  //  from NeoDys
  //  433 Eros
  //  Itokawa
  //  earlier neo(60,9) was 6.01e10 -> updated from papers
  // Castalia
  //  1999KW4
  //  2062 Aten
  //  3908 Nyx
  //  1036 Ganymed
  //  1620 Geographos
  //  Quetzalcoatl
  // --------------------------------------------------------------------------
  //  - new PHO -
  //  2005QK76
  //  2002TX55
  //  2005EL70
  //  2001BB16
  //  2002VU17
  //  2000TU28
  //  2001Av43
  //  2003RB182
  //  2002GJ8
  //  2001FB90
  //  2005NX55
  //  1996TC1
  //  6344P-L
  //  2004ME6
  //  2001QJ96
  //  2004GE2
  //  2002AA29
  //  from http://neo.jpl.nasa.gov/cgi-bin/neo_elem 07/06/2008
  //  2008JL24
  //  from SysNova
  st.site = &nc_emlrtRSI;
  //  date2mjd.m - modified Julian day number from Gregorian calendar date.
  //
  //  PROTOTYPE:
  //    mjd = date2mjd(date)
  //
  //  DESCRIPTION:
  //    Returns the modified Julian day number corresponding to the Gregorian
  //    calendar date (year, month, day, hour, minute, and second).
  //    Note: The function is valid for the whole range of dates since 12:00
  //        noon 24 November -4713, Gregorian calendar. (This bound is set in
  //        order to have symmetry with the inverse function jd2date)
  //    Note: The inputs must be feasible (i.e. the date must exist!). If an
  //        unfeasible date is inputed, wrong results are given because no
  //        check is done on that.
  //
  //  INPUT:
  //    date[6]     Date in the Gregorian calendar, as a 6-element vector
  //                [year, month, day, hour, minute, second]. For dates before
  //                1582, the resulting date components are valid only in the
  //                Gregorian proleptic calendar. This is based on the
  //                Gregorian calendar but extended to cover dates before its
  //                introduction. date must be after 12:00 noon, 24 November
  //                -4713.
  //
  //  OUTPUT:
  //    mjd[1]      Date in modified Julian Day. The MJD count is from 00:00
  //                midnight at the beginning of Wednesday November 17, 1858.
  //
  //  See also mjd2date.m.
  //
  //  CALLED FUNCTIONS:
  //     date2jd
  //
  //  AUTHOR:
  //    Nicolas Croisard, 16/02/2008, MATLAB, date2mjd
  //
  //  CHANGELOG:
  //    03/03/2008, REVISION, Camilla Colombo
  //    22/04/2010, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //
  //  -------------------------------------------------------------------------
  b_st.site = &mc_emlrtRSI;
  date2jd(b_st, date);
  //  % 2006RH120
  //  % from SysNova before encounter
  //  neo(86,1) = 1.033252056035198;         neo(86,2) = 0.02447403062284801;
  //  neo(86,3) = 0.5952660003048117;        neo(86,4) = 51.14334927580387;
  //  neo(86,5) = 10.14353817485877;         neo(86,6) = 221.2498016727181;
  //  neo(86,8) = date2mjd([2013, 02, 24, 20, 32,  38.4881]);
  //  neo(86,7) = 29.527;
  //  2006RH120
  //  from SysNova  after encounter
  st.site = &oc_emlrtRSI;
  //  date2mjd.m - modified Julian day number from Gregorian calendar date.
  //
  //  PROTOTYPE:
  //    mjd = date2mjd(date)
  //
  //  DESCRIPTION:
  //    Returns the modified Julian day number corresponding to the Gregorian
  //    calendar date (year, month, day, hour, minute, and second).
  //    Note: The function is valid for the whole range of dates since 12:00
  //        noon 24 November -4713, Gregorian calendar. (This bound is set in
  //        order to have symmetry with the inverse function jd2date)
  //    Note: The inputs must be feasible (i.e. the date must exist!). If an
  //        unfeasible date is inputed, wrong results are given because no
  //        check is done on that.
  //
  //  INPUT:
  //    date[6]     Date in the Gregorian calendar, as a 6-element vector
  //                [year, month, day, hour, minute, second]. For dates before
  //                1582, the resulting date components are valid only in the
  //                Gregorian proleptic calendar. This is based on the
  //                Gregorian calendar but extended to cover dates before its
  //                introduction. date must be after 12:00 noon, 24 November
  //                -4713.
  //
  //  OUTPUT:
  //    mjd[1]      Date in modified Julian Day. The MJD count is from 00:00
  //                midnight at the beginning of Wednesday November 17, 1858.
  //
  //  See also mjd2date.m.
  //
  //  CALLED FUNCTIONS:
  //     date2jd
  //
  //  AUTHOR:
  //    Nicolas Croisard, 16/02/2008, MATLAB, date2mjd
  //
  //  CHANGELOG:
  //    03/03/2008, REVISION, Camilla Colombo
  //    22/04/2010, Camilla Colombo: Header and function name in accordance
  //        with guidlines.
  //
  //  -------------------------------------------------------------------------
  b_st.site = &mc_emlrtRSI;
  date2jd(b_st, b_date);
  //  % Max
  //  % from SysNova
  //  neo(87,1) = 1.0;         neo(87,2) = 0.00001;
  //  neo(87,3) = 5;        neo(87,4) = 0;
  //  neo(87,5) = 0;         neo(87,6) = 10;
  //  neo(87,8) = date2mjd([2013, 02, 24, 20, 32,  38.4881]);
  //  neo(87,7) = 29.527;
  //   Apophis
  //  neo(88,1)=ainAU;        neo(88,2)=e;
  //  neo(88,3)=i in deg ;        neo(88,4)=Om in deg;
  //  neo(88,5)=om in deg ;         neo(88,6)=M in deg;
  //  neo(88,8)=date2mjd([2029, 12, 3, 18, 30,  40.3551757335663]); ref time in
  //  data 2010RF12
  // =====================================================
  st.site = &pc_emlrtRSI;
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
  //  From DE405
  //  km
  st.site = &qc_emlrtRSI;
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
  //  km^3/s^2 Sun gravitational constant
  t = (b_time - 856.0) * 86400.0 + 3.2677393086146684E+6;
  t -= 2.4818273229984824E+7 * muDoubleScalarFloor(t / 2.4818273229984824E+7);
  phi = 2.5316770626847628E-7 * t;
  M = phi;
  if (phi > 3.1415926535897931) {
    M = phi - 6.2831853071795862;
  }
  //  Ciclo di Newton: devo mandare a zero la funzione
  // f=(M-E+esinE)...parto da E=M...deltaE=-fdot^-1*f
  for (int32_T i{0}; i < 5; i++) {
    real_T ddf;
    ddf = 0.446 * muDoubleScalarCos(phi) - 1.0;
    phi = (phi - t * 2.5316770626847628E-7 / ddf) +
          (phi - 0.446 * muDoubleScalarSin(phi)) / ddf;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtConstCTX)&sp);
    }
  }
  st.site = &rc_emlrtRSI;
  //  In radians
  kep[0] = 1.2745738582873201E+8;
  kep[1] = 0.446;
  kep[2] = 0.41887902047863906;
  kep[3] = 4.125958351714595;
  kep[4] = 5.599016240397809;
  kep[5] = 2.0 * muDoubleScalarAtan(1.6155829608067478 *
                                    muDoubleScalarTan(phi * 0.5));
  //    polynomial fit to the H-d diagram which gives the diameter of the NEO as
  //    a function of its magnitude. polynomial of fifth order, Note: NOT VERY
  //    ACCURATE! estimated mass of the NEO computed considering a density of 2
  //    kg/dm^3
  return M;
}

// End of code generation (ephNEO.cpp)
