//
// Created by quartzar on 18/10/22.
//

#ifndef ORBITERV6_CONSTANTS_H
#define ORBITERV6_CONSTANTS_H

// Nice setups:
// [shell] seed=86, sys-size = 700, v_s = 0.005, centre_m=2000000000 for something special
// [shell] seed=88, N=40, CM=2000000, Size=30, VScale=2
// [shell] seed=1, N=30, CM=10000, Size=20, Scale=1

// Simulation parameters
#define SNAPSHOT_INTERVAL 10000 // iterations between each snapshot
#define SEED 6 // seed for IC generator
#define N_B_MULTIPLIER 1 // for thread calculation -> 960/12 = 80 [SM's on RTX 3080ti]
#define N_BODIES (10 * N_B_MULTIPLIER) // number of bodies [*N_B_MULTIPLIER]
#define TIME_STEP 1 // time between integration steps in days // 0.003472222222222223 is 5 minutes
#define ITERATIONS 1000000 // iterations before finishing simulation
#define SOFTENING 0.000125 // ~0.011 AU // softening factor for close interactions [0.0125?]
#define ETA_ACC 0.002 // acceleration variable timestep coefficient
#define ETA_VEL 0.0002 // velocity variable timestep coefficient
#define TIME_STEP_INTERVAL 100 // iterations between each timestep change
#define MAX_DELTA_TIME 500 // maximum timestep in days
#define Q 1 // rows of ->  [[threads per body]] != REMOVED !=
#define P 20 // P <= 640 || N/80 to calculate // MAX PxQ = 1024
// Q=8 threads p/b & P = 80 ->> PxQ==640 (50*NB_M /640)==80
// At N=10*NB_M, P=128 & Q=8 give enormous performance boosts
// solar system=>1 | cluster => 64? | fogr N=50*multi, P=640 means all 80 SMP's used
// for N=200*m, P=1024 (max)

// Physical constants
#define BIG_G 2.9599e-4 // gravitational constant [AU^3 / M_solar * days^2]
#define KMS_TO_AUD (float)1731.5 // 1754.385965 // 1731.5 // km.s to AU.day
#define ALPHA_VIR 2.f // virial theorem constant
#define auTOkm (float)1.49597871e8 // unit.kilometres /= unit.AU || unit.AU *= unit.kilometres
#define SOLAR_MASS (float) 1.989e30 // kg

// IC for proper small-N simulations
#define NUM_CLUSTERS 1 // number of clusters
#define STARS_PER_CLUSTER 10 // number of stars per cluster
#define R_CLUSTER 2062.f //10e4; // AU // 0.01 pc
#define FILAMENT_OFFSET_X 6186.f
#define FILAMENT_OFFSET_Y 0.f
#define FILAMENT_OFFSET_Z 0.f

// Initial Conditions
#define INIT_POS (500) // initial radius from centre in AU
#define INIT_VEL (10/ KMS_TO_AUD) //(5 * KMS_TO_MS) // initial velocity in km/s
#define INIT_M_LOWER (0.08) // initial minimum mass in solar mass
#define INIT_M_HIGHER (1500) //initial maximum mass in solar mass
#define CENTRE_STAR_M 10 // mass of centre star in disk galaxy
#define SYSTEM_THICKNESS 100 // z thickness in AU
#define SYSTEM_SIZE 100 // furthest orbital from centre in AU
#define VEL_SCALE 1 // 1 is nice with size = 20
/* Basic config options*/
#define SYS_WIDTH 5000
#define SYS_HEIGHT 5000


// Window & render parameters
#define WIDTH 1700 // 1700 PC // 2800 LT
#define HEIGHT 910 // 910 PC // 1550 LT
#define RENDER_INTERVAL 100 // timesteps between each frame

// OpenGL parameters
#define FOV 90 // 90 normal // 2 solar system
#define V_FAR 500 // 50000 normal // 50 solar system
#define INIT_ZOOM 1000 // 5000 normal // 100 for solar system

// Camera controls
#define ZOOM_SCALE 0.01 // how fast to zoom
#define SHIFT_FACTOR 10 // how much faster all movement is with shift held
#define CTRL_FACTOR 0.1 // how much slower all movement is with ctrl held
#define MOVE_SPEED 1 // how fast to move [AU/frame] // 1 for solar system
#define ORB_SIZE 1 // 2-3 LT // 0.5 PC // range of pixels to display dot

// Proto-cluster options



// ADV_DISK options
#define ADVD_CENTRE_M 2000000 //
#define ADVD_C_INNER 0.02 //
#define ADVD_M_INNER_MIN 1.0 //
#define ADVD_M_INNER_MAX 100.0 //
#define ADVD_R_INNER 3000.0 //
#define ADVD_C_OUTER 0.1 //
#define ADVD_M_OUTER 10000.0 //
#define ADVD_R_OUTER 4000.0 //
#define ADVD_OUTER_N 500 //
#define ADVD_G2_MASS 1000000 //
// ADV_DISK collision options
#define ADVD_G2_X 50000
#define ADVD_G2_Y 1000
#define ADVD_G2_Z (1000)
#define ADVD_G2_VX (0.5)
#define ADVD_G2_VY (0.0)
#define ADVD_G2_VZ (0.01)

// MISC
#define PI 3.14159265358979323846 // it's just a pie

// UNUSED DEFINITIONS
#define INNER_BOUND 50 // closest orbital to centre in AU
#define ROT_SPEED 20 // speed to rotate camera if it's enabled
#define V_S_SYSTEM_SCALE 7000 // scaling for solar system viewing (height|width / scale)
//#define SOLAR_MASS 1.989e30  // solar mass in kg
//
//#define AU_TO_M 1.49e11 // metres in 1 AU
//#define KMS_TO_MS 1.0e3 // km/s to m/s
#define VIEW_ANGLE 40 //
#define CAMERA_SPEED 50 //


#endif //ORBITERV6_CONSTANTS_H
