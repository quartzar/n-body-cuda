//
// Created by quartzar on 23/10/22.
//

#ifndef ORBITERV6_NBSYSTEMCUDA_CUH
#define ORBITERV6_NBSYSTEMCUDA_CUH

// INCLUDES
#include "NbodyRenderer.h"

// IC options
enum NBodyICConfig
{
    NORB_SMALLN_CLUSTER,
    // NORB_CONFIG_BASIC,
    // NORB_CONFIG_BASIC_DISK,
    // NORB_CONFIG_SHELL,
    // NORB_CONFIG_EXPAND,
    // NORB_CONFIG_ADV_DISK,
    // NORB_CONFIG_ADV_DISK_COLLSION,
    NORB_CONFIG_SOLAR
};

// Integrator Options
enum NbodyIntegrator
{
    LEAPFROG_VERLET,
    KICK_DRIFT_VERLET
};


// Memory copy/retrieve options
// enum OrbitalArray
// {
//     NORB_POSITION,
//     NORB_VELOCITY,
// };



/* these are explained in the kernel */
//---------------------------------------


//---------------------------------------
// METHODS
//---------------------------------------
// CPU =>>
// void initialise(NBodyICConfig config, int N_orbitals, int iteration, float timestep,
//                 uint m_currentRead, uint m_currentWrite, uint m_p, uint m_q,
//                 float4*m_hPos[2], float4*m_hVel[2],
//                 float4*m_dPos[2], float4*m_dVel[2]);
void runTimer(std::chrono::system_clock::time_point start,
              int N_orbitals, bool init);
GLFWwindow* initGL(GLFWwindow *window);

void runSingleSimulation(const std::string& simulation_base, uint32_t mass_seed, uint32_t position_seed, uint32_t velocity_seed,
                         int N_bodies, float softening, float time_start, float time_end, float snap_rate, float delta_time,
                         bool cross_time, float eta_cross, float eta_accel, float eta_veloc);

void runMultipleSimulations(const std::string& simulation_base, int parallel_runs, uint32_t mass_seed, uint32_t position_seed,
                            uint32_t velocity_seed, int N_bodies, float softening, float time_start, float time_end,
                            float snap_rate, float delta_time, bool cross_time, float eta_cross, float eta_accel, float eta_veloc);

void readParameters(const std::string &filename, std::string &simulation_base,
                    uint32_t &mass_seed, uint32_t &position_seed, uint32_t &velocity_seed, int &N_bodies, float &softening,
                    float &time_start, float &time_end, float &snap_rate, float &initial_dt,
                    bool &cross_time, float &ETA_cross, float &ETA_acc, float &ETA_vel, int &parallel_runs);
void writeBinaryData(const std::string& filename, float current_time, float dT,
                     float softening_factor, int N, float4* pos, float4* vel, float4* force,
                     uint mass_seed, uint position_seed, uint velocity_seed);
// void writeBinaryData(const std::string& filename, int snapshot_interval, int iteration, int total_iterations, float deltaTime,
//                      float softening_factor, int N, float4* pos, float4* vel, float4* force);
float calculateCrossingTime(const float4 *vel, int N);
std::string getCurrentTime();
void randomiseOrbitals(NBodyICConfig config, float4* pos, float4* vel, int N,
                       uint32_t &mass_seed, uint32_t &position_seed, uint32_t &velocity_seed);
float4 calculateCentreOfMass(float4* body, int N);
float calculateGravitationalEnergy(float4* pos, int N);
float calculateKineticEnergy(float4* vel, int N);
void initialiseForces(float4* pos, float4* force, int N);
float calculateTimeStep(float4 *pos, float4 *vel, float4 *force, float curDT, int N, float eta_v, float eta_a);
void simulate(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2],
              uint m_currentRead, uint m_currentWrite,
              float& m_hDeltaTime, float* m_dDeltaTime[2], int N, uint m_p, uint m_q);
void getCUDAError();
void finalise(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2],
              float* m_dDeltaTime[2]);
void deleteFilesInDirectory(const std::string& directory_path);
float normalise(float3& vector);
float3 cross(float3 v0, float3 v1);
float dot(float3 v0, float3 v1);
// Camera Translations
float zoom, xRot, yRot, zRot, xTrans, yTrans, zTrans;
float shiftSpeed;
// float m_hDeltaTime;
// float timestep;
// ->mouse
// bool firstMouse;
// float lastX;
// float lastY;
// float yaw;
// float pitch;
// Callbacks for OpenGL =>
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// Distribution functions ==>



// GPU =>>
extern "C"
{
void setDeviceSoftening(float softening);
void setDeviceBigG(float G);
void setDeviceEtaAcc(float eta);
void setDeviceEtaVel(float eta);
void allocateNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2], float* dT[2],  int N);
void deleteNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2], float* dT[2]);
void copyDataToDevice(float4* device, const float4* host, int N);
void copyDataToHost(float4* host, const float4* device, int N);
void deployToGPU(float4* oldPos, float4* newPos,
                 float4* oldVel, float4* newVel,
                 float4* oldForce, float4* newForce,
                 float* oldDT, float* newDT, int N, uint p, uint q);

}
//---------------------------------------

// DATA

// // booleans =>
// bool displayEnabled = false;
// bool outputEnabled = false;
// //-------------------------
//
// // CPU data =>
// float *m_hPos, *m_hVel;
// //-------------------------
// // memory transfers =>
// uint m_currentRead, m_currentWrite;
// //-------------------------
// // GPU data =>
// float *m_dPos[2], *m_dVel[2];
// //-------------------------
//
// // OpenGL =>
// GLFWwindow *window = nullptr; // TODO: protect these!
// //-------------------------
//
// // Green things =>
// NbodyRenderer *renderer = nullptr;
// //-------------------------
//
// // Timers & benchmarking =>
// std::chrono::system_clock::time_point start;
// std::chrono::system_clock::time_point end;
// //-------------------------
//
// // File output =>
// std::string outputFName = "outputCSV.csv";
// //-------------------------
//
// // Simulation =>
// int iteration;
// float timestep;
// int N_orbitals;
//-------------------------

class NbSystemCUDA {


};



#endif //ORBITERV6_NBSYSTEMCUDA_CUH
