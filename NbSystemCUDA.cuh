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
    NORB_SMALLN_CLUSTER
    NORB_CONFIG_BASIC,
    NORB_CONFIG_BASIC_DISK,
    NORB_CONFIG_SHELL,
    NORB_CONFIG_EXPAND,
    NORB_CONFIG_ADV_DISK,
    NORB_CONFIG_ADV_DISK_COLLSION,
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
void printToFile(const std::string& outputFName, int step, float deltaTime, int N, float4* pos, float4* vel, float4* force);
void randomiseOrbitals(NBodyICConfig config, float4* pos, float4* vel, int N);
void initialiseForces(float4* pos, float4* force, int N);
void simulate(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2],
              uint m_currentRead, uint m_currentWrite,
              float deltaTime, int N, uint m_p, uint m_q);
void getCUDAError();
void finalise(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2]);
float normalise(float3& vector);
float3 cross(float3 v0, float3 v1);
float dot(float3 v0, float3 v1);
// Camera Translations
float zoom, xRot, yRot, zRot, xTrans, yTrans, zTrans;
float shiftSpeed;
float timestep;
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



// GPU =>>
extern "C"
{
void setDeviceSoftening(float softening);
void setDeviceBigG(float G);
void allocateNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2],  int N);
void deleteNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2]);
void copyDataToDevice(float4* device, const float4* host, int N);
void copyDataToHost(float4* host, const float4* device, int N);
void deployToGPU(float4* oldPos, float4* newPos,
                 float4* oldVel, float4* newVel,
                 float4* oldForce, float4* newForce,
                 float deltaTime, int N, uint p, uint q);

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
