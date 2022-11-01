//
// Created by quartzar on 18/10/22.
//

#ifndef ORBITERV6_NBODYN2_CUH
#define ORBITERV6_NBODYN2_CUH

#include "NbodyRenderer.h"

// PARAMETERS
NbodyRenderer::RenderMode renderMode = NbodyRenderer::POINTS;

// VARIABLES
int xMonitor;
int yMonitor;

// Camera Translations
float zoom, xRot, yRot, zRot, xTrans, yTrans, zTrans;
float shiftSpeed;
// ->mouse
bool firstMouse;
float lastX;
float lastY;
float yaw;
float pitch;

// CPU functions
void initialiseOrbitals(float4* orbPos, float3* orbVel);
void simulateOrbitals(float4* orbPos, float3* orbVel, GLFWwindow* window, const std::string& outputFName);
void writeCSV(float4* orbPos, float3* orbVel, int step, const std::string& outputFName);

// GPU functions
__global__ void clearForce(float3* orbForce);
__global__ void orbitalOrbitalInteraction(float4* orbPos, float3* orbVel, float3* orbForce);
__global__ void updateOrbitals(float4 *orbPos, float3 *orbVel, float3 *orbForce, int fKick);

// OpenGL functions
GLFWwindow *initGL(GLFWwindow *window);
void initCamera();
void processInput(GLFWwindow *window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void window_size_callback(GLFWwindow* window, int width, int height);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void cursor_position_callback(GLFWwindow* window, double xposIn, double yposIn);

// structs

float
normalise(float3& vector);

float
dot(float3 v0, float3 v1);

float3
cross(float3 v0, float3 v1);

#endif //ORBITERV6_NBODYN2_CUH
