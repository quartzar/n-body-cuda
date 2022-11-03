//
// Created by quartzar on 23/10/22.
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>

#include <GL/glew.h>
// #include <GL/glut.h>
#include <GLFW/glfw3.h>

#include "NbKernel_N2.cuh"
#include "CONSTANTS.h"
#include "NbSystemCUDA.cuh"

/////////////////////////////////////////
// █▀█░█▀█░█▄▄░█░▀█▀░█▀▀░█▀█░░░█░█░▀▀█ //
// █▄█░█▀▄░█▄█░█░░█░░██▄░█▀▄░░░▀▄▀░░░█ //
/////////////////////////////////////////

extern __constant__ float softeningSqr;
extern __constant__ float big_G;

//------------PARAMETERS---------------//
NbodyRenderer::RenderMode renderMode = NbodyRenderer::POINTS;
NBodyICConfig sysConfig = NORB_CONFIG_SOLAR;
NbodyRenderer *renderer = nullptr;
// booleans =>
bool displayEnabled = true;
bool glxyCollision = true;
bool colourMode = false;
bool trailMode = true;
bool outputEnabled = false;
bool rotateCam = false;
//---------------------------------------q

/////////////////////////////////////////

//---------------------------------------
int main(int argc, char** argv)
{
    //-------------------------
    // CPU data =>
    float4 *m_hPos, *m_hVel, *m_hForce;
    //-------------------------
    // memory transfers =>
    uint m_currentRead, m_currentWrite;
    //-------------------------
    // GPU data =>
    float4 *m_dPos[2], *m_dVel[2], *m_dForce[2];
    //-------------------------
    // OpenGL =>
    GLFWwindow *window = nullptr;
    //-------------------------
    // Timers & benchmarking =>
    auto start = std::chrono::system_clock::now();
    // std::chrono::system_clock::time_point end;
    //-------------------------
    // File output =>
    std::string outputFName = "outputCSV.csv";
    
    //-------------------------
    // Simulation =>
    int iteration;
    // float timestep;
    int N_orbitals;
    uint m_p;
    uint m_q;
    // if (sysConfig == NORB_CONFIG_SOLAR)
    //     N_orbitals = 3; // TODO: constructor for solar system
    N_orbitals = N_BODIES;
    
    iteration = 0;
    timestep = TIME_STEP;
    m_currentRead = 0;
    m_currentWrite = 1;
    
    m_p = P;
    m_q = Q;
    zoom = 1;
    //---------------------------------------
    // INITIALISE ARRAYS & ALLOCATE DEVICE STORAGE
    //---------------------------------------
    
    // OLD / HOST
    m_hPos = new float4[N_orbitals]; // x, y, z, mass
    m_hVel = new float4[N_orbitals]; // vx,vy,vz, empty
    m_hForce = new float4[N_orbitals]; // fx, fy, fz, empty
    // NEW / DEVICE
    m_dPos[0] = m_dPos[1] = nullptr;
    m_dVel[0] = m_dVel[1] = nullptr;
    m_dForce[0] = m_dForce[1] = nullptr;
    // set memory for host arrays
    memset(m_hPos, 0, N_orbitals*sizeof(float4));
    memset(m_hVel, 0, N_orbitals*sizeof(float4));
    memset(m_hForce, 0, N_orbitals*sizeof(float4));
    getCUDAError();
    // set memory for device arrays
    allocateNOrbitalArrays(m_dPos,m_dVel, m_dForce, N_orbitals);
    getCUDAError();
    // set device constants
    setDeviceSoftening(SOFTENING);
    setDeviceBigG(1.0f * BIG_G);
    getCUDAError();
    
    //---------------------------------------
    /////////////////////////////////////////
    //---------------------------------------
    
    
    // BEGIN TIMER
    runTimer(start, N_orbitals, true);
    
    // INITIALISE OPENGL
    if (displayEnabled)
    {
        // glutInit(&argc, argv);
        // glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
        window = initGL(window);
    }
    
    // PRINT TO FILE
    if (outputEnabled)
    {
        std::ofstream outputFile(outputFName);
        for (int orbital = 0; orbital < N_orbitals; orbital++)
        {
            outputFile << orbital << ","
                       << orbital << "," << orbital << "," << orbital << ","
                       << orbital << "," << orbital << "," << orbital << ","
                       << orbital << "," << orbital << "," << orbital;
            if (orbital != N_orbitals - 1) outputFile << ",";
        }
        outputFile << std::endl;
        for (int orbital = 0; orbital < N_orbitals; orbital++)
        {
            outputFile << "M" << ","
                       <<"x"  << "," << "y"  << "," << "z"  << ","
                       << "vx" << "," << "vy" << "," << "vz" << ","
                       << "fx" << "," << "fy" << "," << "fz";
            if (orbital != N_orbitals - 1) outputFile << ",";
        }
        outputFile << std::endl;
        outputFile.close();
    }
    
    // Randomise Orbitals
    randomiseOrbitals(sysConfig, m_hPos, m_hVel, N_orbitals);
    // Set Initial Forces
    initialiseForces(m_hPos, m_hForce, N_orbitals);
    
    //---------------------------------------
    // MAIN UPDATE LOOP
    while (iteration <= ITERATIONS)
    {
        if (iteration % 100 == 0)
            std::cout << "\nSTEP =>> " << iteration << std::flush;
    
        if (outputEnabled)
            printToFile(outputFName, iteration, N_orbitals, m_hPos, m_hVel, m_hForce);
        
        simulate(m_hPos, m_dPos,
                 m_hVel, m_dVel,
                 m_hForce, m_dForce,
                 m_currentRead, m_currentWrite,
                 timestep, N_orbitals, m_p, m_q);
        
        if (displayEnabled && iteration%RENDER_INTERVAL == 0)
        {
            // CHECK FOR INPUT FIRST
            processInput(window);
    
            // CLOSE WINDOW IF ESC PRESSED
            if (glfwWindowShouldClose(window))
            {
                std::cout << "\nPROGRAM TERMINATED BY USER\nEXITING AT STEP " << iteration;
                runTimer(start,  N_orbitals,false);
                finalise(m_hPos, m_dPos,
                         m_hVel, m_dVel,
                         m_hForce, m_dForce);
                glfwTerminate();
                exit(EXIT_SUCCESS);
            }
            
            // render
            renderer->setPositions(reinterpret_cast<float *>(m_hPos));
            renderer->setVelocities(reinterpret_cast<float *>(m_hVel));
            renderer->display(renderMode, zoom, xRot, yRot, zRot, xTrans, yTrans, zTrans, trailMode, colourMode);
    
            glfwSwapBuffers(window);
            // glutSwapBuffers();
            glfwPollEvents();
    
            // set window title to current timestep
            std::string s = std::to_string(iteration);
            const char* cstr = s.c_str();
            glfwSetWindowTitle(window, cstr);
        }
        
        iteration++;
    }
    //---------------------------------------
    
    
    // END TIMER
    runTimer(start,  N_orbitals,false);
    
    // DELETE ARRAYS
    finalise(m_hPos, m_dPos,
             m_hVel, m_dVel,
             m_hForce, m_dForce);
    
    // TERMINATE SUCCESSFULLY
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
//---------------------------------------


// Print to file
//---------------------------------------
void printToFile(const std::string& outputFName, int step, int N, float4* pos, float4* vel, float4* force)
{
    std::ofstream outputFile;
    outputFile.open(outputFName, std::ios::app); // open file
    
    float mass, xPos, yPos, zPos, xVel, yVel, zVel, xFrc, yFrc, zFrc;
    for (int orbital = 0; orbital < N; orbital++)
    {
        mass = pos[orbital].w;
        xPos = pos[orbital].x;
        yPos = pos[orbital].y;
        zPos = pos[orbital].z;
        xVel = vel[orbital].x * (float)KMS_TO_AUD;
        yVel = vel[orbital].y * (float)KMS_TO_AUD;
        zVel = vel[orbital].z * (float)KMS_TO_AUD;
        xFrc = force[orbital].x;
        yFrc = force[orbital].y;
        zFrc = force[orbital].z;
    
        outputFile << mass << ","
                   << xPos << "," << yPos << "," << zPos << ","
                   << xVel << "," << yVel << "," << zVel << ","
                   << xFrc << "," << yFrc << "," << zFrc;
        if (orbital != N - 1) outputFile << ",";
    }
    outputFile << std::endl;
    // outputFile.close();

    // outputFile << step << "," << xPos << "," << yPos << "," << zPos << ","
    //            << xVel << "," << yVel << "," << zVel << std::endl;
}
//---------------------------------------


// IC generator
//---------------------------------------
void randomiseOrbitals(NBodyICConfig config, float4* m_hPos, float4* m_hVel, int N)
{
    using std::uniform_real_distribution;
    std::default_random_engine gen(SEED); // NOLINT(cert-msc51-cpp)
    float totalMass = 0.0;
    
    switch(config) {
        case NORB_CONFIG_BASIC:
        {
            uniform_real_distribution<float> randXPos(-SYS_WIDTH / 2.0, SYS_WIDTH / 2.0);
            uniform_real_distribution<float> randYPos(-SYS_HEIGHT / 2.0, SYS_HEIGHT / 2.0);
            uniform_real_distribution<float> randVel(-INIT_VEL, INIT_VEL);
            uniform_real_distribution<float> randHeight(-SYSTEM_THICKNESS, SYSTEM_THICKNESS);
            uniform_real_distribution<float> randMass(INIT_M_LOWER, INIT_M_HIGHER);
            // returns -1 to 3, so multiply by max mass/3 and clamp between min and max mass
            std::normal_distribution<float> normalDistMass(1, 0.5);
            
            
            // ASSIGNMENT LOOP
            for (int i = 0; i < N_BODIES; i++)
            {
                // getting and clamping normal distribution of mass
                const float mass = normalDistMass(gen) * ((float)INIT_M_HIGHER / 3.f);
                float massClamped;
                if (mass > 1.f * (float)INIT_M_HIGHER)
                {
                    std::cout << "\nbig boi";
                    massClamped = 100000.f;
                }
                else
                    massClamped = std::clamp(mass, (float)INIT_M_LOWER, (float)INIT_M_HIGHER);
                
                // random position assignment
                m_hPos[i].x = randXPos(gen);
                m_hPos[i].y = randYPos(gen);
                m_hPos[i].z = randHeight(gen);
                m_hPos[i].w = massClamped;
    
                // random velocity assignment
                float r = sqrtf(m_hPos[i].x * m_hPos[i].x + m_hPos[i].y * m_hPos[i].y + m_hPos[i].z * m_hPos[i].z);
                m_hVel[i].x = randVel(gen) * (r/m_hPos[i].x);//0.001f;
                m_hVel[i].y = randVel(gen) * (r/m_hPos[i].y);//0.001f;
                m_hVel[i].z = 0.0f;
                m_hVel[i].w = m_hPos[i].w;
        
                totalMass += m_hPos[i].w;
            }
        }
            break;
        
        case NORB_CONFIG_BASIC_DISK:
        {
            std::cout << "basic disk model to be implemented";
            // also to be implemented
        }
            break;
        
        case NORB_CONFIG_SHELL:
        {
            uniform_real_distribution<float> randF(0.0f, (float) RAND_MAX);
            uniform_real_distribution<float> randMass(INIT_M_LOWER, INIT_M_HIGHER);
    
            float scale = SYSTEM_SIZE;
            float vScale = scale * (float) VEL_SCALE / (float) KMS_TO_AUD;
            float inner = 2.5f * scale;
            float outer = 4.0f * scale;
    
            m_hPos[0].x = 0.0;
            m_hPos[0].y = 0.0;
            m_hPos[0].z = 0.0;
            m_hPos[0].w = CENTRE_STAR_M;
    
            m_hVel[0].x = 0.0;
            m_hVel[0].y = 0.0;
            m_hVel[0].z = 0.0;
            m_hVel[0].w = CENTRE_STAR_M;
    
    
            int i = 1;
            while (i < N_BODIES) {
                float x, y, z;
                x = randF(gen) / (float) RAND_MAX * 2.0f - 1.0f;
                y = randF(gen) / (float) RAND_MAX * 2.0f - 1.0f;
                z = randF(gen) / (float) RAND_MAX * 2.0f - 1.0f;
        
                float3 point = {x, y, z};
                float len = normalise(point);
                if (len > 1)
                    continue;
        
                m_hPos[i].x = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
                m_hPos[i].y = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
                m_hPos[i].z = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
                m_hPos[i].w = randMass(gen);
                
        
                x = 0.0f;
                y = 0.0f;
                z = 1.0f;
        
                float3 axis = {x, y, z};
                normalise(axis);
        
                if (1 - dot(point, axis) < 1e-6) {
                    axis.x = point.y;
                    axis.y = point.x;
                    normalise(axis);
                }
                float3 vv = {m_hPos[i].x, m_hPos[i].y, m_hPos[i].z};
                vv = cross(vv, axis);
                m_hVel[i].x = vv.x * vScale;
                m_hVel[i].y = vv.y * vScale;
                m_hVel[i].z = vv.z * vScale;
                m_hVel[i].w = m_hPos[i].w;
        
                i++;
            }
        }
            break;
        case NORB_CONFIG_EXPAND:
        {
            uniform_real_distribution<float> randF(0.0f, (float) RAND_MAX);
            uniform_real_distribution<float> randMass(INIT_M_LOWER, INIT_M_HIGHER);
    
            float scale = SYSTEM_SIZE * std::max(1.0f, (float)N / (1024.f));
            float vScale = scale * (float) VEL_SCALE / (float) KMS_TO_AUD;
            
            for (int i = 0; i < N;)
            {
                float3 point;
                point.x = randF(gen) / (float) RAND_MAX * 2.0f - 1.0f;
                point.y = randF(gen) / (float) RAND_MAX * 2.0f - 1.0f;
                point.z = randF(gen) / (float) RAND_MAX * 2.0f - 1.0f;
                
                float lengthSq = dot(point, point);
                if (lengthSq > 1)
                    continue;
                
                m_hPos[i].x = point.x * scale;
                m_hPos[i].y = point.y * scale;
                m_hPos[i].z = point.z * scale;
                m_hPos[i].w = randMass(gen);
                m_hVel[i].x = point.x * vScale; //* float(PI)/180 * lengthSq;
                m_hVel[i].y = point.y * vScale;
                m_hVel[i].z = point.z * vScale;
                m_hVel[i].w = m_hPos[i].w;
                
                i++;
            }
            
        }
            break;
        case NORB_CONFIG_ADV_DISK:
        {
            // uniform_real_distribution<float> randF(0.0f, (float) RAND_MAX);
            // uniform_real_distribution<float> randMass(0.0, 5);
            uniform_real_distribution<float> randMassInner(ADVD_M_INNER_MIN, ADVD_M_INNER_MAX);
            // uniform_real_distribution<float> randMassOuter(INIT_M_LOWER, INIT_M_HIGHER);
            
            m_hPos[0].x = 0.0;
            m_hPos[0].y = 0.0;
            m_hPos[0].z = 0.0;
            m_hPos[0].w = ADVD_CENTRE_M;
    
            m_hVel[0].x = 0.0;
            m_hVel[0].y = 0.0;
            m_hVel[0].z = 0.0;
            m_hVel[0].w = ADVD_CENTRE_M;
    
            float c      = ADVD_C_INNER; // flatness
            float mass   = randMassInner(gen);
            // float mass = randMassInner(gen);
            float radius = ADVD_R_INNER;
            
            int start;
            if (glxyCollision)
            {
                m_hPos[1].x = 1000.0;
                m_hPos[1].y = 500.0;
                m_hPos[1].z = -10000.0;
                m_hPos[1].w = ADVD_G2_MASS;
    
                m_hVel[1].x = -0.1;
                m_hVel[1].y = 0.0;
                m_hVel[1].z = 1.0;
                m_hVel[1].w = ADVD_G2_MASS;
                start = 2;
            }
            else
                start = 1;
            for (int i = start;i < N; i++)
            {
                if (i == N - ADVD_OUTER_N) {
                    c = ADVD_C_OUTER;
                    mass = ADVD_M_OUTER;
                    // mass = randMassOuter(gen) * 100.0f;
                    radius = ADVD_R_OUTER;
                }
                
                float3 pos;
                while (true)
                {
                    pos.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    pos.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    
                    if (pos.y >= -1.0f * sqrtf(1.0f - powf(pos.x, 2.0f))
                    && pos.y <= sqrtf(1.0f - powf(pos.x, 2.0f)))
                        break;
                }
                
                float zPosMax = sqrtf(c * (1.0f - powf(pos.x, 2.0f)
                        -powf(pos.y, 2.0f)));
                float zPosMin = -1.0f * zPosMax;
                float zPosRand = rand() / (float) RAND_MAX;
                pos.z = (zPosMax - zPosMin) * zPosRand + zPosMin;

                pos.x *= radius;
                pos.y *= radius;
                pos.z *= radius;
                
                float m = pos.y / pos.x;
                m = -1.0f / m;
                float b = pos.y - pos.x * m;
                
                float3 vel;
                // float vel_m = sqrtf(((float)BIG_G * (1e6f + mass * 1.2e-6f)) /
                //                     sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z));
                float vel_m = sqrtf(((float)BIG_G * (ADVD_CENTRE_M + mass * 1.2e2f)) /
                                    sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z));
                
                if (pos.y > 0)
                {
                    vel = {-1.0f * (radius/2.0f), (pos.x - radius/2.0f) * m + b - pos.y, 0};
                    vel_m /= sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
                    vel.x *= vel_m;
                    vel.y *= vel_m;
                    vel.z *= vel_m;
                    
                }
                else
                {
                    vel = {(radius/2.0f), (pos.x + radius/2.0f) * m + b - pos.y, 0};
                    vel_m /= sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
                    vel.x *= vel_m;
                    vel.y *= vel_m;
                    vel.z *= vel_m;
                }
                
                float pScale = 1.0f;
                m_hPos[i].x = pos.x * pScale;
                m_hPos[i].y = pos.y * pScale;
                m_hPos[i].z = pos.z * pScale;
                m_hPos[i].w = mass;
                
                m_hVel[i].x = vel.x;
                m_hVel[i].y = vel.y;
                m_hVel[i].z = vel.z;
                m_hVel[i].w = mass;
                
                // std::cout << "\n " << m_hVel[i].x << " " << m_hVel[i].y << " " << m_hVel[i].z;
    
            }
        }
            break;
        case NORB_CONFIG_ADV_DISK_COLLSION:
        {
            // hi
            uniform_real_distribution<float> randMassInner(ADVD_M_INNER_MIN, ADVD_M_INNER_MAX);
    
            m_hPos[0].x = 0.0;
            m_hPos[0].y = 0.0;
            m_hPos[0].z = 0.0;
            m_hPos[0].w = ADVD_CENTRE_M;
    
            m_hVel[0].x = 0.0;
            m_hVel[0].y = 0.0;
            m_hVel[0].z = 0.0;
            m_hVel[0].w = ADVD_CENTRE_M;
    
            m_hPos[N/2].x = ADVD_G2_X;
            m_hPos[N/2].y = ADVD_G2_Y;
            m_hPos[N/2].z = ADVD_G2_Z;
            m_hPos[N/2].w = ADVD_CENTRE_M;
    
            m_hVel[N/2].x = -1.f * ADVD_G2_VX;
            m_hVel[N/2].y = -1.f * ADVD_G2_VY;
            m_hVel[N/2].z = -1.f * ADVD_G2_VZ;
            m_hVel[N/2].w = ADVD_CENTRE_M;
    
            float c      = ADVD_C_INNER; // flatness
            float mass;//   = randMassInner(gen);
            // float mass = randMassInner(gen);
            float radius = ADVD_R_INNER;
    
            int start = 1;
            for (int i = start;i < N/2; i++)
            {
                mass = randMassInner(gen);
                if (i == N/2 - ADVD_OUTER_N) {
                    c = ADVD_C_OUTER;
                    mass = ADVD_M_OUTER;
                    // mass = randMassOuter(gen) * 100.0f;
                    radius = ADVD_R_OUTER;
                }
        
                float3 pos;
                while (true)
                {
                    pos.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    pos.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            
                    if (pos.y >= -1.0f * sqrtf(1.0f - powf(pos.x, 2.0f))
                        && pos.y <= sqrtf(1.0f - powf(pos.x, 2.0f)))
                        break;
                }
        
                float zPosMax = sqrtf(c * (1.0f - powf(pos.x, 2.0f)
                                           -powf(pos.y, 2.0f)));
                float zPosMin = -1.0f * zPosMax;
                float zPosRand = rand() / (float) RAND_MAX;
                pos.z = (zPosMax - zPosMin) * zPosRand + zPosMin;
        
                pos.x *= radius;
                pos.y *= radius;
                pos.z *= radius;
        
                float m = pos.y / pos.x;
                m = -1.0f / m;
                float b = pos.y - pos.x * m;
        
                float3 vel;
                // float vel_m = sqrtf(((float)BIG_G * (1e6f + mass * 1.2e-6f)) /
                //                     sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z));
                float vel_m = sqrtf(((float)BIG_G * (ADVD_CENTRE_M + mass * 1.2e2f)) /
                                    sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z));
        
                if (pos.y > 0)
                {
                    vel = {-1.0f * (radius/2.0f), (pos.x - radius/2.0f) * m + b - pos.y, 0};
                    vel_m /= sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
                    vel.x *= vel_m;
                    vel.y *= vel_m;
                    vel.z *= vel_m;
            
                }
                else
                {
                    vel = {(radius/2.0f), (pos.x + radius/2.0f) * m + b - pos.y, 0};
                    vel_m /= sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
                    vel.x *= vel_m;
                    vel.y *= vel_m;
                    vel.z *= vel_m;
                }
        
                float pScale = 1.0f;
                m_hPos[i].x = pos.x * pScale;
                m_hPos[i].y = pos.y * pScale;
                m_hPos[i].z = pos.z * pScale;
                m_hPos[i].w = mass;
        
                m_hVel[i].x = vel.x;
                m_hVel[i].y = vel.y;
                m_hVel[i].z = vel.z;
                m_hVel[i].w = mass;
                }
    
            c      = ADVD_C_INNER; // flatness
            mass   = randMassInner(gen);
            // float mass = randMassInner(gen);
            radius = ADVD_R_INNER;
            start = N/2 + 1;
            for (int i = start;i < N; i++)
            {
                mass = randMassInner(gen);
                if (i == N - ADVD_OUTER_N) {
                    c = ADVD_C_OUTER;
                    mass = ADVD_M_OUTER;
                    // mass = randMassOuter(gen) * 100.0f;
                    radius = ADVD_R_OUTER;
                }
        
                float3 pos;
                while (true)
                {
                    pos.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    pos.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            
                    if (pos.y >= -1.0f * sqrtf(1.0f - powf(pos.x, 2.0f))
                        && pos.y <= sqrtf(1.0f - powf(pos.x, 2.0f)))
                        break;
                }
        
                float zPosMax = sqrtf(c * (1.0f - powf(pos.x, 2.0f)
                                           -powf(pos.y, 2.0f)));
                float zPosMin = -1.0f * zPosMax;
                float zPosRand = rand() / (float) RAND_MAX;
                pos.z = (zPosMax - zPosMin) * zPosRand + zPosMin;
        
                pos.x *= radius;
                pos.y *= radius;
                pos.z *= radius;
        
                float m = pos.y / pos.x;
                m = -1.0f / m;
                float b = pos.y - pos.x * m;
        
                float3 vel;
                // float vel_m = sqrtf(((float)BIG_G * (1e6f + mass * 1.2e-6f)) /
                //                     sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z));
                float vel_m = sqrtf(((float)BIG_G * (ADVD_CENTRE_M + mass * 1.2e2f)) /
                                    sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z));
        
                if (pos.y > 0)
                {
                    vel = {-1.0f * (radius/2.0f), (pos.x - radius/2.0f) * m + b - pos.y, 0};
                    vel_m /= sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
                    vel.x *= vel_m;
                    vel.y *= vel_m;
                    vel.z *= vel_m;
            
                }
                else
                {
                    vel = {(radius/2.0f), (pos.x + radius/2.0f) * m + b - pos.y, 0};
                    vel_m /= sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
                    vel.x *= vel_m;
                    vel.y *= vel_m;
                    vel.z *= vel_m;
                }
        

                m_hPos[i].x = pos.z + (float)ADVD_G2_X;
                m_hPos[i].y = pos.y + (float)ADVD_G2_Y;
                m_hPos[i].z = pos.x + (float)ADVD_G2_Z;
                m_hPos[i].w = mass;
        
                m_hVel[i].x = vel.z - (float)ADVD_G2_VX;
                m_hVel[i].y = vel.y - (float)ADVD_G2_VY;
                m_hVel[i].z = vel.x - (float)ADVD_G2_VZ;
                m_hVel[i].w = mass;
            }
            
        }
            break;
        case NORB_CONFIG_SOLAR:
        {
            // sun
            m_hPos[0].x = 0.f;
            m_hPos[0].y = 0.f;
            m_hPos[0].z = 0.f;
            m_hPos[0].w = 1.f;
            
            m_hVel[0].x = 0.f;
            m_hVel[0].y = 0.f;
            m_hVel[0].z = 0.f;
            m_hVel[0].w = 1.f;
            
            // earth
            m_hPos[1].x = 1.f;
            m_hPos[1].y = 0.f;
            m_hPos[1].z = 0.f;
            m_hPos[1].w = 3.00273e-6f;// 2.9861e-6f;

            m_hVel[1].x = 0.f;
            m_hVel[1].y = 29.78f / (float)KMS_TO_AUD;//29.78f / (float)KMS_TO_AUD;
            m_hVel[1].z = 0.f;
            m_hVel[1].w = 3.00273e-6f;
    
            // venus
            m_hPos[2].x = 0.723f;
            m_hPos[2].y = 0.f;
            m_hPos[2].z = 0.f;
            m_hPos[2].w = 2.447e-6f;

            m_hVel[2].x = 0.f;
            m_hVel[2].y = 35.02f / (float)KMS_TO_AUD;
            m_hVel[2].z = 0.f;
            m_hVel[2].w = 2.447e-6f;
            
        }
            break;
    }
    std::cout << "\nTOTAL MASS ->> " << totalMass;
}
//---------------------------------------


// Print to file
//---------------------------------------
void initialiseForces(float4* pos, float4* force, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == j)
                continue;
            
            float3 r;
    
            // r_ij -> AU [distance]
            r.x = pos[j].x - pos[i].x;
            r.y = pos[j].y - pos[i].y;
            r.z = pos[j].z - pos[i].z;
    
            // distance squared == dot(r_ij, r_ij) + softening^2
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
            distSqr += SOFTENING * SOFTENING;
    
            // inverse distance cubed == 1 / distSqr^(3/2) [fastest method]
            float distSixth = distSqr * distSqr * distSqr;
            float invDistCube = 1.0f / sqrtf(distSixth);
    
            // force = mass_j * inverse distance cube
            float f = pos[j].w * invDistCube;
    
            // acceleration = acceleration_i + force * r_ij
            force[i].x += r.x * f * (float)BIG_G;
            force[i].y += r.y * f * (float)BIG_G;
            force[i].z += r.z * f * (float)BIG_G;
        }
    }
}
//---------------------------------------



//---------------------------------------
// MAIN UPDATE LOOP
//---------------------------------------
void simulate(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2],
              uint m_currentRead, uint m_currentWrite,
              float deltaTime, int N, uint m_p, uint m_q)
{
    // set pos,vel -> update -> get pos,vel ~@ repeat
    
    // Send data to device first
    copyDataToDevice(m_dPos[m_currentRead], m_hPos, N);
    copyDataToDevice(m_dVel[m_currentRead], m_hVel, N);
    copyDataToDevice(m_dForce[m_currentRead], m_hForce, N);


    // Do the thing
    deployToGPU(m_dPos[m_currentRead], m_dPos[m_currentWrite],
                m_dVel[m_currentRead], m_dVel[m_currentWrite],
                m_dForce[m_currentRead], m_dForce[m_currentWrite],
                deltaTime, N, m_p, m_q);
    // Swap read and write
    std::swap(m_currentRead, m_currentWrite);

    // cudaDeviceSynchronize();
    
    // Retrieve data from device
    copyDataToHost(m_hPos, m_dPos[m_currentRead], N);
    copyDataToHost(m_hVel, m_dVel[m_currentRead], N);
    copyDataToHost(m_hForce, m_dForce[m_currentRead], N);

    // Retrieve any CUDA errors and output
    getCUDAError();
}
//---------------------------------------


// CUDA error check
//---------------------------------------
void getCUDAError()
{
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) std::cout << "\nCUDA error:%s\n" << cudaGetErrorString(cudaError);
}
//---------------------------------------


// Finalise & delete arrays TODO: reimplement this
//---------------------------------------
void finalise(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2])
{
    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hForce;
    
    deleteNOrbitalArrays(m_dPos, m_dVel, m_dForce);
}
//---------------------------------------


// A nice little normalisation function
//---------------------------------------
float normalise(float3& vector)
{
    float dist = sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
    if (dist > 1e-6)
    {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }
    return dist;
}
//---------------------------------------



//////////////////////////////////////////////////////
// █▀▀ █▀█ █░█   █▀▀ █░█ █▄░█ █▀▀ ▀█▀ █ █▀█ █▄░█ █▀ //
// █▄█ █▀▀ █▄█   █▀░ █▄█ █░▀█ █▄▄ ░█░ █ █▄█ █░▀█ ▄█ //
//////////////////////////////////////////////////////

extern "C"
{

// Send softening_sqr value to device
//---------------------------------------
void setDeviceSoftening(float softening)
{
    float softeningSq = softening * softening;
    
    cudaMemcpyToSymbol(softeningSqr, &softeningSq, sizeof(float),0);
}
//---------------------------------------


// Send gravitational constant to device
//---------------------------------------
void setDeviceBigG(float G)
{
    cudaMemcpyToSymbol(big_G, &G, sizeof(float),0);
}
//---------------------------------------


// Allocate device memory for variables
//---------------------------------------
void allocateNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2],  int N)
{
    // memory size for device allocation
    uint memSize = sizeof(float4) * N;
    // uint fMemSize = sizeof(float3) * N;
    
    cudaMalloc((void**)&pos[0], memSize);
    cudaMalloc((void**)&pos[1], memSize);
    cudaMalloc((void**)&vel[0], memSize);
    cudaMalloc((void**)&vel[1], memSize);
    cudaMalloc((void**)&force[0], memSize);
    cudaMalloc((void**)&force[1], memSize);
}
//---------------------------------------


// De-allocate device memory variables
//---------------------------------------
void deleteNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2])
{
    cudaFree((void**)pos[0]);
    cudaFree((void**)pos[1]);
    cudaFree((void**)vel[0]);
    cudaFree((void**)vel[1]);
    cudaFree((void**)force[0]);
    cudaFree((void**)force[1]);
}
//---------------------------------------


// Copy data from host[CPU] ->> device[GPU]
//---------------------------------------
void copyDataToDevice(float4* device, const float4* host, int N)
{
    uint memSize = sizeof(float4) * N;
    cudaMemcpy(device, host, memSize, cudaMemcpyHostToDevice);
    getCUDAError();
}
//---------------------------------------


// Copy data from device[GPU] ->> host[CPU]
//---------------------------------------
void copyDataToHost(float4* host, const float4* device, int N)
{
    uint memSize = sizeof(float4) * N;
    cudaMemcpy(host, device, memSize, cudaMemcpyDeviceToHost);
    getCUDAError();
}
//---------------------------------------


// Initiates GPU kernel computations every iteration
//---------------------------------------
void deployToGPU(float4* oldPos, float4* newPos,
                 float4* oldVel, float4* newVel,
                 float4* oldForce, float4* newForce,
                 float deltaTime, int N, uint p, uint q)
{
    uint shMemSize = p * q * sizeof(float4);
    
    // thread and grid time :D
    dim3 threads(p, q, 1);
    dim3 grid(N / p, 1, 1);
    
    // DEPLOY
    /*If multithreading is enabled (i.e. q>1 | multiple threads per
     * body) then the more complicated code is executed (using bool template
     * over in the kernel), and if it is not, then the simpler code is executed*/
    if (grid.y == 1)
    {
        integrateNOrbitals<<<grid, threads, shMemSize
        >>>(oldPos, newPos, oldVel, newVel, oldForce, newForce, deltaTime, N, false);
    }
    else
    {
        integrateNOrbitals<<<grid, threads, shMemSize
        >>>(oldPos, newPos, oldVel, newVel, oldForce, newForce, deltaTime, N, true);
    }
}
//---------------------------------------
}

// MISC FUNCTIONS

// Timer, very simple
//---------------------------------------
void runTimer(std::chrono::system_clock::time_point start,
              int N_orbitals, bool init)
{
    if (init)
    {
        start = std::chrono::system_clock::now();
        std::time_t start_time = std::chrono::system_clock::to_time_t(start);
        std::cout << "Starting Simulation at ->> " << std::ctime(&start_time)
                  << "For N == " << N_orbitals << " || Iterations == " << ITERATIONS;
    }
    else // end
    {
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "\nFinished Computation at ->> " << std::ctime(&end_time)
                  << "Elapsed Time : " << elapsed_seconds.count() << "s"
                  << " for N = " << N_orbitals << std::endl;
    }
}
//---------------------------------------


// Initialise OpenGL for particle rendering
//---------------------------------------
GLFWwindow* initGL(GLFWwindow *window)
{
    if(!glewInit())
    {
        std::cout << "\nCritical OpenGL error ::\nFailed to initialise GLEW\nTERMINATING";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    if (!glfwInit())
    {   // SAFETY CHECK
        std::cout << "\nCritical OpenGL error ::\nFailed to initialise GLFW\nTERMINATING";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    
    // CREATE WINDOW IN WINDOWED MODE
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "orbiterV6", nullptr, nullptr);
    
    if (!window)
    {   // SAFETY CHECK
        std::cout << "\n Critical OpenGL error ::\nFailed to open GLFW window\nTERMINATING";
        glfwTerminate();
        exit (EXIT_FAILURE);
    }
    // CALLBACKS
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // -> viewport
    glfwSetKeyCallback(window, key_callback); // -> key input
    glfwSetScrollCallback(window, scroll_callback); // -> scroll input
    
    // set window context | synchronise to refresh rate with swapinterval
    glfwMakeContextCurrent(window);
    
    // SET THE VIEWPORT
    glViewport(0, 0, WIDTH, HEIGHT);
    // SET THE PROJECTION TRANSFORM
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV, (GLfloat)WIDTH/(GLfloat)HEIGHT, 0, V_FAR); // TODO: rename to Z_FAR
    
    // PREPARE WINDOW
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL); // experimental
    glClearColor(0.0, 0.0, 0.0, 1.0);
    
    // PREPARE RENDERER
    renderer = new NbodyRenderer;
    
    // TODO: add GL error check here
    return window;
}
//---------------------------------------


// A nice little vector cross product function
//---------------------------------------
float3 cross(float3 v0, float3 v1)
{
    float3 rt;
    rt.x = v0.y*v1.z-v0.z*v1.y;
    rt.y = v0.z*v1.x-v0.x*v1.z;
    rt.z = v0.x*v1.y-v0.y*v1.x;
    return rt;
}
//---------------------------------------


// A nice little vector dot product function
//---------------------------------------
float dot(float3 v0, float3 v1)
{
    return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
}
//---------------------------------------


// Processes user input for sim control
//---------------------------------------
void processInput(GLFWwindow *window)
{
    // if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        shiftSpeed = 1 * SHIFT_FACTOR;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
        shiftSpeed = 1;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        zTrans += shiftSpeed * MOVE_SPEED * 1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        zTrans -= shiftSpeed * MOVE_SPEED * 1.0f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        xTrans += shiftSpeed * MOVE_SPEED;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        xTrans -= shiftSpeed * MOVE_SPEED;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        yTrans -= shiftSpeed * MOVE_SPEED;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        yTrans += shiftSpeed * MOVE_SPEED;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        xRot += shiftSpeed * 1;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        xRot -= shiftSpeed * 1;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        yRot += shiftSpeed * 1;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        yRot -= shiftSpeed * 1;
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
        zRot += shiftSpeed * 1;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
        zRot -= shiftSpeed * 1;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        zoom += (zoom * (float)ZOOM_SCALE);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        zoom -= (zoom * (float)ZOOM_SCALE);
    
    // timestep adjustment
    if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
        timestep -= 0.1f ;
    if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS)
        timestep += 0.1f;
}
//---------------------------------------


// Triggered when scrollwheel is used
//---------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{   // SCROLL => ZOOM
    zoom += (float)yoffset * (zoom * (float)ZOOM_SCALE);
}
//---------------------------------------


// Triggered when key state changes
//---------------------------------------
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{   // THIS GETS CALLED FOR ALL KEY EVENTS DETECTED
    if (key == GLFW_KEY_F11 && action == GLFW_PRESS)
    {   // CHECKING FOR FULLSCREEN OR NOT
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        GLFWmonitor *curMonitor = glfwGetWindowMonitor(window);
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);
        
        if (curMonitor == nullptr)
            glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        if (curMonitor != nullptr)
            glfwSetWindowMonitor(window, nullptr, 0,0, WIDTH, HEIGHT, 0);
        glfwSwapBuffers(window);
    }
    // BACKSPACE KEY => CLOSE WINDOW
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    // Q ENABLES/DISABLES AUTO-ROTATE
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        if (!rotateCam) {
            rotateCam = true;
        } else rotateCam = false;
    }
    // COMMA/PERIOD FOR TIMESTEP
    // if (key == GLFW_KEY_COMMA && action == GLFW_PRESS)
    //     timestep -= 0.25f;
    // if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS)
    //     timestep += 0.25f;
}
//---------------------------------------


// Triggered when the OpenGL window is resized
//---------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{   // DYNAMICALLY UPDATES VIEWPORT UPON WINDOW RESIZE
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV, (GLfloat)width/(GLfloat)height, 0, V_FAR);
    // TODO: rename to Z_FAR
}
//---------------------------------------



