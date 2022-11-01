//
// Created by quartzar on 18/10/22.
//

#include <cmath>
#include <iostream>
#include <random>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <ctime>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include "NbodyN2.cuh"
#include "CONSTANTS.h"

bool renderEnabled = true;
bool rotateCam = false;
bool debugMode = false;
bool infoMode = false;
bool diskSim = false;
bool shellSim = true;
bool solarSystem = false;
bool trailMode = false;
bool printToFile = false;

NbodyRenderer *renderer = nullptr;


int main()
{   // BEGIN TIMER
    std::cout << "N -> " << N_BODIES;
    auto start = std::chrono::system_clock::now();
    
    zoom = 1;
    firstMouse = true;
    
    // OpenGL INITIALISER
    GLFWwindow *window = nullptr;
    if (renderEnabled) {
        window = initGL(window);
        // initCamera();
        glfwGetWindowPos(window, &xMonitor, &yMonitor);
    }
    
    // OUTPUT FILE
    std::string outputFName = "outputCSV.csv";
    if (printToFile)
    {
        std::ofstream outputFile(outputFName);
        outputFile << "T_S" << "," << "xPos" << "," << "yPos" << "," << "zPos" << ","
                                   << "xVel" << "," << "yVel" << "," << "zVel" << std::endl;
        outputFile.close();
    }
    
    // INITIALISE ORBITAL ARRAYS
    auto* orbPos = new float4[N_BODIES]; // x, y, z, w=mass
    auto* orbVel = new float3[N_BODIES]; // vx, vy, vz

    // IC GENERATOR
    initialiseOrbitals(orbPos, orbVel);
    // BEGIN SIMULATION
    simulateOrbitals(orbPos, orbVel, window, outputFName);
    // ...
    
    // OUTPUT RUN TIME
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "\nFinished Simulation at : " << std::ctime(&end_time)
              << "Elapsed Time : " << elapsed_seconds.count() << "s"
              << " for N = " << N_BODIES << std::endl;

    // TERMINATE SUCCESSFULLY
    glfwTerminate();
    exit(EXIT_SUCCESS);
}


void initialiseOrbitals(float4* orbPos, float3* orbVel) {
    using std::uniform_real_distribution;
    float totalMass = 0.0;
    if (!diskSim && !solarSystem) {
        // random distribution functions
        uniform_real_distribution<float> randXPos (-WIDTH/2.0, WIDTH/2.0);
        uniform_real_distribution<float> randYPos (-HEIGHT/2.0, HEIGHT/2.0);
        uniform_real_distribution<float> randVel (-INIT_VEL, INIT_VEL);
        uniform_real_distribution<float> randHeight (-SYSTEM_THICKNESS, SYSTEM_THICKNESS);
        uniform_real_distribution<float> randMass (INIT_M_LOWER, INIT_M_HIGHER);
        std::default_random_engine gen (SEED); // seeds!!!! // NOLINT(cert-msc51-cpp)

        // initialisation procedure w/ calculation of total mass
        for (int i = 0; i < N_BODIES; i++) {

            // random position assignment
            orbPos[i].x = randXPos(gen);
            orbPos[i].y = randYPos(gen);
            orbPos[i].z = randHeight(gen);
            // random velocity assignment
            orbVel[i].x = randVel(gen);
            orbVel[i].y = randVel(gen);
            orbVel[i].z = randVel(gen);
            // random mass assignment
            orbPos[i].w = randMass(gen);

            totalMass += orbPos[i].w;
        }

    }
    if (diskSim && !solarSystem) {
        // disk galaxy
        uniform_real_distribution<float> randAngle(0.0f, 200.0 * PI);
        uniform_real_distribution<float> randRadius(((float)INNER_BOUND + CENTRE_STAR_M) / (SYSTEM_SIZE + SYSTEM_SIZE), SYSTEM_SIZE);
        uniform_real_distribution<float> randHeight(0.0, SYSTEM_THICKNESS);
        uniform_real_distribution<float> randMass(INIT_M_LOWER, INIT_M_HIGHER);
        std::default_random_engine gen(SEED); // NOLINT(cert-msc51-cpp)

        // centre star
        orbPos[0].x = 0.0;
        orbPos[0].y = 0.0;
        orbPos[0].z = 0.0;

        orbVel[0].x = 0.0;
        orbVel[0].y = 0.0;
        orbVel[0].z = 0.0;
        // random mass assignment
        orbPos[0].w = CENTRE_STAR_M;
        totalMass += CENTRE_STAR_M;


        // rest of stars
        for (int i = 1; i < N_BODIES; i++) {
            // variables and things and stuff IDK
            float theta = randAngle(gen);
            float radius = randRadius(gen);
            // float velocity = 1.0f * sqrtf(((float)BIG_G * (float)CENTRE_STAR_M * 1.25f) / radius);
            float velocity = 1.0f* sqrtf(BIG_G * CENTRE_STAR_M * 1.0f/ radius * radius);
            // float velocity = powf((((float)BIG_G * (radius - INNER_BOUND) / SYSTEM_SIZE)) /radius, 0.5f);

            // set position of orbital
            orbPos[i].x = radius * cosf(theta+0.2f); // cos
            orbPos[i].y = radius * sinf(theta); // sin
            orbPos[i].z = randHeight(gen) - SYSTEM_THICKNESS / 2.0f;

            // set velocity of orbital
            float rotation = 1.0f; // 1 -> clockwise || -1 -> anticlockwise
            orbVel[i].x =  rotation * velocity * sinf(theta+0.1f); // sin
            orbVel[i].y = -rotation * velocity * cosf(theta); // cos
            orbVel[i].z = 0.0;

            // mass
            orbPos[i].w = randMass(gen);
            totalMass += orbPos[i].w;
        }
    }

    if (!diskSim && solarSystem) {
        // sun
        orbPos[0].x = 0.0;
        orbPos[0].y = 0.0;
        orbPos[0].z = 0.0;

        orbVel[0].x = 0.0;
        orbVel[0].y = 0.0;
        orbVel[0].z = 0.0;

        orbPos[0].w = 1.0f;

        // earth
        orbPos[1].x = 1.0f;
        orbPos[1].y = 0.0;
        orbPos[1].z = 0.0;

        orbVel[1].x = 0.0f;
        orbVel[1].y = 29.78f / (float)KMS_TO_AUD;
        orbVel[1].z = 0.0;

        orbPos[1].w = 2.9861e-6f;

        // venus
        orbPos[2].x = -0.723f;
        orbPos[2].y = 0.0;
        orbPos[2].z = 0.0;

        orbVel[2].x = 0.0f;
        orbVel[2].y = -35.02 / (float)KMS_TO_AUD;
        orbVel[2].z = 0.0;

        orbPos[2].w = 2.447e-6f;
    }

    // FROM NVIDIA
    if (shellSim) {
        
        uniform_real_distribution<float> randF (0.0f, (float)RAND_MAX);
        uniform_real_distribution<float> randMass(INIT_M_LOWER, INIT_M_HIGHER);
        std::default_random_engine gen(SEED); // NOLINT(cert-msc51-cpp)
        
        float scale = SYSTEM_SIZE;
        float vScale = scale * (float)VEL_SCALE / (float)KMS_TO_AUD;
        float inner = 2.5f * scale;
        float outer = 4.0f * scale;
    
        orbPos[0].x = 0.0;
        orbPos[0].y = 0.0;
        orbPos[0].z = 0.0;
    
        orbVel[0].x = 0.0;
        orbVel[0].y = 0.0;
        orbVel[0].z = 0.0;
    
        orbPos[0].w = CENTRE_STAR_M;
        
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
            
            orbPos[i].x = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
            orbPos[i].x = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
            orbPos[i].x = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
            
            orbPos[i].w = randMass(gen);
            
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
            float3 vv = {orbPos[i].x, orbPos[i].y, orbPos[i].z};
            vv = cross(vv, axis);
            orbVel[i].x = vv.x * vScale;
            orbVel[i].y = vv.y * vScale;
            orbVel[i].z = vv.z * vScale;
            
            i++;
        }
      
        // NbodyRenderer().setPositions(reinterpret_cast<float *>(orbPos));
    }
    
    std::cout << "\nTOTAL MASS -> " << totalMass;
}


void simulateOrbitals(float4* orbPos, float3* orbVel, GLFWwindow* window, const std::string& outputFName)
{
    // new arrays for device memory
    float4* d_orbPos; float3 *d_orbVel;
    float3 *orbForce;
   
    // allocating memory for arrays [devPtr -> pointer to allocated memory]
    cudaMalloc(&d_orbPos, N_BODIES * sizeof(float4));
    cudaMalloc(&d_orbVel, N_BODIES * sizeof(float3));
    cudaMalloc(&orbForce, N_BODIES * sizeof(float3));
    
    // CUDA grid allocation for parallelised GPU execution
    int nBlocks = (N_BODIES + N_B_MULTIPLIER - 1) / N_B_MULTIPLIER;
    long nBlocksSqr  = (N_BODIES / N_B_MULTIPLIER) * (N_BODIES / N_B_MULTIPLIER);
    dim3 grid(nBlocksSqr, N_B_MULTIPLIER, 1);

    // BEGIN SIMULATION
    for (int step = 1; step < ITERATIONS + 1; step++) {
        if (debugMode) std::cout << "\n-------------------" <<"\nBEGINNING TIMESTEP |->> " << step;

        // CUDA Memcpy copies data between host (this) and device
        if (debugMode) std::cout << "\nSTART KERNEL";
        cudaMemcpy(d_orbPos, orbPos, N_BODIES * sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_orbVel, orbVel, N_BODIES * sizeof(float3), cudaMemcpyHostToDevice);
        
        // CLEAR PREVIOUS FORCES
        clearForce<<<nBlocks + 1, N_B_MULTIPLIER>>>(orbForce);
        cudaDeviceSynchronize();
        
        // UPDATE HALF KICK POSITIONS
        updateOrbitals<<<nBlocks + 1, N_B_MULTIPLIER>>>(d_orbPos, d_orbVel, orbForce, 0);
        cudaDeviceSynchronize();
        
        // CALCULATE GRAVITATIONAL FORCES [DRIFT-STEP]
        orbitalOrbitalInteraction<<<grid, N_B_MULTIPLIER>>>(d_orbPos, d_orbVel, orbForce);
        cudaDeviceSynchronize();
        
        // ERROR CHECK
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) std::cout << "\nCUDA error:%s\n" << cudaGetErrorString(cudaError);

        // UPDATE FULL KICK POSITIONS
        updateOrbitals<<<nBlocks + 1, N_B_MULTIPLIER>>>(d_orbPos, d_orbVel, orbForce, 1);
        
        // COPY FLOATS BACK TO HOST
        cudaMemcpy(orbPos, d_orbPos, N_BODIES * sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(orbVel, d_orbVel, N_BODIES * sizeof(float3), cudaMemcpyDeviceToHost);
        
        // DEBUG | INFO MODE
        if (debugMode) std::cout << "\nCOMPLETED TIMESTEP |->> " << step << std::flush;
        if (infoMode) {
            std::cout << "\nACCELERATION |> " << orbVel[1].x << " " << orbVel[1].y << " " << orbVel[1].z;
            std::cout << "\nPOSITION |> " << orbPos[1].x << " " << orbPos[1].y << " " << orbPos[1].z;
        }
        // RENDERING MODE
        if (renderEnabled && step % RENDER_INTERVAL == 0) {
            // CHECK FOR INPUT FIRST
            processInput(window);
            
            // CLOSE WINDOW IF ESC PRESSED
            if (glfwWindowShouldClose(window))
            {
                std::cout << "\nPROGRAM TERMINATED BY USER\nEXITING AT STEP " << step;
                glfwTerminate();
                exit(EXIT_SUCCESS);
            }
            
            // CALL RENDERER
            renderer->setPositions(reinterpret_cast<float *>(orbPos));
            renderer->display(renderMode, trailMode, zoom, xRot, yRot, zRot, xTrans, yTrans, zTrans);
    
            // SWAP BUFFERS
            glfwPollEvents();
            glfwSwapBuffers(window);
            
            // set window title to current timestep
            std::string s = std::to_string(step);
            const char* cstr = s.c_str();
            glfwSetWindowTitle(window, cstr);
    

            // CAMERA ROTATE
            if (rotateCam && !solarSystem)
            {
                GLfloat rotation = 2.0f * sinf((float) step / 1000000.0f) * (180.0f / (float) PI);
                // glRotatef(rotation, -1 * ROT_SPEED, 0, ROT_SPEED);
                yRot -= rotation;
                zRot += rotation;
            }
        }
        // PRINT TO FILE MODE
        if (printToFile) {
            // std::cout << "\nVelocity -> " << orbVel[1].x * KMS_TO_AUD << " " << orbVel[1].y * KMS_TO_AUD << " " << orbVel[1].z * KMS_TO_AUD ;
            // std::cout << "\nPosition -> " << orbPos[1].x << " " << orbPos[1].y << " " << orbPos[1].z;
            writeCSV(orbPos, orbVel, step, outputFName);
        }
    }
}


__global__ void
clearForce(float3* orbForce) {
    // id from thread allocation
    uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N_BODIES) {
        orbForce[id].x = 0.0;
        orbForce[id].y = 0.0;
        orbForce[id].z = 0.0;
    }
}


__global__ void
orbitalOrbitalInteraction(float4* orbPos,float3* orbVel, float3* orbForce) {

    // id from thread allocation
    uint id = blockDim.x * blockIdx.x + threadIdx.x + blockDim.y * blockIdx.y;
    long i = id%N_BODIES;
    long j = id/N_BODIES;

    if (i < N_BODIES && j < N_BODIES && i != j) {

        float3 r;
        float3 a;

        // r_ij -> metres [distance]
        r.x = orbPos[i].x - orbPos[j].x;
        r.y = orbPos[i].y - orbPos[j].y;
        r.z = orbPos[i].z - orbPos[j].z;

        // distance squared == dot(r_ij, r_ij) + softening^2
        float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
        distSqr += SOFTENING;

        // inverse distance cubed == 1 / distSqr^(3/2) [fastest method]
        float distSixth = distSqr * distSqr * distSqr;
        float invDistCube = 1.0f / sqrtf(distSixth);

        // force = mass_j * inverse distance cube
        float force = orbPos[j].w * invDistCube;

        // acceleration = acceleration_i + force * r_ij
        a.x = r.x * force * (-1.0f * (float)BIG_G);
        a.y = r.y * force * (-1.0f * (float)BIG_G);
        a.z = r.z * force * (-1.0f * (float)BIG_G);

        orbForce[i].x += a.x;
        orbForce[i].y += a.y;
        orbForce[i].z += a.z;
    }
}


__global__ void
updateOrbitals(float4* orbPos, float3* orbVel, float3* orbForce, int fKick) {
    // LEAPFROG-VERLET 3-STEP INTEGRATOR
    // https://en.wikipedia.org/wiki/Leapfrog_integration
    // https://www.maths.tcd.ie/~btyrrel/nbody.pdf
    
    // id from thread allocation
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N_BODIES) {
        if (fKick == 0) {
            // first -> update just positions with old velocities
            orbPos[id].x += orbVel[id].x * ((float)TIME_STEP * 0.5f);
            orbPos[id].y += orbVel[id].y * ((float)TIME_STEP * 0.5f);
            orbPos[id].z += orbVel[id].z * ((float)TIME_STEP * 0.5f);
        }
        else {
            // second -> update velocities with new force
            orbVel[id].x += orbForce[id].x * (float)TIME_STEP;
            orbVel[id].y += orbForce[id].y * (float)TIME_STEP;
            orbVel[id].z += orbForce[id].z * (float)TIME_STEP;
            // third -> update positions with new velocities
            orbPos[id].x += orbVel[id].x * ((float)TIME_STEP * 0.5f);
            orbPos[id].y += orbVel[id].y * ((float)TIME_STEP * 0.5f);
            orbPos[id].z += orbVel[id].z * ((float)TIME_STEP * 0.5f);
        }
    }
}


GLFWwindow *initGL(GLFWwindow *window)
{
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
    // glfwSetCursorPosCallback(window, cursor_position_callback);
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // if (glfwRawMouseMotionSupported())  // enable raw mouse motion
    //     glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    
    // set window context | synchronise to refresh rate with swapinterval
    glfwMakeContextCurrent(window);
    
    // glfwSwapInterval( 1 );
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

void initCamera() {
    GLdouble vvLeft, vvRight, vvBottom, vvTop, vvNear, vvFar;
    // SET VIEWPORT COORDINATE DIMENSIONS
    if (solarSystem) {
        vvLeft = -((float)WIDTH/V_S_SYSTEM_SCALE); vvRight = ((float)WIDTH/V_S_SYSTEM_SCALE);
        vvBottom = -((float)HEIGHT/V_S_SYSTEM_SCALE); vvTop = ((float)HEIGHT/V_S_SYSTEM_SCALE);
        vvNear = 0.0; vvFar = V_FAR;
    } else {
        vvLeft = -WIDTH / 2.0; vvRight = WIDTH / 2.0;
        vvBottom = -HEIGHT / 2.0; vvTop = HEIGHT / 2.0;
        vvNear = 0.0; vvFar = V_FAR;
    }
    
    GLdouble vvDepth = vvFar - vvNear;
    GLdouble vvHeight = vvTop - vvBottom;
    GLdouble vvFovDegs = FOV;
    GLdouble vvFovRads = vvFovDegs * (PI/180);
    vvNear = (vvHeight / 2.0) / tan(vvFovRads / 2.0);
    vvFar = vvNear + vvDepth;
    
    // SET VIEWPORT AND VIEWING MODE
    glFrustum(vvLeft, vvRight, vvBottom, vvTop, vvNear, vvFar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(0.0, 0.0, -(vvNear  + (vvDepth / 2.0)));
    if (!solarSystem | !trailMode)
        glRotatef((GLdouble)VIEW_ANGLE * 1.0, 1, 0, 1);
    
}

void processInput(GLFWwindow *window)
{
    // if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        shiftSpeed = 1 * SHIFT_FACTOR;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
        shiftSpeed = 1;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        zTrans += shiftSpeed * MOVE_SPEED * 0.25f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        zTrans -= shiftSpeed * MOVE_SPEED * 0.25f;
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
}


// glfw: whenever there is cursor input, this is called
// -------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{   // SCROLL => ZOOM
    zoom += (float)yoffset * (zoom * (float)ZOOM_SCALE);
}


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
            glfwSetWindowMonitor(window, nullptr, xMonitor, yMonitor, WIDTH, HEIGHT, 0);
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
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{   // DYNAMICALLY UPDATES VIEWPORT UPON WINDOW RESIZE
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(FOV, (GLfloat)WIDTH/(GLfloat)HEIGHT, 0, V_FAR);
    // TODO: rename to Z_FAR
}


void writeCSV(float4* orbPos, float3* orbVel, int timeStep, const std::string& outputFName) {

    std::ofstream outputFile;
    outputFile.open(outputFName, std::ios::app); // open file

    float xPos = orbPos[1].x;
    float yPos = orbPos[1].y;
    float zPos = orbPos[1].z;
    float xVel = orbVel[1].x * (float)KMS_TO_AUD;
    float yVel = orbVel[1].y * (float)KMS_TO_AUD;
    float zVel = orbVel[1].z * (float)KMS_TO_AUD;

    outputFile << timeStep << "," << xPos << "," << yPos << "," << zPos << ","
                                  << xVel << "," << yVel << "," << zVel << std::endl;
}

float
normalise(float3& vector)
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

float
dot(float3 v0, float3 v1)
{
    return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z;
}

float3
cross(float3 v0, float3 v1)
{
    float3 rt;
    rt.x = v0.y*v1.z-v0.z*v1.y;
    rt.y = v0.z*v1.x-v0.x*v1.z;
    rt.z = v0.x*v1.y-v0.y*v1.x;
    return rt;
}
