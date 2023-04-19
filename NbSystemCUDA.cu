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
#include <cstdio>
#include <filesystem>

#include <GL/glew.h> // glut
#include <GLFW/glfw3.h>

// lognormal distribution
#include <map>
#include <iomanip>
// #include <gsl>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>

#include "NbKernel_N2.cuh"
#include "CONSTANTS.h"
#include "NbSystemCUDA.cuh"
// #include "gnuplot-iostream.h"

/////////////////////////////////////////
// █▀█░█▀█░█▄▄░█░▀█▀░█▀▀░█▀█░░░█░█░▀▀█ //
// █▄█░█▀▄░█▄█░█░░█░░██▄░█▀▄░░░▀▄▀░░░█ //
/////////////////////////////////////////

extern __constant__ float softeningSqr;
extern __constant__ float big_G;
extern __constant__ float eta_acc;
extern __constant__ float eta_vel;

//------------PARAMETERS---------------//
NbodyRenderer::RenderMode renderMode = NbodyRenderer::POINTS;
NBodyICConfig sysConfig = NORB_SMALLN_CLUSTER;//NORB_CONFIG_SOLAR;
NbodyIntegrator integrator = LEAPFROG_VERLET;
NbodyRenderer *renderer = nullptr;
// booleans =>
bool displayEnabled = true;
bool outputBinary = true;
bool glxyCollision = false;
bool colourMode = false;
bool trailMode = false;
bool rotateCam = false;
//---------------------------------------q

/////////////////////////////////////////

//---------------------------------------
int main(int argc, char** argv)
{
    //-------------------------
    // CPU data =>
    float4 *m_hPos, *m_hVel, *m_hForce;
    float m_hDeltaTime;
    //-------------------------
    // memory transfers =>
    uint m_currentRead, m_currentWrite;
    //-------------------------
    // GPU data =>
    float4 *m_dPos[2], *m_dVel[2], *m_dForce[2];
    float *m_dDeltaTime[2];
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
    /* Binary file output */
    // std::ofstream snapshot_file;
    std::string snapshot_filename;
    std::string output_directory;

    
    //-------------------------
    // Simulation =>
    int iteration;
    int total_iterations;
    // int N_bodies;
    int snapshot_interval;
    float softening_factor;
    int snapshot_counter = 1;
    uint m_p;
    uint m_q;
    // N_orbitals = N_BODIES;
    iteration = 0;
    total_iterations = ITERATIONS;
    snapshot_interval = SNAPSHOT_INTERVAL;
    softening_factor = SOFTENING;
    // timestep = TIME_STEP;
    // float deltaTime = TIME_STEP;
    m_currentRead = 0;
    m_currentWrite = 1;
    m_p = P;
    m_q = Q;
    zoom = 1;
    ////////////////////////////////
    std::string simulation_base;
    int N_bodies;
    float softening;
    float time_start;
    float time_end;
    float current_time = 0.f;
    float delta_time;
    float snap_rate;
    float time_since_snap = 0.f;
    bool cross_time = false;
    float eta_cross;
    ////////////////////////////////
    // Read in parameters from file
    readParameters("../parameters.dat", simulation_base, N_bodies, softening, time_start,
                   time_end, snap_rate, delta_time, cross_time, eta_cross, eta_acc, eta_vel);
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Simulation base: " << simulation_base << std::endl;
    std::cout << "N_bodies: " << N_bodies << std::endl;
    std::cout << "Softening: " << softening << std::endl;
    std::cout << "Time start: " << time_start << std::endl;
    std::cout << "Time end: " << time_end << std::endl;
    std::cout << "Snap rate: " << snap_rate << std::endl;
    std::cout << "Delta time: " << delta_time << std::endl;
    std::cout << "Cross time: " << cross_time << std::endl;
    std::cout << "Eta cross: " << eta_cross << std::endl;
    std::cout << "Eta acc: " << eta_acc << std::endl;
    std::cout << "Eta vel: " << eta_vel << std::endl;
    std::cout << "---------------------------------" << std::endl;
    ////////////////////////////////
    //---------------------------------------
    // INITIALISE ARRAYS & ALLOCATE DEVICE STORAGE
    //---------------------------------------
    
    // OLD / HOST
    m_hPos = new float4[N_bodies]; // x, y, z, mass
    m_hVel = new float4[N_bodies]; // vx,vy,vz, empty
    m_hForce = new float4[N_bodies]; // fx, fy, fz, empty
    // m_hDeltaTime = new float; // dt
    m_hDeltaTime = delta_time;
    // m_hDeltaTime = reinterpret_cast<float *>(TIME_STEP);
    // NEW / DEVICE
    m_dPos[0] = m_dPos[1] = nullptr;
    m_dVel[0] = m_dVel[1] = nullptr;
    m_dForce[0] = m_dForce[1] = nullptr;
    m_dDeltaTime[0] = m_dDeltaTime[1] = nullptr;
    // set memory for host arrays
    memset(m_hPos, 0, N_bodies*sizeof(float4));
    memset(m_hVel, 0, N_bodies*sizeof(float4));
    memset(m_hForce, 0, N_bodies*sizeof(float4));
    // memset(&m_hDeltaTime, 0, sizeof(float));
    getCUDAError();
    // set memory for device arrays
    allocateNOrbitalArrays(m_dPos,m_dVel, m_dForce, m_dDeltaTime, N_bodies);
    getCUDAError();
    // set device constants
    setDeviceSoftening(softening * softening);
    setDeviceBigG(1.0f * BIG_G);
    setDeviceEtaAcc(ETA_ACC);
    setDeviceEtaVel(ETA_VEL);
    getCUDAError();
    
    //---------------------------------------
    /////////////////////////////////////////
    //---------------------------------------
    
    
    // BEGIN TIMER
    runTimer(start, N_bodies, true);
    
    
    
    // INITIALISE OPENGL
    if (displayEnabled)
    {
        // glutInit(&argc, argv);
        // glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
        window = initGL(window);
    }
    
    // Create new directory for output
    if (outputBinary) {
        // output_directory = "../out/" + getCurrentTime();
        output_directory = "../out/" + simulation_base;
        std::filesystem::create_directory(output_directory);
        if (std::filesystem::exists(output_directory)) {
            deleteFilesInDirectory(output_directory);
        } else {
            std::filesystem::create_directory(output_directory);
        }
    }
    // Set initial timestep
    // m_hDeltaTime = TIME_STEP;
    // Randomise Orbitals
    randomiseOrbitals(sysConfig, m_hPos, m_hVel, N_bodies);
    
    // Set Initial Forces [only run for solar system, HUGE performance hit]
    // if (sysConfig == NORB_CONFIG_SOLAR)
    initialiseForces(m_hPos, m_hForce, N_bodies);
    
    // Calculate and set the crossing time if needed
    if (cross_time)
    {
        float t_crossing = calculateCrossingTime(m_hVel, N_bodies);
        time_end = eta_cross * t_crossing;
        std::cout << "\n---------------------------------" << std::endl;
        std::cout << "Crossing time: " << t_crossing / 365.25 << " years" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    
    //---------------------------------------
    // MAIN UPDATE LOOP
    while (current_time < time_end)
    {
        if (iteration  % 10000 == 0)
            std::cout << "\nSTEP =>> " << iteration << std::flush;
    
        // Write a snapshot every snap_rate days
        if (outputBinary && time_since_snap >= snap_rate)
        {
            std::stringstream snapshot_filename_ss;
            snapshot_filename_ss << output_directory << "/snapshot_"
                                 << std::setfill('0') << std::setw(6) << std::to_string(snapshot_counter) << ".bin";
        
            snapshot_filename = snapshot_filename_ss.str();
            snapshot_counter++;
        
            writeBinaryData(snapshot_filename, current_time, m_hDeltaTime,
                            softening_factor, N_bodies, m_hPos, m_hVel, m_hForce);
            time_since_snap = 0.f;
        }
        
        simulate(m_hPos, m_dPos,
                 m_hVel, m_dVel,
                 m_hForce, m_dForce,
                 m_currentRead, m_currentWrite,
                 m_hDeltaTime, m_dDeltaTime, N_bodies, m_p, m_q);
        // std::cout << "\n m_hDeltaTime = " << m_hDeltaTime << std::flush;
        
        if (iteration % TIME_STEP_INTERVAL == 0)
        {
            m_hDeltaTime = calculateTimeStep(m_hPos, m_hVel, m_hForce, m_hDeltaTime, N_bodies);
            // std::cout << "\n m_hDeltaTime = " << m_hDeltaTime << std::flush;
        }
        
        if (displayEnabled && iteration%RENDER_INTERVAL == 0)
        {
            // CHECK FOR INPUT FIRST
            processInput(window);
    
            // CLOSE WINDOW IF ESC PRESSED
            if (glfwWindowShouldClose(window))
            {
                std::cout << "\nPROGRAM TERMINATED BY USER\nEXITING AT STEP " << iteration;
                runTimer(start,  N_bodies,false);
                finalise(m_hPos, m_dPos,
                         m_hVel, m_dVel,
                         m_hForce, m_dForce, m_dDeltaTime);
                glfwTerminate();
                exit(EXIT_SUCCESS);
            }
            
            // Render
            renderer->setPositions(reinterpret_cast<float *>(m_hPos));
            renderer->setVelocities(reinterpret_cast<float *>(m_hVel));
            renderer->display(renderMode, N_bodies, zoom, xRot, yRot, zRot, xTrans, yTrans, zTrans, trailMode, colourMode);
    
            glfwSwapBuffers(window);
            // glutSwapBuffers();
            glfwPollEvents();
    
            // Set window title to current timestep
            std::string s = std::to_string(m_hDeltaTime);//iteration);
            const char* cstr = s.c_str();
            glfwSetWindowTitle(window, cstr);
        }
        time_since_snap += m_hDeltaTime;
        current_time += m_hDeltaTime;
        iteration++;
    }
    //---------------------------------------
    
    // END TIMER
    runTimer(start,  N_bodies,false);
    
    // DELETE ARRAYS
    finalise(m_hPos, m_dPos,
             m_hVel, m_dVel,
             m_hForce, m_dForce, m_dDeltaTime);
    
    // TERMINATE SUCCESSFULLY
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
//---------------------------------------

// Read in parameters from config file
//---------------------------------------
void readParameters(const std::string &filename, std::string &simulation_base, int &N_bodies, float &softening,
                    float &time_start, float &time_end, float &snap_rate, float &initial_dt,
                    bool &cross_time, float &ETA_cross, float &ETA_acc, float &ETA_vel)
{
    std::ifstream file(filename);
    
    if (file.is_open())
    {
        std::string line;
        
        while (std::getline(file, line))
        {
            std::string value;
            std::getline(file, value);
            
            if (line == "SIMULATION_BASE")
                simulation_base = value;
            else if (line == "N-BODIES")
                N_bodies = std::stoi(value);
            else if (line == "SOFTENING")
                softening = std::stof(value);
            else if (line == "TIME_START")
                time_start = std::stof(value);
            else if (line == "TIME_END")
                time_end = std::stof(value);
            else if (line == "SNAP_RATE")
                snap_rate = std::stof(value);
            else if (line == "INITIAL_DT")
                initial_dt = std::stof(value);
            else if (line == "CROSS_TIME")
            {
                if (value == "true")
                    cross_time = true;
                else if (value == "false")
                    cross_time = false;
                else
                    std::cerr << "Invalid value for CROSS_TIME: " << value << ". Use 'true' or 'false'." << std::endl;
            }
            else if (line == "ETA_CROSS")
                ETA_cross = std::stof(value);
            else if (line == "ETA_ACC")
                ETA_acc = std::stof(value);
            else if (line == "ETA_VEL")
                ETA_vel = std::stof(value);
            else
                std::cerr << "Unknown parameter: " << line << std::endl;
        }
        file.close();
    }
    else
    {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}
//---------------------------------------

// WIP: Write data to snapshot binary file
//---------------------------------------
void writeBinaryData(const std::string& filename, float current_time, float dT,
                     float softening_factor, int N, float4* pos, float4* vel, float4* force)
{
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (file.is_open())
    {
        file.write((char*)&N, sizeof(int));
        file.write((char*)&current_time, sizeof(float));
        file.write((char*)&dT, sizeof(float));
        file.write((char*)&softening_factor, sizeof(float));
        
        for (int orbital = 0; orbital < N; orbital++)
        {
            file.write((char*)&pos[orbital].w, sizeof(float));  // Mass
            file.write((char*)&pos[orbital].x, sizeof(float));  // x position
            file.write((char*)&pos[orbital].y, sizeof(float));  // y position
            file.write((char*)&pos[orbital].z, sizeof(float));  // z position
            file.write((char*)&vel[orbital].x, sizeof(float));  // x velocity
            file.write((char*)&vel[orbital].y, sizeof(float));  // y velocity
            file.write((char*)&vel[orbital].z, sizeof(float));  // z velocity
            
            float xFrc = force[orbital].x * dT;
            float yFrc = force[orbital].y * dT;
            float zFrc = force[orbital].z * dT;
            
            file.write((char*)&xFrc, sizeof(float));    // x force
            file.write((char*)&yFrc, sizeof(float));    // y force
            file.write((char*)&zFrc, sizeof(float));    // z force
        }
        file.close();
    }
    else { std::cerr << "Error opening file: " << filename << std::endl; }
}
//---------------------------------------

// Calculate the crossing time of the system
//---------------------------------------
float calculateCrossingTime(const float4 *vel, int N)
{
    float max_v2 = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float v2 = (vel[i].x * vel[i].x) + (vel[i].y * vel[i].y) + (vel[i].z * vel[i].z);
        if (v2 > max_v2) {
            max_v2 = v2;
        }
        
    }
    //     mean_v2 += (vel[i].x * vel[i].x) + (vel[i].y * vel[i].y) + (vel[i].z * vel[i].z);
    // }
    // mean_v2 /= float(N);
    
    return R_CLUSTER / std::sqrt(max_v2);
}
//---------------------------------------

// Returns the current time in the format yymmddhhmmss
//---------------------------------------
std::string getCurrentTime()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&now_c);
    
    std::stringstream ss;
    ss << std::put_time(tm, "%Y-%m-%d--%H-%M-%S");
    
    return ss.str();
}
//---------------------------------------



// IC generator
//---------------------------------------
void randomiseOrbitals(NBodyICConfig config, float4* pos, float4* vel, int N)
{
    using std::uniform_real_distribution;
    std::default_random_engine gen(SEED); // NOLINT(cert-msc51-cpp)
    float totalMass = 0.0;
    
    switch(config) {
        case NORB_SMALLN_CLUSTER:
        {
            float mu = 0.1;        // mean in base_10 log space, solar masses, m_0
            float sigma = 0.627;   // std. deviations in log 10 space | Chabrier, 2002
            
            // Convert the mean and standard deviation to natural logarithm space
            float ln_mu = mu * std::log(10.f);
            float ln_sigma = sigma * std::log(10.f);
    
            // Generate the lognormal distribution random masses in natural log space
            std::random_device rd;
            std::mt19937_64 rng(rd()); // mersenne-twister 19937 generator with 64-bit output
            std::lognormal_distribution<float> dist(ln_mu, ln_sigma);
    
            //  Max radius of each cluster
            // float radius = 2062.f; //10e4; // AU // 0.01 pc
            float3 cluster_centre = {0.f, 0.f, 0.f};
            float3 filament_offset = {R_CLUSTER * 2, 0.f, 0.f};
            uniform_real_distribution<float> r(-R_CLUSTER/2.f, R_CLUSTER/2.f);
            uniform_real_distribution<float> v(-1.f, 1.f); // -.1 to .1 before scaling
            
            for (int i = 0; i < N; i++)
            {
                // How many clusters? How many stars/cluster?
                // if (i % STARS_PER_CLUSTER == 0 && i > 0) { // generate new cluster
                //     cluster_centre.x += filament_offset.x;
                //     cluster_centre.y += filament_offset.y;
                //     cluster_centre.z += filament_offset.z;
                //     std::cout << "New cluster at: " << cluster_centre.x << ", " << cluster_centre.y << ", " << cluster_centre.z << std::endl;
                // }
                
                // Lognormal Initial Mass Function
                float ln_mass = dist(rng);
                float mass = std::log10(std::exp(ln_mass)); // convert back to base-10 log space
                
                // Randomised positions based on radius
                float px = r(gen);
                float py = r(gen);
                float pz = r(gen);
                
                // std::cout << "Star " << i << " at: " << px << ", " << py << ", " << pz << std::endl;
                
                // Randomised velocities
                float vx = v(gen);
                float vy = v(gen);
                float vz = v(gen);
                
                // Assign pos, vel, mass
                pos[i] = make_float4(px, py, pz, mass);
                vel[i] = make_float4(vx, vy, vz, mass);
                
                totalMass += mass;
            }
            
            // Loop through each cluster
            for (int cluster = 0; cluster < (N / STARS_PER_CLUSTER); cluster++) {
                std::cout << "Cluster " << cluster << " N: " << N << std::endl;
                int start_idx = cluster * STARS_PER_CLUSTER;
                int end_idx = start_idx + STARS_PER_CLUSTER;
    
    
                // Create a temporary vector to store positions and velocities of the cluster
                // float4 cluster_pos(STARS_PER_CLUSTER);
                float4 cluster_pos[STARS_PER_CLUSTER];
                // std::vector<float4> cluster_vel(STARS_PER_CLUSTER);
                float4 cluster_vel[STARS_PER_CLUSTER];
                
                // TODO: extract this into a function
                // Fill the temporary vector with the positions and velocities of the stars in the cluster
                for (int i = 0; i < STARS_PER_CLUSTER; i++) {
                    cluster_pos[i] = pos[start_idx + i];
                    cluster_vel[i] = vel[start_idx + i];
                    // std::cout << "StarCLUSTER " << i << " at: " << cluster_pos[i].x << ", " << cluster_pos[i].y << ", " << cluster_pos[i].z << std::endl;
                }
    
                // Apply centre of mass correction for the cluster
                float4 centreOfMassPos = calculateCentreOfMass(cluster_pos, STARS_PER_CLUSTER);
                float4 centreOfMassVel = calculateCentreOfMass(cluster_vel, STARS_PER_CLUSTER);
                for (int i = start_idx; i < end_idx; i++) {
                    // std::cout << "StarCOM " << i << " at: " << pos[i].x << ", " << pos[i].y << ", " << pos[i].z << std::endl;
                    pos[i].x -= centreOfMassPos.x;
                    pos[i].y -= centreOfMassPos.y;
                    pos[i].z -= centreOfMassPos.z;
                    vel[i].x -= centreOfMassVel.x;
                    vel[i].y -= centreOfMassVel.y;
                    vel[i].z -= centreOfMassVel.z;
                    // std::cout << "Star " << i << " at: " << pos[i].x << ", " << pos[i].y << ", " << pos[i].z << std::endl;
                    pos[i].x += filament_offset.x * float(cluster);
                    pos[i].y += filament_offset.y * float(cluster);
                    pos[i].z += filament_offset.z * float(cluster);
                    // std::cout << "StarCOM " << i << " at: " << pos[i].x << ", " << pos[i].y << ", " << pos[i].z << std::endl;
                }
                
                for (int i = 0; i < STARS_PER_CLUSTER; i++) {
                    cluster_pos[i] = pos[start_idx + i];
                    cluster_vel[i] = vel[start_idx + i];
                }
    
                // Scale the velocities to the virial theorem
                float gravitationalEnergy = -calculateGravitationalEnergy(cluster_pos, STARS_PER_CLUSTER); // W = -U
                float kineticEnergy = calculateKineticEnergy(cluster_vel, STARS_PER_CLUSTER);               // K  E = K + W | VIR = W/2 = K
    
                // float scalingFactor = sqrtf(ALPHA_VIR * gravitationalEnergy / kineticEnergy) ;
                float virialRatio = -kineticEnergy / gravitationalEnergy;
                float scalingFactor = sqrtf(.5f / virialRatio);
    
                for (int i = start_idx; i < end_idx; i++) {
                    vel[i].x *= scalingFactor;
                    vel[i].y *= scalingFactor;
                    vel[i].z *= scalingFactor;
                }
    
                for (int i = 0; i < STARS_PER_CLUSTER; i++) {
                    cluster_pos[i] = pos[start_idx + i];
                    cluster_vel[i] = vel[start_idx + i];
                }
                
                float kineticEnergyScaled = calculateKineticEnergy(cluster_vel, STARS_PER_CLUSTER);
                float verifyVirial = -kineticEnergyScaled / gravitationalEnergy;
    
                // std::cout << "Centre of mass [pos]: " << centreOfMassPos.x << ", " << centreOfMassPos.y << ", "
                //           << centreOfMassPos.z << std::endl;
                // std::cout << "Scaling factor: " << scalingFactor << std::endl;
                // std::cout << "Gravitational energy: " << gravitationalEnergy << std::endl;
                // std::cout << "Kinetic energy (unscaled): " << kineticEnergy << std::endl;
                // std::cout << "Kinetic energy (scaled): " << kineticEnergyScaled << std::endl;
                // std::cout << "Virial Ratio (initial): " << virialRatio << std::endl;
                // std::cout << "Virial Ratio (should == 0.5?): " << verifyVirial << std::endl;
            }
        }
            break;
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
                pos[i].x = randXPos(gen);
                pos[i].y = randYPos(gen);
                pos[i].z = randHeight(gen);
                pos[i].w = massClamped;
    
                // random velocity assignment
                float r = sqrtf(pos[i].x * pos[i].x + pos[i].y * pos[i].y + pos[i].z * pos[i].z);
                vel[i].x = randVel(gen) * (r / pos[i].x);//0.001f;
                vel[i].y = randVel(gen) * (r / pos[i].y);//0.001f;
                vel[i].z = 0.0f;
                vel[i].w = pos[i].w;
        
                totalMass += pos[i].w;
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
    
            pos[0].x = 0.0;
            pos[0].y = 0.0;
            pos[0].z = 0.0;
            pos[0].w = CENTRE_STAR_M;
    
            vel[0].x = 0.0;
            vel[0].y = 0.0;
            vel[0].z = 0.0;
            vel[0].w = CENTRE_STAR_M;
    
    
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
    
                pos[i].x = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
                pos[i].y = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
                pos[i].z = point.x * (inner + (outer - inner) * randF(gen) / (float) RAND_MAX);
                pos[i].w = randMass(gen);
                
        
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
                float3 vv = {pos[i].x, pos[i].y, pos[i].z};
                vv = cross(vv, axis);
                vel[i].x = vv.x * vScale;
                vel[i].y = vv.y * vScale;
                vel[i].z = vv.z * vScale;
                vel[i].w = pos[i].w;
        
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
    
                pos[i].x = point.x * scale;
                pos[i].y = point.y * scale;
                pos[i].z = point.z * scale;
                pos[i].w = randMass(gen);
                vel[i].x = point.x * vScale; //* float(PI)/180 * lengthSq;
                vel[i].y = point.y * vScale;
                vel[i].z = point.z * vScale;
                vel[i].w = pos[i].w;
                
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
            
            pos[0].x = 0.0;
            pos[0].y = 0.0;
            pos[0].z = 0.0;
            pos[0].w = ADVD_CENTRE_M;
    
            vel[0].x = 0.0;
            vel[0].y = 0.0;
            vel[0].z = 0.0;
            vel[0].w = ADVD_CENTRE_M;
    
            float c      = ADVD_C_INNER; // flatness
            float mass   = randMassInner(gen);
            // float mass = randMassInner(gen);
            float radius = ADVD_R_INNER;
            
            int start;
            if (glxyCollision)
            {
                pos[1].x = 1000.0;
                pos[1].y = 500.0;
                pos[1].z = -10000.0;
                pos[1].w = ADVD_G2_MASS;
    
                vel[1].x = -0.1;
                vel[1].y = 0.0;
                vel[1].z = 1.0;
                vel[1].w = ADVD_G2_MASS;
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
                
                float3 position;
                while (true)
                {
                    position.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    position.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    
                    if (position.y >= -1.0f * sqrtf(1.0f - powf(position.x, 2.0f))
                        && position.y <= sqrtf(1.0f - powf(position.x, 2.0f)))
                        break;
                }
                
                float zPosMax = sqrtf(c * (1.0f - powf(position.x, 2.0f)
                        -powf(position.y, 2.0f)));
                float zPosMin = -1.0f * zPosMax;
                float zPosRand = rand() / (float) RAND_MAX;
                position.z = (zPosMax - zPosMin) * zPosRand + zPosMin;
    
                position.x *= radius;
                position.y *= radius;
                position.z *= radius;
                
                float m = position.y / position.x;
                m = -1.0f / m;
                float b = position.y - position.x * m;
                
                float3 velocity;
                // float vel_m = sqrtf(((float)BIG_G * (1e6f + mass * 1.2e-6f)) /
                //                     sqrtf(position.x*position.x + position.y*position.y + position.z*position.z));
                float vel_m = sqrtf(((float)BIG_G * (ADVD_CENTRE_M + mass * 1.2e2f)) /
                                    sqrtf(position.x * position.x + position.y * position.y + position.z * position.z));
                
                if (position.y > 0)
                {
                    velocity = {-1.0f * (radius / 2.0f), (position.x - radius / 2.0f) * m + b - position.y, 0};
                    vel_m /= sqrtf(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
                    velocity.x *= vel_m;
                    velocity.y *= vel_m;
                    velocity.z *= vel_m;
                    
                }
                else
                {
                    velocity = {(radius / 2.0f), (position.x + radius / 2.0f) * m + b - position.y, 0};
                    vel_m /= sqrtf(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
                    velocity.x *= vel_m;
                    velocity.y *= vel_m;
                    velocity.z *= vel_m;
                }
                
                float pScale = 1.0f;
                pos[i].x = position.x * pScale;
                pos[i].y = position.y * pScale;
                pos[i].z = position.z * pScale;
                pos[i].w = mass;
    
                vel[i].x = velocity.x;
                vel[i].y = velocity.y;
                vel[i].z = velocity.z;
                vel[i].w = mass;
                
                // std::cout << "\n " << velocity[i].x << " " << velocity[i].y << " " << velocity[i].z;
    
            }
        }
            break;
        case NORB_CONFIG_ADV_DISK_COLLSION:
        {
            // hi
            uniform_real_distribution<float> randMassInner(ADVD_M_INNER_MIN, ADVD_M_INNER_MAX);
    
            pos[0].x = 0.0;
            pos[0].y = 0.0;
            pos[0].z = 0.0;
            pos[0].w = ADVD_CENTRE_M;
    
            vel[0].x = 0.0;
            vel[0].y = 0.0;
            vel[0].z = 0.0;
            vel[0].w = ADVD_CENTRE_M;
    
            pos[N / 2].x = ADVD_G2_X;
            pos[N / 2].y = ADVD_G2_Y;
            pos[N / 2].z = ADVD_G2_Z;
            pos[N / 2].w = ADVD_CENTRE_M;
    
            vel[N / 2].x = -1.f * ADVD_G2_VX;
            vel[N / 2].y = -1.f * ADVD_G2_VY;
            vel[N / 2].z = -1.f * ADVD_G2_VZ;
            vel[N / 2].w = ADVD_CENTRE_M;
    
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
        
                float3 position;
                while (true)
                {
                    position.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    position.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            
                    if (position.y >= -1.0f * sqrtf(1.0f - powf(position.x, 2.0f))
                        && position.y <= sqrtf(1.0f - powf(position.x, 2.0f)))
                        break;
                }
        
                float zPosMax = sqrtf(c * (1.0f - powf(position.x, 2.0f)
                                           -powf(position.y, 2.0f)));
                float zPosMin = -1.0f * zPosMax;
                float zPosRand = rand() / (float) RAND_MAX;
                position.z = (zPosMax - zPosMin) * zPosRand + zPosMin;
    
                position.x *= radius;
                position.y *= radius;
                position.z *= radius;
        
                float m = position.y / position.x;
                m = -1.0f / m;
                float b = position.y - position.x * m;
        
                float3 velocity;
                // float vel_m = sqrtf(((float)BIG_G * (1e6f + mass * 1.2e-6f)) /
                //                     sqrtf(position.x*position.x + position.y*position.y + position.z*position.z));
                float vel_m = sqrtf(((float)BIG_G * (ADVD_CENTRE_M + mass * 1.2e2f)) /
                                    sqrtf(position.x * position.x + position.y * position.y + position.z * position.z));
        
                if (position.y > 0)
                {
                    velocity = {-1.0f * (radius / 2.0f), (position.x - radius / 2.0f) * m + b - position.y, 0};
                    vel_m /= sqrtf(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
                    velocity.x *= vel_m;
                    velocity.y *= vel_m;
                    velocity.z *= vel_m;
            
                }
                else
                {
                    velocity = {(radius / 2.0f), (position.x + radius / 2.0f) * m + b - position.y, 0};
                    vel_m /= sqrtf(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
                    velocity.x *= vel_m;
                    velocity.y *= vel_m;
                    velocity.z *= vel_m;
                }
        
                float pScale = 1.0f;
                pos[i].x = position.x * pScale;
                pos[i].y = position.y * pScale;
                pos[i].z = position.z * pScale;
                pos[i].w = mass;
    
                vel[i].x = velocity.x;
                vel[i].y = velocity.y;
                vel[i].z = velocity.z;
                vel[i].w = mass;
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
        
                float3 position;
                while (true)
                {
                    position.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
                    position.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            
                    if (position.y >= -1.0f * sqrtf(1.0f - powf(position.x, 2.0f))
                        && position.y <= sqrtf(1.0f - powf(position.x, 2.0f)))
                        break;
                }
        
                float zPosMax = sqrtf(c * (1.0f - powf(position.x, 2.0f)
                                           -powf(position.y, 2.0f)));
                float zPosMin = -1.0f * zPosMax;
                float zPosRand = rand() / (float) RAND_MAX;
                position.z = (zPosMax - zPosMin) * zPosRand + zPosMin;
    
                position.x *= radius;
                position.y *= radius;
                position.z *= radius;
        
                float m = position.y / position.x;
                m = -1.0f / m;
                float b = position.y - position.x * m;
        
                float3 velocity;
                // float vel_m = sqrtf(((float)BIG_G * (1e6f + mass * 1.2e-6f)) /
                //                     sqrtf(position.x*position.x + position.y*position.y + position.z*position.z));
                float vel_m = sqrtf(((float)BIG_G * (ADVD_CENTRE_M + mass * 1.2e2f)) /
                                    sqrtf(position.x * position.x + position.y * position.y + position.z * position.z));
        
                if (position.y > 0)
                {
                    velocity = {-1.0f * (radius / 2.0f), (position.x - radius / 2.0f) * m + b - position.y, 0};
                    vel_m /= sqrtf(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
                    velocity.x *= vel_m;
                    velocity.y *= vel_m;
                    velocity.z *= vel_m;
            
                }
                else
                {
                    velocity = {(radius / 2.0f), (position.x + radius / 2.0f) * m + b - position.y, 0};
                    vel_m /= sqrtf(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
                    velocity.x *= vel_m;
                    velocity.y *= vel_m;
                    velocity.z *= vel_m;
                }
    
    
                pos[i].x = position.z + (float)ADVD_G2_X;
                pos[i].y = position.y + (float)ADVD_G2_Y;
                pos[i].z = position.x + (float)ADVD_G2_Z;
                pos[i].w = mass;
    
                vel[i].x = velocity.z - (float)ADVD_G2_VX;
                vel[i].y = velocity.y - (float)ADVD_G2_VY;
                vel[i].z = velocity.x - (float)ADVD_G2_VZ;
                vel[i].w = mass;
            }
            
        }
            break;
        case NORB_CONFIG_SOLAR:
        {
            int i = 0;
            // The Sun
            pos[i].x = pos[i].y = pos[i].z = 0.f;
            pos[i].w = 1.f;
    
            vel[i].x = vel[i].y = vel[i].z = 0.f;
            vel[i].w = 1.f;
    
            // Earth
            pos[++i].x = 1.f;
            pos[i].y = 0.f;
            pos[i].z = 0.f;
            pos[i].w = 3.00273e-6f;// 2.9861e-6f;
    
            vel[i].x = 0.f;
            vel[i].y = 29.795f / KMS_TO_AUD;//29.78f / (float)KMS_TO_AUD;
            vel[i].z = 0.f;
            vel[i].w = 3.00273e-6f;
    
            // // Mercury
            // pos[++i] = {.387f, 0.f, 0.f, 1.651e-7f};
            // vel[i]   = {0.f, 47.36f/KMS_TO_AUD, 0.f, 1.651e-7f};
            //
            // // Venus
            // pos[++i].x = 0.723f;
            // pos[i].y = 0.f;
            // pos[i].z = 0.f;
            // pos[i].w = 2.447e-6f;
            //
            // vel[i].x = 0.f;
            // vel[i].y = 35.02f / KMS_TO_AUD;
            // vel[i].z = 0.f;
            // vel[i].w = 2.447e-6f;
            //
            // // Mars
            // pos[++i] = {1.524f, 0.f, 0.f, 3.213e-7f};
            // vel[i]   = {0.f, 24.07f/KMS_TO_AUD, 0.f, 3.213e-7f};
            
        }
            break;
    }
    std::cout << "\nTOTAL MASS ->> " << totalMass;
}
//---------------------------------------

// Calculate Centre of Mass
//---------------------------------------
float4 calculateCentreOfMass(float4* body, int N)
{
    float4 centreOfMass = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < N; i++)
    {
        centreOfMass.x += body[i].x * body[i].w;
        centreOfMass.y += body[i].y * body[i].w;
        centreOfMass.z += body[i].z * body[i].w;
        centreOfMass.w += body[i].w;
    }
    centreOfMass.x /= centreOfMass.w;
    centreOfMass.y /= centreOfMass.w;
    centreOfMass.z /= centreOfMass.w;
    return centreOfMass;
}
//---------------------------------------


// Calculate Gravitational Energy
//---------------------------------------
float calculateGravitationalEnergy(float4* pos, int N)
{
    float gravitationalEnergy = 0.0f;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i==j)
                continue;
            
            float3 r;
            r.x = pos[j].x - pos[i].x;
            r.y = pos[j].y - pos[i].y;
            r.z = pos[j].z - pos[i].z;
            
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
            
            gravitationalEnergy += float(BIG_G) * pos[i].w * pos[j].w / sqrtf(distSqr);
            // std::cout << "Distance Sqr ->> " << distSqr << std::endl;
            // std::cout << "Distance Sqrt ->> " << sqrtf(distSqr) << std::endl;
            // std::cout << "Gravitational Energy ->> " << gravitationalEnergy << std::endl;
        }
    }
    return gravitationalEnergy * .5f;   // .5f because each pairwise interaction is counted twice
}
//---------------------------------------


// Calculate Kinetic Energy
//---------------------------------------
float calculateKineticEnergy(float4* vel, int N)
{
    float kineticEnergy = 0.0f;
    for (int i = 0; i < N; i++)
    {
        kineticEnergy += 0.5f * vel[i].w * (vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z);
    }
    return kineticEnergy;
}
//---------------------------------------



// Initialise Forces [typically unnecessary unless for solar-system]
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


// Calculate variable time-step
//---------------------------------------
float calculateTimeStep(float4* pos, float4* vel, float4* force, float curDT, int N)
{
    auto* acc_dot = new float3[N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == j) continue;
            
            float3 r; float3 v;
            // r_ij -> AU [distance]
            r.x = pos[j].x - pos[i].x;
            r.y = pos[j].y - pos[i].y;
            r.z = pos[j].z - pos[i].z;
            
            // v_ij -> AU/day [velocity]
            v.x = vel[j].x - vel[i].x;
            v.y = vel[j].y - vel[i].y;
            v.z = vel[j].z - vel[i].z;
            
            // distance squared == dot(r_ij, r_ij) + softening^2
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
            float dist = sqrtf(distSqr);
            distSqr += SOFTENING * SOFTENING;
            
            // inverse distance cubed == 1 / distSqr^(3/2) [fastest method]
            float distSixth = distSqr * distSqr * distSqr;
            float invDistCube = 1.0f / sqrtf(distSixth);
            float distFifth = distSqr * distSqr * dist;
            float invDistFifth = 1.0f / sqrtf(distFifth);
            
            // force = mass_j * inverse distance cube
            float f1 = pos[j].w * invDistCube;
            float f2 = pos[j].w * invDistFifth;
            
            float vdotr = dot(v, r);
            
            // acceleration
            // acc_dot[i].x += ( (v.x * f1) + ((3 * v.x * r.x) * r.x * f2) ) * (float)BIG_G;
            acc_dot[i].x += ( (v.x * f1) + ((3 * vdotr) * r.x * f2) ) * (float)BIG_G;
            acc_dot[i].y += ( (v.y * f1) + ((3 * vdotr) * r.y * f2) ) * (float)BIG_G;
            acc_dot[i].z += ( (v.z * f1) + ((3 * vdotr) * r.z * f2) ) * (float)BIG_G;
        }
    }
    
    float max_aa_dot = 0.0f;
    float max_a = 0;
    float max_v = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float a_dot = acc_dot[i].x * acc_dot[i].x + acc_dot[i].y * acc_dot[i].y + acc_dot[i].z * acc_dot[i].z;
        float a = force[i].x * force[i].x + force[i].y * force[i].y + force[i].z * force[i].z;
        float v = vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z;
        a_dot = sqrtf(a_dot);
        a = sqrtf(a);
        v = sqrtf(v);
        
        float aa_dot = a / a_dot;
        max_aa_dot = (aa_dot > max_aa_dot) ? aa_dot : max_aa_dot;
        max_a = (a > max_a) ? a : max_a;
        max_v = (v > max_v) ? v : max_v;
    }
    float dt = ETA_ACC * max_aa_dot + ETA_VEL * (max_v / max_a);
    return fminf(dt, MAX_DELTA_TIME);
}

//---------------------------------------

// float calculateTimeStep(float4* pos, float4* vel, float4* force, float curDT, int N)
// {
//     float maxAcc = 0.0f;
//     int maxAccIndex = 0;
//     float maxVel = 0.0f;
//     for (int i = 0; i < N; i++)
//     {
//         // Acceleration
//         float accel = force[i].x * force[i].x + force[i].y * force[i].y + force[i].z * force[i].z;
//         accel /= pos[i].w;
//         float accelMag = sqrtf(accel);
//
//         maxAccIndex = (accelMag > maxAcc) ? i : maxAccIndex;
//         maxAcc = (accelMag > maxAcc) ? accelMag : maxAcc;
//
//         // Velocity
//         float velocity = vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z;
//         float velMag = sqrtf(velocity);
//
//         maxVel = (velMag > maxVel) ? velMag : maxVel;
//     }
//     float m = pos[maxAccIndex].w;
//     float3 a = make_float3(force[maxAccIndex].x / m,
//                            force[maxAccIndex].y / m,
//                            force[maxAccIndex].z / m);
//     float aDot = dot(a, a);
//     float dt = ETA_ACC * maxAcc / aDot + ETA_VEL * maxVel / curDT * maxAcc;
//     std::cout << "\nDelta Time ->> " << dt << std::endl;
//     return fminf(dt, MAX_DELTA_TIME);
// }

//---------------------------------------
// MAIN UPDATE LOOP
//---------------------------------------
void simulate(float4* m_hPos, float4* m_dPos[2],
              float4* m_hVel, float4* m_dVel[2],
              float4* m_hForce, float4* m_dForce[2],
              uint m_currentRead, uint m_currentWrite,
              float& m_hDeltaTime, float* m_dDeltaTime[2], int N, uint m_p, uint m_q)
{
    // set pos,vel -> update -> get pos,vel ~@ repeat
    
    // Send data to device first
    copyDataToDevice(m_dPos[m_currentRead], m_hPos, N);
    copyDataToDevice(m_dVel[m_currentRead], m_hVel, N);
    copyDataToDevice(m_dForce[m_currentRead], m_hForce, N);
    cudaMemcpy(m_dDeltaTime[m_currentRead], &m_hDeltaTime, sizeof(float), cudaMemcpyHostToDevice);
    getCUDAError();


    // Do the thing
    deployToGPU(m_dPos[m_currentRead], m_dPos[m_currentWrite],
                m_dVel[m_currentRead], m_dVel[m_currentWrite],
                m_dForce[m_currentRead], m_dForce[m_currentWrite],
                m_dDeltaTime[m_currentRead], m_dDeltaTime[m_currentWrite], N, m_p, m_q);
    // Swap read and write
    std::swap(m_currentRead, m_currentWrite);

    cudaDeviceSynchronize();
    
    // Retrieve data from device
    copyDataToHost(m_hPos, m_dPos[m_currentRead], N);
    copyDataToHost(m_hVel, m_dVel[m_currentRead], N);
    copyDataToHost(m_hForce, m_dForce[m_currentRead], N);
    cudaMemcpy(&m_hDeltaTime, m_dDeltaTime[m_currentRead], sizeof(float), cudaMemcpyDeviceToHost);
    getCUDAError();
    // std::cout << "Time step: " << m_hDeltaTime << std::endl;
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
              float4* m_hForce, float4* m_dForce[2],
              float* m_dDeltaTime[2])
{
    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hForce;
    // delete m_hDeltaTime;
    
    deleteNOrbitalArrays(m_dPos, m_dVel, m_dForce, m_dDeltaTime);
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

// Deletes snapshot folder files if already exists
//---------------------------------------
void deleteFilesInDirectory(const std::string& directory_path)
{
    try
    {
        for (const auto &entry : std::filesystem::directory_iterator(directory_path))
        {
            std::filesystem::remove(entry.path());
        }
    }
    catch (std::filesystem::filesystem_error &e)
    {
        std::cerr << "Error deleting files in directory: " << e.what() << std::endl;
    }
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


// Send eta_acc value to device
//---------------------------------------
void setDeviceEtaAcc(float eta)
{
    cudaMemcpyToSymbol(eta_acc, &eta, sizeof(float),0);
}

// Send eta_vel value to device
//---------------------------------------
void setDeviceEtaVel(float eta)
{
    cudaMemcpyToSymbol(eta_vel, &eta, sizeof(float),0);
}


// Allocate device memory for variables
//---------------------------------------
void allocateNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2], float* dT[2],  int N)
{
    // memory size for device allocation
    uint memSize = sizeof(float4) * N;
    
    cudaMalloc((void**)&pos[0], memSize);
    cudaMalloc((void**)&pos[1], memSize);
    cudaMalloc((void**)&vel[0], memSize);
    cudaMalloc((void**)&vel[1], memSize);
    cudaMalloc((void**)&force[0], memSize);
    cudaMalloc((void**)&force[1], memSize);
    cudaMalloc((void**)&dT[0], sizeof(float));
    cudaMalloc((void**)&dT[1], sizeof(float));
}
//---------------------------------------


// De-allocate device memory variables
//---------------------------------------
void deleteNOrbitalArrays(float4* pos[2], float4* vel[2], float4* force[2], float* dT[2])
{
    cudaFree((void**)pos[0]);
    cudaFree((void**)pos[1]);
    cudaFree((void**)vel[0]);
    cudaFree((void**)vel[1]);
    cudaFree((void**)force[0]);
    cudaFree((void**)force[1]);
    cudaFree((void**)dT[0]);
    cudaFree((void**)dT[1]);
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
                 float* oldDT, float* newDT, int N, uint p, uint q)
{
    uint shMemSize = p * q * sizeof(float4);
    
    // thread and grid time :D
    dim3 threads(p, q, 1);
    dim3 grid(N / p, 1, 1);
    
    // DEPLOY TODO: removed feature
    /*If multithreading is enabled (i.e. q>1 | multiple threads per
     * body) then the more complicated code is executed (using bool template
     * over in the kernel), and if it is not, then the simpler code is executed*/

    switch(integrator)
    {
        case LEAPFROG_VERLET:
        default:
        {
            integrateNOrbitals<<<grid, threads, shMemSize
            >>>(oldPos, newPos, oldVel, newVel, oldForce, newForce, oldDT, newDT, N);
        }
        break;
        case KICK_DRIFT_VERLET:
        {
            initHalfKickForces<<<grid, threads, shMemSize
            >>>(oldPos, newPos, oldVel, newVel, oldForce, newForce, oldDT, N);
            cudaDeviceSynchronize();
            fullKickForces<<<grid, threads, shMemSize
            >>>(oldPos, newPos, oldVel, newVel, oldForce, newForce, oldDT, N);
        }
        break;
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
                  << "For N == " << N_orbitals << " || Iterations == " << ITERATIONS << std::endl;
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
    // if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    //     shiftSpeed = 1 * CTRL_FACTOR;
    // if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_RELEASE)
    //     shiftSpeed = 1;
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
    // if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
    //     timestep -= 0.1f ;
    // if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS)
    //     timestep += 0.1f;
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
    // CTRL TO SLOW THINGS DOWN
    if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS)
        shiftSpeed = 1 * CTRL_FACTOR;
    // COMMA/PERIOD FOR TIMESTEP
    // if (key == GLFW_KEY_COMMA && action == GLFW_PRESS)
    //     m_hDeltaTime -= 0.25f;
    // if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS)
    //     m_hDeltaTime += 10000.25f;
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



