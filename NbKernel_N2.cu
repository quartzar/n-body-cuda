//
// Created by quartzar on 24/10/22.
//

#ifndef ORBITERV6_NBKERNEL_N2_H
#define ORBITERV6_NBKERNEL_N2_H

#include <cmath>
#include "NbKernel_N2.cuh"

/* This is a clamping parameter to limit the max velocity of a body.*/
#define LIGHT_SPEED 173.265 // theoretical max V of stars in our galaxy
// 173.265 // speed of light c ->> AU/day
#define LOOP_UNROLL 1

__constant__ float softeningSqr;
__constant__ float big_G;
__constant__ float eta_acc;
__constant__ float eta_vel;

// Simplified shared memory (refer to CUDA programming guide 2007)
/* All this essentially does is take the index provided by
 * using the macro in computeOrbitalForces, along with the
 * sharedPos[] array that is exposed with 'extern __shared__'
 * in the same function, to dynamically calculate the shared
 * memory addressing every time the kernel is called.
 * This is useful, as it means all threads can access the data
 * independant of one another, along with being able to adjust it
 * based upon the size of N, and the configuration of p & q. */
#define SH_M(i) shPositions[i + blockDim.x * threadIdx.y]
/* If multithreading is enabled, this macro is used instead in order
 * to allocate the right num. of threads per body per tile calculation*/
#define SH_M_SUM(i, j) shPositions[i + blockDim.x * j]


//============================================================//
__global__ void
initHalfKickForces(float4* oldPos, float4* newPos,
                   float4* oldVel, float4* newVel,
                   float4* oldForce, float4* newForce,
                   float *deltaTime, int N)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 curPos = oldPos[id];
    
    float3 force = computeOrbitalForces(curPos, oldPos, N);
    
    float4 curVel = oldVel[id];
    
    // Update half-kick velocities
    curVel.x += force.x * (*deltaTime) * 0.5f;
    curVel.y += force.y * (*deltaTime) * 0.5f;
    curVel.z += force.z * (*deltaTime) * 0.5f;
    
    // Update positions [drift]
    curPos.x += curVel.x * (*deltaTime);
    curPos.y += curVel.y * (*deltaTime);
    curPos.z += curVel.z * (*deltaTime);
    
    newPos[id] = curPos;
    newVel[id] = curVel;
}
//============================================================//


//============================================================//
__global__ void
fullKickForces(float4* oldPos, float4* newPos,
               float4* oldVel, float4* newVel,
               float4* oldForce, float4* newForce,
               float *deltaTime, int N)
{
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 curPos = newPos[id];
    
    float3 force = computeOrbitalForces(curPos, newPos, N);
    
    float4 curVel = newVel[id];
    
    // Update half-kick velocities
    curVel.x += force.x * (*deltaTime) * 0.5f;
    curVel.y += force.y * (*deltaTime) * 0.5f;
    curVel.z += force.z * (*deltaTime) * 0.5f;
    
    newPos[id] = curPos;
    newVel[id] = curVel;
    newForce[id].x = force.x;
    newForce[id].y = force.y;
    newForce[id].z = force.z;
}
//============================================================//


//============================================================//
__global__ void
integrateNOrbitals(float4* oldPos, float4* newPos,
                   float4* oldVel, float4* newVel,
                   float4* oldForce, float4* newForce,
                   float* oldDT, float* newDT, int N)
/*   where:  */
// |-> new[Pos/Vel] is calculated here, in kernel/device,
// |-> old[Pos/Vel] is previous calculation, held on host,
// |-> deltaTime is the current time-step/iteration,
// |-> N_orbitals is just N-bodies (moving away from using #DEFINE's)
{
    // fused-multiply-add [previously __mul24(a,b)+c]
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 curPos = oldPos[idx];
    
    float3 force = computeOrbitalForces(curPos, oldPos, N);
    
    float4 curVel = oldVel[idx];
    
    float deltaTime = (*oldDT);
    if (idx == 0) (*newDT) = (*oldDT);
    
    
    // Leapfrog-Verlet integrator (1967)
    
    // v(t + dt/2) = v(t - dt/2) + dt * a(t)
    curVel.x += force.x * deltaTime;
    curVel.y += force.y * deltaTime;
    curVel.z += force.z * deltaTime;
    
    // Clamping to speed of light
    curVel.x = max(-1.f * (float)LIGHT_SPEED, min(curVel.x, (float)LIGHT_SPEED));
    curVel.y = max(-1.f * (float)LIGHT_SPEED, min(curVel.y, (float)LIGHT_SPEED));
    curVel.z = max(-1.f * (float)LIGHT_SPEED, min(curVel.z, (float)LIGHT_SPEED));
    
    // r(t + dt) = r(t) + dt * v(t + dt/2)
    curPos.x += curVel.x * deltaTime;
    curPos.y += curVel.y * deltaTime;
    curPos.z += curVel.z * deltaTime;
    
    // *newDT = *oldDT * 1000.f;
    // newDT[id] = *oldDT * 2.f;
    // (*newDT) = (*oldDT) + 0.01f;
    // complete by storing updated position and velocity
    newPos[idx] = curPos;
    newVel[idx] = curVel;
    // store force as well
    newForce[idx].x = force.x;
    newForce[idx].y = force.y;
    newForce[idx].z = force.z;
    
}

//============================================================//


//============================================================//
__device__ float3
orbOrbInteraction(float4 oi, float4 oj, float3 ai)
{
    float3 r;
    
    // r_ij -> AU [distance]
    r.x = oi.x - oj.x;
    r.y = oi.y - oj.y;
    r.z = oi.z - oj.z;
    
    // distance squared == dot(r_ij, r_ij) + softening^2
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSqr;
    
    // inverse distance cubed == 1 / distSqr^(3/2) [fastest method]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    
    // force = mass_j * inverse distance cube
    float force = oi.w * invDistCube;
    
    // acceleration = acceleration_i + force * r_ij
    ai.x += r.x * force * big_G;
    ai.y += r.y * force * big_G;
    ai.z += r.z * force * big_G;
    
    return ai;
}
//============================================================//


//============================================================//
__device__ float3
tileCalculation(float4 orbPos, float3 force)
{
    /* Declaring this as external & shared means that all threads are able
     * to access it, and it makes me happy as they all get along and I don't
     * have to write more complicated shared memory access :D
     * The compiler will allocate this in shared memory space. By using the
     * shared memory, memory access times are about as fast as a register
     * when there are no bank conflicts. It can be accessed by any thread of the
     * block from which it was created, and has the lifetime of the block*/
    extern __shared__ float4 shPositions[];
    
    /* A technique called loop unrolling is employed here to further increase
     * maximum performance. This works by replacing 1 body-body interaction call
     * per iteration with from 2-32 calls, reducing the loop overhead. It just speeds
     * up the loop basically. Each line is called until LOOP_UNROLL is either at max,
     * or it is set below the amount of unrolls below. 1->2->4->8->16->32.
     * Currently experimental.*/
    int i;
    for (i = 0; i < blockDim.x;)
    {
        force = orbOrbInteraction(  SH_M(i),orbPos, force); i++; // always called
#if LOOP_UNROLL > 1
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
#endif
#if LOOP_UNROLL > 2
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
#endif
#if LOOP_UNROLL > 4
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
#endif
#if LOOP_UNROLL > 8
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;//
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;//
#endif
#if LOOP_UNROLL > 16
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;//
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;//
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;//
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;
        force = orbOrbInteraction(SH_M(i), orbPos, force); i++;//
#endif
    }
    return force;
}
//============================================================//


// WRAP ->> this is a simple macro used to determine if the current block
/* is about to try and access the same memory location as one already is,
 * and will force it to move onto another chunk. This stops the performance
 * impacting effect of multiple multiprocessors all trying to read the same
 * memory location simultaneously. Plus, it sounds cool */
#define WRAP(xtile, xgrid) (((xtile) < xgrid) ? (xtile) : (xtile - xgrid))


//============================================================//
__device__ float3
computeOrbitalForces(float4 orbPos, float4* positions, int N)
{
    /* Declaring this as external & shared means that all threads are able
     * to access it, and it makes me happy as they all get along and I don't
     * have to write more complicated shared memory access :D
     * The compiler will allocate this in shared memory space*/
    extern __shared__ float4 shPositions[];
    
    float3 force = {0.0f, 0.0f, 0.0f};
    
    /* Here we calculate the size of each tile, where the grid
     * of said tile is N/p, each thread in the TODO: finish explanation*/
    uint p = blockDim.x;
    uint q = blockDim.y;
    uint n = N;
    uint begin = n / q * threadIdx.y;
    uint tile0 = begin / (n / q);
    uint tile = tile0;
    uint end = begin + n / q;
    
    for (uint i = begin; i < end; i += p, tile++)
    {
        // assign shared positions for each tile
        shPositions[threadIdx.x + blockDim.x * threadIdx.y] =
            positions[WRAP(blockIdx.x + tile, gridDim.x) * blockDim.x + threadIdx.x];
        
        // synchronize the local threads writing to the local memory cache
        __syncthreads();
        
        // begin>:) [and synchronise again]
        force = tileCalculation(orbPos, force);
        __syncthreads();
    }
    
    return force;
}
//============================================================//

// float clamp(float* a)
// {
//
// }


#endif // ORBITERV6_NBKERNEL_N2_H

// /* LEAPFROG-VERLET 3-STEP INTEGRATOR */
// // I haven't a clue if this will work here
// // firstly -> update just positions with old velocities
// curPos.x += curVel.x * (deltaTime * 0.5f);
// curPos.y += curVel.y * (deltaTime * 0.5f);
// curPos.z += curVel.z * (deltaTime * 0.5f);
//
// // secondly -> update velocities with new force
// curVel.x += force.x * deltaTime;
// curVel.y += force.y * deltaTime;
// curVel.z += force.z * deltaTime;
//
// // finally -> update positions with new velocities
// curPos.x += curVel.x * (deltaTime * 0.5f);
// curPos.y += curVel.y * (deltaTime * 0.5f);
// curPos.z += curVel.z * (deltaTime * 0.5f);