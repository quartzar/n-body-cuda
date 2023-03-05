//
// Created by quartzar on 24/10/22.
//

#ifndef ORBITERV6_NBKERNEL_N2_CUH
#define ORBITERV6_NBKERNEL_N2_CUH

__device__ float3
orbOrbInteraction(float4 oi, float4 oj, float3 ai);

__device__ float3
tileCalculation(float4 orbPos, float3 force);


__device__ float3
computeOrbitalForces(float4 orbPos, float4* positions, int N);

__global__ void
initHalfKickForces(float4* oldPos, float4* newPos,
                   float4* oldVel, float4* newVel,
                   float4* oldForce, float4* newForce,
                   float *deltaTime, int N);

__global__ void
fullKickForces(float4* oldPos, float4* newPos,
               float4* oldVel, float4* newVel,
               float4* oldForce, float4* newForce,
               float *deltaTime, int N);

__global__ void
integrateNOrbitals(float4* oldPos, float4* newPos,
                   float4* oldVel, float4* newVel,
                   float4* oldForce, float4* newForce,
                   float* oldDT, float* newDT, int N);

#endif //ORBITERV6_NBKERNEL_N2_CUH
