//
//  kernel.metal
//  test_metal
//
//  Created by Владимир Ушаков on 04.01.2023.
//

#include <metal_stdlib>
#include "util.h"

#define index(i,j,m) (i) * (m) + (j)

using namespace metal;

kernel void solver_kernel(
    device const float* T,
    device float* T_new,
    device const int* n_ponter,
    device const int* m_ponter,
    device const Params* param_ponter,
    device const Flux* flux_ponter,
    device const int* mpi_rank_pointer,
    device const int* mpi_n_pointer,
    device const float* mpi_left_buffer,
    device const float* mpi_right_buffer,
        uint2 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
        uint2 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
        uint2 threads_per_threadgroup        [[ threads_per_threadgroup ]])
{
    int n = *n_ponter;
    int m = *m_ponter;
    Params param = *param_ponter;
    Flux flux = *flux_ponter;
    int mpi_rank = *mpi_rank_pointer;
    int mpi_n = *mpi_n_pointer;

    int global_i = threadgroup_position_in_grid[0] * threads_per_threadgroup[0] + thread_position_in_threadgroup[0];
    // int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadgroup_position_in_grid[1] * threads_per_threadgroup[1] + thread_position_in_threadgroup[1];
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = mpi_rank == -1 || mpi_n == -1? global_i : global_i - mpi_n * mpi_rank;
    int current_n = mpi_n == -1 ? n : mpi_n;
    if (i >= 0 && i < current_n && j >= 0 && j < m)
    {
        float left_flux = 0;
        float right_flux = 0;
        float top_flux = 0;
        float bottom_flux = 0;

        if (global_i == 0)
            left_flux = flux.left_edge * (param.Tb - T[index(i, j, m)]);
        else
            if (i == 0 && mpi_left_buffer != nullptr)
                left_flux = flux.left * (mpi_left_buffer[j] - T[index(i, j, m)]);
            else
                left_flux = flux.left * (T[index(i - 1, j, m)] - T[index(i, j, m)]);
        if (global_i == n - 1)
            right_flux = flux.right_edge * (param.T0 - T[index(i, j, m)]);
        else
            if ((i + 1) % current_n == 0 && mpi_right_buffer != nullptr)
                right_flux = flux.right * (mpi_right_buffer[j] - T[index(i, j, m)]);
            else
                right_flux = flux.right * (T[index(i + 1, j, m)] - T[index(i, j, m)]);

        if (j == 0)
            bottom_flux = flux.bottom_edge * (param.T0 - T[index(i, j, m)]);
        else
            bottom_flux = flux.bottom * (T[index(i, j - 1, m)] - T[index(i, j, m)]);
        if (j == m - 1)
            top_flux = flux.top_edge * (param.T0 - T[index(i, j, m)]);
        else
            top_flux = flux.top * (T[index(i, j + 1, m)] - T[index(i, j, m)]);

        T_new[index(i, j, m)] = T[index(i, j, m)] - param.dt * param.h * param.h * param.dz * (left_flux + right_flux + top_flux + bottom_flux);
    }
}
