//
//  main.cpp
//  test_metal
//
//  Created by Владимир Ушаков on 04.01.2023.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/Foundation/Foundation.hpp"

#include <iostream>
#include "util.h"


//***C++11 Style:***
#include <chrono>

MTL::ComputePipelineState* mSolverKernelPSO;

// The command queue used to pass commands to the device.
MTL::CommandQueue* mCommandQueue;

// Buffers to hold data.

MTL::Buffer* T;
MTL::Buffer* T_new;
MTL::Buffer* n_ponter;
MTL::Buffer* m_ponter;
MTL::Buffer* param_ponter;
MTL::Buffer* flux_ponter;
MTL::Buffer* mpi_rank_pointer;
MTL::Buffer* mpi_n_pointer;
MTL::Buffer* mpi_left_buffer;
MTL::Buffer* mpi_right_buffer;



void printResults(int n, int m)
{
    auto* p_T_new = (float*)T_new->contents();

    for (unsigned long i = 0; i < n; i++)
    {
        for (unsigned long j = 0; j < m; j++)
        {
            std::cout << p_T_new[i * m + j] << ' ';
        }
        std::cout << std::endl;
    }
}

Flux init_flux(Params param) {
    auto flux = Flux();

    flux.right = -param.k * param.h;
    flux.left = -param.k * param.h;
    flux.top = -param.k * param.h;
    flux.bottom = -param.k * param.h;
    flux.left_edge = -param.kL * param.h * 2;
    flux.right_edge = -param.kR * param.h * 2;
    flux.top_edge = -param.kU * param.h * 2;
    flux.bottom_edge = -param.kD * param.h * 2;
    flux.C = -4 * param.k * param.h;
    flux.CLgran = flux.right + flux.left_edge + flux.top + flux.bottom;
    flux.CRgran = flux.right_edge + flux.left + flux.top + flux.bottom;
    flux.CUgran = flux.right + flux.left + flux.top_edge + flux.bottom;
    flux.DUgran = flux.right + flux.left + flux.top + flux.bottom_edge;
    flux.CLUgran = flux.right + flux.left_edge + flux.top_edge + flux.bottom;
    flux.CRUgran = flux.right_edge + flux.left + flux.top_edge + flux.bottom;
    flux.CLDgran = flux.right + flux.left_edge + flux.top + flux.bottom_edge;
    flux.DRUgran = flux.right_edge + flux.left + flux.top + flux.bottom_edge;

    return flux;
}

void prepareData(int n, int m)
{
    ((int*)n_ponter->contents())[0] = n;
    ((int*)m_ponter->contents())[0] = m;
    auto param = (Params*)param_ponter->contents();
    param[0] = Params();
    param->h = 0.1;
    param->k = 50;

    param->dt = 1;
    param->dz = 1;
    param->Tb = 240;
    param->T0 = 0;

    param->kL = 1;
    param->kR = 0;
    param->kU = 10;
    param->kD = 10;

    auto flux = (Flux*)flux_ponter->contents();
    flux[0] = init_flux(param[0]);

    ((int*)mpi_rank_pointer->contents())[0] = 0;
    ((int*)mpi_n_pointer->contents())[0] = n;
    ((int*)m_ponter->contents())[0] = m;
    ((float**)mpi_left_buffer->contents())[0] = nullptr;
    ((float**)mpi_right_buffer->contents())[0] = nullptr;
}


void encodeSolverCommand (MTL::ComputeCommandEncoder* computeEncoder, int n, int m) {

    // Encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(mSolverKernelPSO);
    computeEncoder->setBuffer(T, 0, 0);
    computeEncoder->setBuffer(T_new, 0, 1);
    computeEncoder->setBuffer(n_ponter, 0, 2);
    computeEncoder->setBuffer(m_ponter, 0, 3);
    computeEncoder->setBuffer(param_ponter, 0, 4);
    computeEncoder->setBuffer(flux_ponter, 0, 5);
    computeEncoder->setBuffer(mpi_rank_pointer, 0, 6);
    computeEncoder->setBuffer(mpi_n_pointer, 0, 7);
    computeEncoder->setBuffer(mpi_left_buffer, 0, 8);
    computeEncoder->setBuffer(mpi_right_buffer, 0, 9);
    computeEncoder->setBuffer(T, 0, 10);

    MTL::Size gridSize = MTL::Size(n, m, 1);

    // Calculate a thread group size.
    NS::UInteger threadGroupSize = mSolverKernelPSO->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > n * m)
    {
        threadGroupSize = n * m;
    }
    MTL::Size threadgroupSize = MTL::Size(32, 32, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}


void sendComputeCommand(int n, int m)
{
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);

    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    encodeSolverCommand(computeEncoder, n, m);

    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}


int main() {
    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();
    NS::Error** error = nullptr;

    MTL::Library* defaultLibrary = pDevice->newDefaultLibrary();
    if (defaultLibrary == nullptr)
    {
        std::cout << "Failed to find the default library.";
        return -1;
    }

    NS::String* funcName = NS::String::string("solver_kernel", NS::ASCIIStringEncoding );
    MTL::Function* solverKernel = defaultLibrary->newFunction(funcName);
    if (solverKernel == nullptr)
    {
        std::cout << "Failed to find the adder function.";
        return -1;
    }

    // Create a compute pipeline state object.
    mSolverKernelPSO = pDevice->newComputePipelineState(solverKernel, error);
    if (mSolverKernelPSO == nullptr)
    {
        std::cout << "Failed to created pipeline state object, error %@.";
        return -1;
    }

    mCommandQueue = pDevice->newCommandQueue();
    if (mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue.";
        return -1;
    }

    int n = 10000;
    int m = 10000;


    T = pDevice->newBuffer(sizeof(float) * n * m, MTL::ResourceStorageModeShared);
    T_new = pDevice->newBuffer(sizeof(float) * n * m, MTL::ResourceStorageModeShared);
    n_ponter = pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    m_ponter = pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    param_ponter = pDevice->newBuffer(sizeof(Params), MTL::ResourceStorageModeShared);
    flux_ponter = pDevice->newBuffer(sizeof(Flux), MTL::ResourceStorageModeShared);
    mpi_rank_pointer = pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    mpi_n_pointer = pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    mpi_left_buffer = pDevice->newBuffer(sizeof(float) * m, MTL::ResourceStorageModeShared);
    mpi_right_buffer = pDevice->newBuffer(sizeof(float) * m, MTL::ResourceStorageModeShared);


    auto tPtr = (float*)T->contents();

    for (unsigned long index = 0; index < n * m; index++)
    {
        tPtr[index] = 0;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i = 0; i < 500; i++) {
        prepareData(n, m);
        sendComputeCommand(n, m);
        auto a = T;
        T = T_new;
        T_new = a;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[µs]" << std::endl;


    printResults(n, m);
}
