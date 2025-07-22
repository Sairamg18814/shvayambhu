#include <metal_stdlib>
using namespace metal;

// Optimized matrix multiplication for Metal
kernel void matmul_float32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],  // Rows of A
    constant uint& N [[buffer(4)]],  // Cols of B
    constant uint& K [[buffer(5)]],  // Cols of A / Rows of B
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

// Tiled matrix multiplication for better cache usage
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpig [[thread_position_in_threadgroup]]
) {
    const uint TILE_SIZE = 16;
    
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    uint row = tgid.y * TILE_SIZE + tpig.y;
    uint col = tgid.x * TILE_SIZE + tpig.x;
    
    float sum = 0.0f;
    
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        uint aRow = row;
        uint aCol = t * TILE_SIZE + tpig.x;
        uint bRow = t * TILE_SIZE + tpig.y;
        uint bCol = col;
        
        As[tpig.y][tpig.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        Bs[tpig.y][tpig.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tpig.y][k] * Bs[k][tpig.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}