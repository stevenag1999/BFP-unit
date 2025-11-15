#ifndef FP32_OPS_HLS_H
#define FP32_OPS_HLS_H

#include <ap_int.h>
#include <cstdint>
#include <cmath>

// No special structure needed for FP32 - direct float operations

//=============================================================================
// FP32 ADD OPERATION: Z = A + B
//=============================================================================
template<std::size_t Block_size>
void fp32_add_blocks(
    const float A[Block_size],
    const float B[Block_size], 
    float Z[Block_size]
) {
#pragma HLS INLINE off
    
ADD_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        Z[i] = A[i] + B[i];
    }
}

//=============================================================================
// FP32 SUB OPERATION: Z = A - B  
//=============================================================================
template<std::size_t Block_size>
void fp32_sub_blocks(
    const float A[Block_size],
    const float B[Block_size],
    float Z[Block_size]
) {
#pragma HLS INLINE off
    
SUB_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        Z[i] = A[i] - B[i];
    }
}

//=============================================================================
// FP32 MUL OPERATION: Z = A * B
//=============================================================================
template<std::size_t Block_size>
void fp32_mul_blocks(
    const float A[Block_size],
    const float B[Block_size], 
    float Z[Block_size]
) {
#pragma HLS INLINE off
    
MUL_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        Z[i] = A[i] * B[i];
    }
}

//=============================================================================
// FP32 DIV OPERATION: Z = A / B
//=============================================================================
template<std::size_t Block_size> 
void fp32_div_blocks(
    const float A[Block_size],
    const float B[Block_size],
    float Z[Block_size]
) {
#pragma HLS INLINE off
    
DIV_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        // Handle division by zero gracefully
        if (B[i] == 0.0f) {
            Z[i] = (A[i] >= 0) ? std::numeric_limits<float>::infinity() 
                               : -std::numeric_limits<float>::infinity();
        } else {
            Z[i] = A[i] / B[i];
        }
    }
}

//=============================================================================
// FP32 RECIPROCAL OPERATION: Z = 1 / B
//=============================================================================
template<std::size_t Block_size>
void fp32_rcp_blocks(
    const float B[Block_size],
    float Z[Block_size]  
) {
#pragma HLS INLINE off
    
RCP_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        // Handle division by zero
        if (B[i] == 0.0f) {
            Z[i] = std::numeric_limits<float>::infinity();
        } else {
            Z[i] = 1.0f / B[i];
        }
    }
}

#endif // FP32_OPS_HLS_H
