// bfp_kernel.cpp - OPTIMIZED VERSION for Alveo U55C
// Key improvements:
// 1. DATAFLOW between load/compute/store
// 2. Array partitioning for parallel access
// 3. Burst memory access optimization
// 4. Pipeline directives

#include <ap_int.h>
#include <hls_stream.h>
#include "bfp_hls.h"
//#include "bfp_hls_opt.h"
#include "bfp_ops_hls.h"

// Configuration
#define WE 5
#define WM 7
#define N 16

using Cfg = BFP_bias<WE, WM>;
using blk_t = BFP_Global<Cfg, N>;

// Operation codes
typedef enum : unsigned int {
    OP_ENCODE = 0,
    OP_DECODE = 1,
    OP_ADD    = 2,
    OP_SUB    = 3,
    OP_MUL    = 4,
    OP_DIV    = 5,
    OP_RCP    = 6
} bfp_op_t;

//=============================================================================
// OPTIMIZED DATA STRUCTURES
//=============================================================================
struct bfp_block_data {
    float fp32_data[N];
    unsigned int exp_shared;
    unsigned int sign[N];
    unsigned int mant[N];
};

//=============================================================================
// LOAD FUNCTION - Optimized burst reads
//=============================================================================
static void load_inputs(
    const unsigned int operation,
    const unsigned int blk_idx,
    // Inputs A
    const float* in_fp32_a,
    const unsigned int* in_exp_a,
    const unsigned int* in_sign_a,
    const unsigned int* in_mant_a,
    // Inputs B
    const unsigned int* in_exp_b,
    const unsigned int* in_sign_b,
    const unsigned int* in_mant_b,
    // Output streams
    bfp_block_data& A_data,
    bfp_block_data& B_data,
    bool& load_A,
    bool& load_B
) {
#pragma HLS INLINE off

    const unsigned int offset = blk_idx * N;
    
    load_A = false;
    load_B = false;
    
    // Load based on operation type
    if (operation == OP_ENCODE) {
        // Load FP32 for encoding
        load_fp32_loop: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            A_data.fp32_data[i] = in_fp32_a[offset + i];
        }
        load_A = true;
        
    } else if (operation == OP_DECODE) {
        // Load BFP A for decoding
        A_data.exp_shared = in_exp_a[blk_idx];
        load_bfp_a_decode: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            A_data.sign[i] = in_sign_a[offset + i];
            A_data.mant[i] = in_mant_a[offset + i];
        }
        load_A = true;
        
    } else if (operation == OP_RCP) {
        // Load only B for reciprocal
        B_data.exp_shared = in_exp_b[blk_idx];
        load_bfp_b_rcp: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            B_data.sign[i] = in_sign_b[offset + i];
            B_data.mant[i] = in_mant_b[offset + i];
        }
        load_B = true;
        
    } else {
        // Binary operations: Load both A and B
        A_data.exp_shared = in_exp_a[blk_idx];
        B_data.exp_shared = in_exp_b[blk_idx];
        
        load_bfp_ab: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            A_data.sign[i] = in_sign_a[offset + i];
            A_data.mant[i] = in_mant_a[offset + i];
            B_data.sign[i] = in_sign_b[offset + i];
            B_data.mant[i] = in_mant_b[offset + i];
        }
        load_A = true;
        load_B = true;
    }
}

//=============================================================================
// COMPUTE FUNCTION - Execute BFP operation
//=============================================================================
static void compute_operation(
    const unsigned int operation,
    const bfp_block_data& A_data,
    const bfp_block_data& B_data,
    const bool load_A,
    const bool load_B,
    bfp_block_data& Z_data
) {
#pragma HLS INLINE off

    blk_t A{}, B{}, Z{};
    std::array<float, N> fp_in{}, fp_out{};
    
    // Transfer input data to BFP structures
    if (operation == OP_ENCODE) {
        transfer_fp32: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            fp_in[i] = A_data.fp32_data[i];
        }
        Z = encode_block<Cfg, N>(fp_in);
        
    } else if (operation == OP_DECODE) {
        A.exp_shared = A_data.exp_shared;
        transfer_bfp_decode: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            A.sign[i] = A_data.sign[i];
            A.mant[i] = A_data.mant[i];
        }
        fp_out = decode_block<Cfg, N>(A);
        
        // Transfer back to output structure
        transfer_fp32_out: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            Z_data.fp32_data[i] = fp_out[i];
        }
        Z = A; // Pass through
        
    } else {
        // Prepare A and B for binary operations
        if (load_A) {
            A.exp_shared = A_data.exp_shared;
            prep_a: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
                A.sign[i] = A_data.sign[i];
                A.mant[i] = A_data.mant[i];
            }
        }
        
        if (load_B) {
            B.exp_shared = B_data.exp_shared;
            prep_b: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
                B.sign[i] = B_data.sign[i];
                B.mant[i] = B_data.mant[i];
            }
        }
        
        // Execute operation
        switch (operation) {
            case OP_ADD:
                Z = add_blocks<Cfg, N>(A, B);
                break;
            case OP_SUB:
                Z = sub_blocks<Cfg, N>(A, B);
                break;
            case OP_MUL:
                Z = mul_blocks<Cfg, N>(A, B);
                break;
            case OP_DIV:
                Z = div_blocks<Cfg, N>(A, B);
                break;
            case OP_RCP:
                Z = rcp_blocks<Cfg, N>(B);
                break;
            default:
                Z = A;
                break;
        }
    }
    
    // Transfer result to output structure
    if (operation != OP_DECODE) {
        Z_data.exp_shared = Z.exp_shared;
        transfer_z: for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            Z_data.sign[i] = Z.sign[i];
            Z_data.mant[i] = Z.mant[i];
        }
    }
}

//=============================================================================
// STORE FUNCTION - Optimized burst writes
//=============================================================================
static void store_outputs(
    const unsigned int operation,
    const unsigned int blk_idx,
    const bfp_block_data& Z_data,
    float* out_fp32,
    unsigned int* out_exp,
    unsigned int* out_sign,
    unsigned int* out_mant
) {
#pragma HLS INLINE off

    const unsigned int offset = blk_idx * N;
    
    if (operation == OP_DECODE) {
        // Write FP32 output
        store_fp32: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            out_fp32[offset + i] = Z_data.fp32_data[i];
        }
    } else {
        // Write BFP output
        out_exp[blk_idx] = Z_data.exp_shared;
        store_bfp: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            out_sign[offset + i] = Z_data.sign[i];
            out_mant[offset + i] = Z_data.mant[i];
        }
    }
}

//=============================================================================
// MAIN KERNEL - OPTIMIZED with DATAFLOW
//=============================================================================
extern "C" {

void bfp_kernel(
    // Control
    const unsigned int operation,
    const unsigned int n_blocks,
    
    // Input A (FP32 or BFP format)
    const float* in_fp32_a,
    const unsigned int* in_exp_a,
    const unsigned int* in_sign_a,
    const unsigned int* in_mant_a,
    
    // Input B (BFP format for binary ops)
    const unsigned int* in_exp_b,
    const unsigned int* in_sign_b,
    const unsigned int* in_mant_b,
    
    // Output (FP32 or BFP format)
    float* out_fp32,
    unsigned int* out_exp,
    unsigned int* out_sign,
    unsigned int* out_mant
) {
    // Interface pragmas
    #pragma HLS INTERFACE s_axilite port=operation bundle=control
    #pragma HLS INTERFACE s_axilite port=n_blocks bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Memory interfaces with optimizations
    #pragma HLS INTERFACE m_axi port=in_fp32_a offset=slave bundle=gmem0 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=in_exp_a offset=slave bundle=gmem1 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=in_sign_a offset=slave bundle=gmem2 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=in_mant_a offset=slave bundle=gmem3 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64

    #pragma HLS INTERFACE m_axi port=in_exp_b offset=slave bundle=gmem1 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=in_sign_b offset=slave bundle=gmem2 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=in_mant_b offset=slave bundle=gmem3 \
        max_read_burst_length=16 num_read_outstanding=4 latency=64

    #pragma HLS INTERFACE m_axi port=out_fp32 offset=slave bundle=gmem0 \
        max_write_burst_length=16 num_write_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=out_exp offset=slave bundle=gmem1 \
        max_write_burst_length=16 num_write_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=out_sign offset=slave bundle=gmem2 \
        max_write_burst_length=16 num_write_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=out_mant offset=slave bundle=gmem3 \
        max_write_burst_length=16 num_write_outstanding=4 latency=64

    // Main processing loop with DATAFLOW
    process_blocks: for (unsigned int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128 avg=32
        
        // Local data structures
        bfp_block_data A_data, B_data, Z_data;
        bool load_A, load_B;
        
        // CRITICAL: Partition arrays for parallel access
        #pragma HLS ARRAY_PARTITION variable=A_data.fp32_data complete
        #pragma HLS ARRAY_PARTITION variable=A_data.sign complete
        #pragma HLS ARRAY_PARTITION variable=A_data.mant complete
        #pragma HLS ARRAY_PARTITION variable=B_data.sign complete
        #pragma HLS ARRAY_PARTITION variable=B_data.mant complete
        #pragma HLS ARRAY_PARTITION variable=Z_data.fp32_data complete
        #pragma HLS ARRAY_PARTITION variable=Z_data.sign complete
        #pragma HLS ARRAY_PARTITION variable=Z_data.mant complete
        
        // Three-stage pipeline with DATAFLOW
        #pragma HLS DATAFLOW
        
        load_inputs(operation, blk_idx,
                   in_fp32_a, in_exp_a, in_sign_a, in_mant_a,
                   in_exp_b, in_sign_b, in_mant_b,
                   A_data, B_data, load_A, load_B);
        
        compute_operation(operation, A_data, B_data, load_A, load_B, Z_data);
        
        store_outputs(operation, blk_idx, Z_data,
                     out_fp32, out_exp, out_sign, out_mant);
    }
}

} // extern "C"
