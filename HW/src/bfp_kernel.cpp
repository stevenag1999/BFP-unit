// bfp_kernel.cpp - Vitis Kernel for Alveo U55C
// Unified BFP operations: ENCODE → OPERATIONS → DECODE

#include <ap_int.h>
#include <hls_stream.h>
#include "bfp_hls.h"
#include "bfp_ops_hls.h"

// Configuration (matches testbench)
#define WE 5
#define WM 7
#define N 16

using Cfg = BFP_bias<WE, WM>;
using blk_t = BFP_Global<Cfg, N>;

//=============================================================================
// Operation codes
//=============================================================================
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
// MAIN KERNEL: BFP Operations
//=============================================================================
extern "C" {

void bfp_kernel(
    // Control
    const unsigned int operation,      // Operation selector
    const unsigned int n_blocks,       // Number of blocks to process
    
    // Input A (FP32 or BFP format)
    const float* in_fp32_a,           // FP32 input for ENCODE
    const unsigned int* in_exp_a,     // BFP exp input
    const unsigned int* in_sign_a,    // BFP sign input
    const unsigned int* in_mant_a,    // BFP mant input
    
    // Input B (BFP format for binary ops)
    const unsigned int* in_exp_b,
    const unsigned int* in_sign_b,
    const unsigned int* in_mant_b,
    
    // Output (FP32 or BFP format)
    float* out_fp32,                  // FP32 output for DECODE
    unsigned int* out_exp,            // BFP exp output
    unsigned int* out_sign,           // BFP sign output
    unsigned int* out_mant            // BFP mant output
) {
    // Todas las s_axilite en el MISMO bundle "control"
    #pragma HLS INTERFACE s_axilite port=operation
    #pragma HLS INTERFACE s_axilite port=n_blocks
    #pragma HLS INTERFACE s_axilite port=return

    // Memorias por AXI (NO s_axilite). Bundles m_axi separados para ancho de banda.
    // A
    #pragma HLS INTERFACE m_axi port=in_fp32_a offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=in_exp_a  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=in_sign_a offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=in_mant_a offset=slave bundle=gmem3

    // B
    #pragma HLS INTERFACE m_axi port=in_exp_b  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=in_sign_b offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=in_mant_b offset=slave bundle=gmem3

    // Salidas
    #pragma HLS INTERFACE m_axi port=out_fp32  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=out_exp   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=out_sign  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=out_mant  offset=slave bundle=gmem3

    // Process each block
    for (unsigned int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        
        const unsigned int offset = blk_idx * N;
        
        blk_t A{};
        blk_t B{};
        blk_t Z{};
        std::array<float, N> fp_in{}, fp_out{};
        
        //=====================================================================
        // PHASE 1: Load inputs based on operation
        //=====================================================================
        if (operation == OP_ENCODE) {
            // Load FP32 input
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
                fp_in[i] = in_fp32_a[offset + i];
            }
        } else {
            // --- Carga A si NO es RCP ---
            if (operation != OP_RCP) {
                A.exp_shared = in_exp_a[blk_idx];
                for (int i = 0; i < N; i++) {
                #pragma HLS PIPELINE II=1
                    A.sign[i] = in_sign_a[offset + i];
                    A.mant[i] = in_mant_a[offset + i];
                }
            }

            // --- Carga B si es binaria (ADD..DIV) o RCP ---
            if ((operation >= OP_ADD && operation <= OP_DIV) || operation == OP_RCP) {
                B.exp_shared = in_exp_b[blk_idx];
                for (int i = 0; i < N; i++) {
                #pragma HLS PIPELINE II=1
                    B.sign[i] = in_sign_b[offset + i];
                    B.mant[i] = in_mant_b[offset + i];
                        }
                    }
        }
        
        //=====================================================================
        // PHASE 2: Execute operation
        //=====================================================================
        switch (operation) {
            case OP_ENCODE:
                Z = encode_block<Cfg, N>(fp_in);
                break;
                
            case OP_DECODE:
                fp_out = decode_block<Cfg, N>(A);
                Z = A; // Pass through for consistency
                break;
                
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
                Z = A; // No-op
                break;
        }
        
        //=====================================================================
        // PHASE 3: Write outputs
        //=====================================================================
        if (operation == OP_DECODE) {
            // Write FP32 output
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
                out_fp32[offset + i] = fp_out[i];
            }
        } else {
            // Write BFP output
            out_exp[blk_idx] = Z.exp_shared;
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
                out_sign[offset + i] = Z.sign[i];
                out_mant[offset + i] = Z.mant[i];
            }
        }
    }
}

} // extern "C"
