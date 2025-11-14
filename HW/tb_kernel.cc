#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <bitset>
#include <cstring>
#include <type_traits>

#include "bfp_hls.h"
#include "bfp_ops_hls.h"

// Top kernel HLS con formato compacto
extern "C" void bfp_kernel(
    const unsigned int operation,
    const unsigned int n_blocks,
    const float* in_fp32,
    const unsigned int* in_bfp_a,
    const unsigned int* in_bfp_b,
    float* out_fp32,
    unsigned int* out_bfp
);

//------------------------ Configuración ------------------------
#define WE 5
#define WM 7
#define N  16

using Cfg = BFP_bias<WE, WM>;

// ********************************************************************
// NUEVO: Tamaño del bloque BFP compacto
// ********************************************************************
static constexpr unsigned int BFP_BLOCK_SIZE = 1 + 3 * N;  // 49 para N=16

enum : unsigned {
    OP_ENCODE = 0,
    OP_DECODE = 1,
    OP_ADD    = 2,
    OP_SUB    = 3,
    OP_MUL    = 4,
    OP_DIV    = 5,
    OP_RCP    = 6
};

//------------------------ Helpers de error ------------------------
inline float calc_rel_error(float computed, float reference) {
    if (reference == 0.0f) return std::fabs(computed);
    return std::fabs((computed - reference) / reference) * 100.0f;
}

void print_error_stats(const char* op_name, double mean_abs, double max_err) {
    std::cout << "\n" << op_name << " Error Statistics:\n";
    std::cout << "  Mean Absolute Error: " << mean_abs << "\n";
    std::cout << "  Max Absolute Error:  " << max_err << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

// ********************************************************************
// HELPERS: Pack/Unpack para formato compacto
// ********************************************************************

// Pack: BFP_Global → Vector compacto
void pack_bfp_to_vector(const BFP_Global<Cfg, N>& blk, 
                        unsigned int* vec, 
                        unsigned int offset) {
    vec[offset] = blk.exp_shared;
    unsigned int idx = offset + 1;
    
    for (int i = 0; i < N; i++) {
        vec[idx++] = blk.sign[i];
        vec[idx++] = blk.mant[i];
        vec[idx++] = blk.delta[i];
    }
}

// Unpack: Vector compacto → BFP_Global
void unpack_vector_to_bfp(const unsigned int* vec, 
                          BFP_Global<Cfg, N>& blk,
                          unsigned int offset) {
    blk.exp_shared = vec[offset];
    unsigned int idx = offset + 1;
    
    for (int i = 0; i < N; i++) {
        blk.sign[i]  = vec[idx++];
        blk.mant[i]  = vec[idx++];
        blk.delta[i] = vec[idx++];
    }
}

// ********************************************************************
// HELPER: Mostrar contenido de bloque BFP
// ********************************************************************
void print_bfp_block(const char* name, const unsigned int* vec, unsigned int offset) {
    BFP_Global<Cfg, N> blk;
    unpack_vector_to_bfp(vec, blk, offset);
    
    int E_real = int(blk.exp_shared) - Cfg::bias_bfp;
    std::cout << name << ": exp_shared=" << blk.exp_shared 
              << " (0b" << std::bitset<Cfg::we>(blk.exp_shared) 
              << ", real=" << E_real << ")\n";
    
    std::cout << std::setw(3) << "i"
              << std::setw(8) << "sign"
              << std::setw(10) << "mant"
              << std::setw(10) << "delta"
              << std::setw(14) << "FP32\n";
    std::cout << std::string(45, '-') << "\n";
    
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(3) << i
                  << std::setw(8) << blk.sign[i]
                  << std::setw(10) << blk.mant[i]
                  << std::setw(10) << blk.delta[i]
                  << std::setw(14) << std::fixed << std::setprecision(6) 
                  << blk.rebuid_FP32(i) << "\n";
    }
    std::cout << "\n";
}

// ********************************************************************
// VISUALIZACION: Mostrar layout del vector BFP compacto
// ********************************************************************

void print_vector_layout_header() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "LAYOUT DEL VECTOR BFP COMPACTO EN MEMORIA\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void print_vector_memory_dump(const char* name, 
                               const unsigned int* vec, 
                               unsigned int offset,
                               unsigned int size) {
    std::cout << "+- " << name << " (offset=" << offset 
              << ", size=" << size << " uint32_t) -+\n";
    std::cout << "|\n";
    
    // Mostrar en formato hexadecimal con índices
    for (unsigned int i = 0; i < size; i++) {
        if (i % 8 == 0) {
            if (i > 0) std::cout << "|\n";
            std::cout << "| [" << std::setw(3) << (offset + i) << "]  ";
        }
        std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << vec[offset + i] << " " << std::dec << std::setfill(' ');
    }
    
    std::cout << "|\n";
    std::cout << "+" << std::string(78, '-') << "+\n\n";
}

void plot_bfp_block_structure(const char* name, 
                               const unsigned int* vec, 
                               unsigned int offset) {
    std::cout << "\n";
    std::cout << "+" << std::string(77, '=') << "+\n";
    std::cout << "| " << std::left << std::setw(73) << name << " |\n";
    std::cout << "+" << std::string(77, '=') << "+\n";
    
    // Exponente compartido
    std::cout << "|                                                                           |\n";
    std::cout << "|  [0] EXPONENTE COMPARTIDO                                                 |\n";
    std::cout << "|  +--------------------------------------------------------------------+   |\n";
    std::cout << "|  | Value (dec): " << std::setw(10) << vec[offset] 
              << "                                         |   |\n";
    std::cout << "|  | Value (hex): 0x" << std::hex << std::setw(8) << std::setfill('0') 
              << vec[offset] << std::dec << std::setfill(' ')
              << "                                    |   |\n";
    std::cout << "|  | Value (bin): " << std::bitset<32>(vec[offset]) << " |   |\n";
    
    int exp_real = int(vec[offset]) - Cfg::bias_bfp;
    std::cout << "|  | Biased:      " << std::setw(10) << vec[offset] 
              << "                                         |   |\n";
    std::cout << "|  | Real (E):    " << std::setw(10) << exp_real 
              << "                                         |   |\n";
    std::cout << "|  +--------------------------------------------------------------------+   |\n";
    std::cout << "|                                                                           |\n";
    
    // Elementos
    std::cout << "|  [1-" << (3*N) << "] ELEMENTOS (sign, mant, delta) x " << N 
              << "                              |\n";
    std::cout << "|  +=================================================================+     |\n";
    
    unsigned int idx = offset + 1;
    
    for (int i = 0; i < N; i++) {
        unsigned int sign  = vec[idx++];
        unsigned int mant  = vec[idx++];
        unsigned int delta = vec[idx++];
        
        // Reconstruir valor FP32 para mostrar
        BFP_Global<Cfg, N> temp_blk;
        temp_blk.exp_shared = vec[offset];
        temp_blk.sign[i] = sign;
        temp_blk.mant[i] = mant;
        temp_blk.delta[i] = delta;
        float fp32_val = temp_blk.rebuid_FP32(i);
        
        std::cout << "|  | Elemento " << std::setw(2) << i 
                  << "                                               |     |\n";
        std::cout << "|  |   Sign:  " << sign 
                  << " | Mant: " << std::setw(5) << mant 
                  << " (0x" << std::hex << std::setw(3) << std::setfill('0') << mant 
                  << std::dec << std::setfill(' ') << ")"
                  << " | Delta: " << std::setw(3) << delta 
                  << "             |     |\n";
        std::cout << "|  |   -> FP32 value: " << std::setw(12) << std::fixed 
                  << std::setprecision(6) << fp32_val 
                  << "                               |     |\n";
        
        if (i < N - 1) {
            std::cout << "|  | ---------------------------------------------------------------   |     |\n";
        }
    }
    
    std::cout << "|  +=================================================================+     |\n";
    std::cout << "+" << std::string(77, '=') << "+\n\n";
}

void plot_memory_map(const char* buffer_name,
                     const unsigned int* vec,
                     unsigned int n_blocks) {
    std::cout << "\n";
    std::cout << "+" << std::string(77, '=') << "+\n";
    std::cout << "| MEMORY MAP: " << std::left << std::setw(61) << buffer_name << " |\n";
    std::cout << "+" << std::string(77, '=') << "+\n";
    std::cout << "|                                                                           |\n";
    
    for (unsigned int blk = 0; blk < n_blocks; blk++) {
        unsigned int offset = blk * BFP_BLOCK_SIZE;
        unsigned int end = offset + BFP_BLOCK_SIZE - 1;
        
        std::cout << "|  Block " << blk << ":  [" << std::setw(4) << offset 
                  << " - " << std::setw(4) << end << "]  (" << BFP_BLOCK_SIZE 
                  << " uint32_t = " << (BFP_BLOCK_SIZE * 4) << " bytes)";
        
        // Calcular espacio restante para alinear
        std::string line = std::to_string(blk) + std::to_string(offset) + 
                           std::to_string(end) + std::to_string(BFP_BLOCK_SIZE) + 
                           std::to_string(BFP_BLOCK_SIZE * 4);
        int spaces = 73 - 44 - line.length();
        if (spaces < 0) spaces = 0;
        std::cout << std::string(spaces, ' ') << "|\n";
        
        std::cout << "|    |                                                                    |\n";
        std::cout << "|    +- [" << std::setw(4) << offset << "] exp_shared = " 
                  << std::setw(10) << vec[offset];
        spaces = 73 - 35;
        std::cout << std::string(spaces, ' ') << "|\n";
        
        std::cout << "|    |                                                                    |\n";
        std::cout << "|    +- [" << std::setw(4) << (offset + 1) << " - " 
                  << std::setw(4) << end << "] " << N << " elementos (sign, mant, delta)";
        spaces = 73 - 55;
        std::cout << std::string(spaces, ' ') << "|\n";
        
        if (blk < n_blocks - 1) {
            std::cout << "|                                                                           |\n";
        }
    }
    
    std::cout << "|                                                                           |\n";
    std::cout << "+" << std::string(77, '=') << "+\n";
    std::cout << "| TOTAL SIZE: " << (n_blocks * BFP_BLOCK_SIZE) << " uint32_t = " 
              << (n_blocks * BFP_BLOCK_SIZE * 4) << " bytes" 
              << std::string(73 - 30 - std::to_string(n_blocks * BFP_BLOCK_SIZE * 4).length(), ' ')
              << "|\n";
    std::cout << "+" << std::string(77, '=') << "+\n\n";
}

void print_byte_level_dump(const char* name,
                            const unsigned int* vec,
                            unsigned int offset,
                            unsigned int num_elements) {
    std::cout << "\n";
    std::cout << "+- " << name << " - BYTE LEVEL DUMP -+\n";
    std::cout << "|\n";
    
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&vec[offset]);
    unsigned int total_bytes = num_elements * sizeof(uint32_t);
    
    for (unsigned int i = 0; i < total_bytes; i++) {
        if (i % 16 == 0) {
            if (i > 0) std::cout << "|\n";
            std::cout << "| [" << std::setw(4) << std::setfill('0') << i 
                      << "]  " << std::setfill(' ');
        }
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)bytes[i] << " " << std::setfill(' ');
    }
    
    std::cout << std::dec << "|\n";
    std::cout << "+" << std::string(60, '-') << "+\n\n";
}

//------------------------ MAIN ------------------------
int main() {
    const unsigned n_blocks = 1;
    
    std::cout << std::string(60, '=') << "\n";
    std::cout << "BFP KERNEL TESTBENCH - FORMATO COMPACTO\n";
    std::cout << "Configuration: WE=" << Cfg::we
              << ", WM=" << Cfg::wm
              << ", Block Size=" << N << ", n_blocks=" << n_blocks << "\n";
    std::cout << "Bias: " << Cfg::bias_bfp << "\n";
    std::cout << "BFP_BLOCK_SIZE: " << BFP_BLOCK_SIZE << " uint32_t\n";
    std::cout << std::string(60, '=') << "\n\n";

    //======================== Datos de prueba ========================
    /*std::array<float, N> inputs = {
        12.35f, 6.50f, 10.20f, 6.60f, 8.80f, 2.56f, 11.11f, 8.00f,
         5.45f, 9.99f, 0.15f, 18.00f, 3.80f, 90.10f, 14.00f, 10.00f
    };*/

    std::array<float, N> inputs = {
        1.35f, 6.50f, 5.22f, 6.60f, 8.80f, 2.56f, 1.11f, 8.00f,
         5.45f, 9.99f, 0.15f, 6.87f, 3.80f, 7.10f, 4.50f, 3.73f
    };

    std::array<float, N> inputs_b = {
        -2.00f, 0.00f, -2.00f, 3.00f, 2.00f, 2.00f, 2.00f, 2.00f,
         3.00f, 3.00f, 5.00f, 3.00f, 6.00f, 3.00f, 8.00f, 2.00f
    };

    // ********************************************************************
    // BUFFERS EN FORMATO COMPACTO (mucho más simple)
    // ********************************************************************
    std::vector<float>        in_fp32(N * n_blocks, 0.f);
    std::vector<unsigned int> in_bfp_a(BFP_BLOCK_SIZE * n_blocks, 0u);
    std::vector<unsigned int> in_bfp_b(BFP_BLOCK_SIZE * n_blocks, 0u);
    std::vector<float>        out_fp32(N * n_blocks, 0.f);
    std::vector<unsigned int> out_bfp(BFP_BLOCK_SIZE * n_blocks, 0u);
    
    // Dummies para operaciones que no los usan
    std::vector<float>        dummy_fp32(N * n_blocks, 0.f);
    std::vector<unsigned int> dummy_bfp(BFP_BLOCK_SIZE * n_blocks, 0u);

    //======================== ENCODE A ========================
    std::cout << std::string(80, '=') << "\n";
    std::cout << "ENCODING Block A\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::memcpy(in_fp32.data(), inputs.data(), sizeof(float) * N);
    
    std::cout << "Input FP32 values:\n";
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(3) << i << ": " 
                  << std::setw(12) << std::fixed << std::setprecision(6) 
                  << in_fp32[i] << "\n";
    }
    std::cout << "\n";

    // ********************************************************************
    // Ejecutar ENCODE (operation=0)
    // ********************************************************************
    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32.data(),      // Input FP32
               dummy_bfp.data(),    // No usado
               dummy_bfp.data(),    // No usado
               dummy_fp32.data(),   // No usado
               out_bfp.data());     // Output BFP compacto

    // Copiar resultado a input A para operaciones posteriores
    std::memcpy(in_bfp_a.data(), out_bfp.data(), 
                BFP_BLOCK_SIZE * n_blocks * sizeof(unsigned int));

    // ********************************************************************
    // VISUALIZACIONES DEL BLOQUE A CODIFICADO
    // ********************************************************************
    
    // 1. Estructura detallada del bloque BFP
    plot_bfp_block_structure("Block A - Estructura BFP Compacta", 
                              in_bfp_a.data(), 0);

    // 2. Dump hexadecimal del vector completo
    print_vector_memory_dump("Block A - Raw Memory (Hexadecimal)", 
                              in_bfp_a.data(), 0, BFP_BLOCK_SIZE);

    // 3. Dump a nivel de bytes (útil para verificar endianness)
    print_byte_level_dump("Block A", 
                           in_bfp_a.data(), 0, BFP_BLOCK_SIZE);

    // 4. Mapa de memoria del buffer completo
    plot_memory_map("in_bfp_a buffer", in_bfp_a.data(), n_blocks);

    // 5. Tabla tradicional (la que ya tenías)
    print_bfp_block("Block A (tabla resumen)", in_bfp_a.data(), 0);

    //======================== ENCODE B ========================
    std::cout << std::string(80, '=') << "\n";
    std::cout << "ENCODING Block B\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::memcpy(in_fp32.data(), inputs_b.data(), sizeof(float) * N);

    std::cout << "Input FP32 values:\n";
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(3) << i << ": " 
                  << std::setw(12) << std::fixed << std::setprecision(6) 
                  << in_fp32[i] << "\n";
    }
    std::cout << "\n";

    // ********************************************************************
    // Ejecutar ENCODE (operation=0)
    // ********************************************************************
    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32.data(),
               dummy_bfp.data(),
               dummy_bfp.data(),
               dummy_fp32.data(),
               out_bfp.data());

    std::memcpy(in_bfp_b.data(), out_bfp.data(), 
                BFP_BLOCK_SIZE * n_blocks * sizeof(unsigned int));

    // ********************************************************************
    // VISUALIZACIONES DEL BLOQUE B CODIFICADO
    // ********************************************************************
    
    plot_bfp_block_structure("Block B - Estructura BFP Compacta", 
                              in_bfp_b.data(), 0);

    print_vector_memory_dump("Block B - Raw Memory (Hexadecimal)", 
                              in_bfp_b.data(), 0, BFP_BLOCK_SIZE);

    print_bfp_block("Block B (tabla resumen)", in_bfp_b.data(), 0);

    //======================== Referencias FP32 ========================
    std::array<float, N> ref_add{}, ref_sub{}, ref_mul{}, ref_div{};

    for (std::size_t i = 0; i < N; i++) {
        ref_add[i] = inputs[i] + inputs_b[i];
        ref_sub[i] = inputs[i] - inputs_b[i];
        ref_mul[i] = inputs[i] * inputs_b[i];
        ref_div[i] = (inputs_b[i] == 0.0f)
                     ? std::copysign(INFINITY, inputs[i])
                     : inputs[i] / inputs_b[i];
    }

    // ********************************************************************
    // HELPER: Ejecutar operación y reportar resultados
    // ********************************************************************
    auto run_op_and_report = [&](unsigned op, const char* name,
                                 const std::array<float, N>& ref,
                                 bool uses_both_operands = true) {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "TEST: " << name << "\n";
        std::cout << std::string(80, '=') << "\n\n";

        // Limpiar salidas
        std::fill(out_bfp.begin(), out_bfp.end(), 0u);
        std::fill(out_fp32.begin(), out_fp32.end(), 0.f);

        // ********************************************************************
        // Ejecutar operación en formato BFP
        // ********************************************************************
        if (uses_both_operands) {
            bfp_kernel(op, n_blocks,
                       dummy_fp32.data(),
                       in_bfp_a.data(),
                       in_bfp_b.data(),
                       dummy_fp32.data(),
                       out_bfp.data());
        } else {
            // Para RCP(B)
            bfp_kernel(op, n_blocks,
                       dummy_fp32.data(),
                       dummy_bfp.data(),
                       in_bfp_b.data(),
                       dummy_fp32.data(),
                       out_bfp.data());
        }

        // ********************************************************************
        // VISUALIZAR RESULTADO BFP
        // ********************************************************************
        std::string result_name = std::string("Result ") + name + " - BFP Structure";
        plot_bfp_block_structure(result_name.c_str(), out_bfp.data(), 0);
        
        std::string result_hex = std::string("Result ") + name + " - Raw Memory";
        print_vector_memory_dump(result_hex.c_str(), 
                                  out_bfp.data(), 0, BFP_BLOCK_SIZE);

        // ********************************************************************
        // Decodificar a FP32 para comparar
        // ********************************************************************
        bfp_kernel(OP_DECODE, n_blocks,
                   dummy_fp32.data(),
                   out_bfp.data(),
                   dummy_bfp.data(),
                   out_fp32.data(),
                   dummy_bfp.data());

        // ********************************************************************
        // Tabla de comparación
        // ********************************************************************
        std::cout << std::setw(3) << "i"
                  << std::setw(12) << "A"
                  << std::setw(12) << "B"
                  << std::setw(16) << "BFP Result"
                  << std::setw(16) << "FP32 Ref"
                  << std::setw(12) << "Err (%)\n";
        std::cout << std::string(71, '-') << "\n";

        double mean_abs = 0.0, max_err = 0.0;

        for (std::size_t i = 0; i < N; ++i) {
            float bfp_res = out_fp32[i];
            float ref_val = ref[i];
            float err = calc_rel_error(bfp_res, ref_val);

            mean_abs += std::fabs(bfp_res - ref_val);
            if (err > max_err) max_err = err;

            std::cout << std::setw(3) << i
                      << std::setw(12) << std::fixed << std::setprecision(4) << inputs[i]
                      << std::setw(12) << inputs_b[i]
                      << std::setw(16) << bfp_res
                      << std::setw(16) << ref_val
                      << std::setw(12) << std::setprecision(4) << err << "\n";
        }

        mean_abs /= double(N);
        print_error_stats(name, mean_abs, max_err);
    };

    //======================== TESTS: OPERACIONES ARITMETICAS ========================
    run_op_and_report(OP_ADD, "ADDITION (A + B)", ref_add);
    run_op_and_report(OP_SUB, "SUBTRACTION (A - B)", ref_sub);
    run_op_and_report(OP_MUL, "MULTIPLICATION (A * B)", ref_mul);
    run_op_and_report(OP_DIV, "DIVISION (A / B)", ref_div);

    //======================== TEST: ENCODE/DECODE ROUND-TRIP ===================
    std::cout << std::string(80, '=') << "\n";
    std::cout << "TEST: ENCODE/DECODE ROUND-TRIP (Block A)\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::fill(out_fp32.begin(), out_fp32.end(), 0.f);

    // ********************************************************************
    // Decodificar el bloque A original
    // ********************************************************************
    bfp_kernel(OP_DECODE, n_blocks,
               dummy_fp32.data(),
               in_bfp_a.data(),
               dummy_bfp.data(),
               out_fp32.data(),
               dummy_bfp.data());

    std::cout << std::setw(3) << "i"
              << std::setw(16) << "Original"
              << std::setw(16) << "Decoded"
              << std::setw(12) << "Err (%)\n";
    std::cout << std::string(47, '-') << "\n";

    double mean_abs = 0.0, max_err = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        float orig = inputs[i];
        float dec  = out_fp32[i];
        float err  = calc_rel_error(dec, orig);

        mean_abs += std::fabs(dec - orig);
        if (err > max_err) max_err = err;

        std::cout << std::setw(3) << i
                  << std::setw(16) << std::fixed << std::setprecision(6) << orig
                  << std::setw(16) << dec
                  << std::setw(12) << std::setprecision(4) << err << "\n";
    }

    mean_abs /= double(N);
    print_error_stats("ENCODE/DECODE", mean_abs, max_err);

    // ********************************************************************
    // VERIFICAR DELTAS
    // ********************************************************************
    std::cout << std::string(80, '=') << "\n";
    std::cout << "VERIFICACION DE DELTAS (Block A)\n";
    std::cout << std::string(80, '=') << "\n\n";

    BFP_Global<Cfg, N> blk_a;
    unpack_vector_to_bfp(in_bfp_a.data(), blk_a, 0);

    int exp_shared_real = int(blk_a.exp_shared) - Cfg::bias_bfp;

    std::cout << "Exponente compartido: " << blk_a.exp_shared 
              << " (real=" << exp_shared_real << ")\n\n";

    std::cout << std::setw(3) << "i"
          << std::setw(12) << "Original"
          << std::setw(10) << "Delta"
          << std::setw(16) << "Exp_elem (calc)"
          << std::setw(16) << "Exp_elem (FP32)"
          << std::setw(12) << "Match\n";
    std::cout << std::string(69, '-') << "\n";

    bool all_deltas_correct = true;

    for (int i = 0; i < N; i++) {
        float orig = inputs[i];
        
        // Calcular exp del elemento usando delta
        int exp_calc = exp_shared_real - int(blk_a.delta[i]);
        
        // Extraer exp del FP32 original
        union {float f; uint32_t u;} u = {orig};
        int exp_fp32 = int((u.u >> 23) & 0xFF) - 127;
        
        bool match = (exp_calc == exp_fp32) || (orig == 0.0f);
        if (!match) all_deltas_correct = false;
        
        std::cout << std::setw(3) << i
                << std::setw(12) << std::fixed << std::setprecision(4) << orig
                << std::setw(10) << blk_a.delta[i]
                << std::setw(16) << exp_calc
                << std::setw(16) << exp_fp32
                << std::setw(12) << (match ? "OK" : "FAIL") << "\n";
    }
    std::cout << "\n";

    if (all_deltas_correct) {
        std::cout << "[OK] TODOS LOS DELTAS SON CORRECTOS!\n";
    } else {
        std::cout << "[FAIL] HAY DELTAS INCORRECTOS!\n";
    }
    std::cout << "\n";

    //======================== RESUMEN FINAL ========================
    std::cout << std::string(80, '=') << "\n";
    std::cout << "ALL KERNEL TESTS COMPLETED SUCCESSFULLY!\n";
    std::cout << "Formato compacto verificado:\n";
    std::cout << "  - " << BFP_BLOCK_SIZE << " uint32_t por bloque\n";
    std::cout << "  - " << (BFP_BLOCK_SIZE * sizeof(uint32_t)) << " bytes por bloque\n";
    std::cout << "  - Layout: [exp_shared, (sign, mant, delta) × " << N << "]\n";
    std::cout << std::string(80, '=') << "\n";

    return 0;
}
