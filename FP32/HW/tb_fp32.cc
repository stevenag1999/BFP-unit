#include <iostream>
#include <cmath>
#include <limits>

// Incluimos las mismas operaciones de referencia
#include "fp32_ops_hls.h"

// =============================
// Configuración
// =============================
#define N           16          // Tamaño de bloque (mismo que el kernel)
#define MAX_BLOCKS  4           // Nº máximo de bloques que vas a probar en cosim
#define TOTAL_ELEMS (N * MAX_BLOCKS)

// =============================
// Prototipo del kernel HLS
// (usa EXACTAMENTE la misma firma que en fp32_kernel.cpp)
// =============================
extern "C" {
void fp32_kernel(
    const unsigned int operation,
    const unsigned int n_blocks,
    const float* in_fp32_a,
    const float* in_fp32_b,
    float* out_fp32
);
}

// =============================
// Códigos de operación (igual que en fp32_kernel.cpp)
// =============================
typedef enum : unsigned int {
    OP_ADD = 2,
    OP_SUB = 3,
    OP_MUL = 4,
    OP_DIV = 5,
    OP_RCP = 6
} fp32_op_t;

// =============================
// Buffers globales para que HLS los vea claros
// =============================
static float A[TOTAL_ELEMS];
static float B[TOTAL_ELEMS];
static float Z_hw[TOTAL_ELEMS];   // salida del kernel
static float Z_sw[TOTAL_ELEMS];   // salida de referencia en C

// =============================
// Función de ayuda: genera datos
// =============================
void init_inputs(unsigned int n_blocks) {
    const unsigned int total = N * n_blocks;
    for (unsigned int i = 0; i < total; ++i) {
        // Algo simple pero no trivial
        A[i] = 0.1f * i;             // 0, 0.1, 0.2, ...
        B[i] = 0.2f * (i + 1);       // 0.2, 0.4, 0.6, ...
    }

    // Opcional: meter algún cero para probar div/rcp por cero
    if (total > 0) {
        B[0] = 0.0f;
    }
}

// =============================
// Calcula la referencia en C usando fp32_ops_hls.h
// =============================
void compute_reference(unsigned int op, unsigned int n_blocks) {
    const unsigned int total = N * n_blocks;

    for (unsigned int blk = 0; blk < n_blocks; ++blk) {
        float* a_blk = &A[blk * N];
        float* b_blk = &B[blk * N];
        float* z_blk = &Z_sw[blk * N];

        switch (op) {
            case OP_ADD:
                fp32_add_blocks<N>(a_blk, b_blk, z_blk);
                break;
            case OP_SUB:
                fp32_sub_blocks<N>(a_blk, b_blk, z_blk);
                break;
            case OP_MUL:
                fp32_mul_blocks<N>(a_blk, b_blk, z_blk);
                break;
            case OP_DIV:
                fp32_div_blocks<N>(a_blk, b_blk, z_blk);
                break;
            case OP_RCP:
                // En RCP solo usamos B como entrada
                fp32_rcp_blocks<N>(b_blk, z_blk);
                break;
            default:
                // Por si las moscas
                fp32_add_blocks<N>(a_blk, b_blk, z_blk);
                break;
        }
    }
}

// =============================
// Compara HW (kernel) vs SW (referencia)
// =============================
bool check_results(unsigned int op, unsigned int n_blocks) {
    const unsigned int total = N * n_blocks;
    bool ok = true;
    float max_abs_err = 0.0f;

    for (unsigned int i = 0; i < total; ++i) {
        float hw = Z_hw[i];
        float sw = Z_sw[i];

        // Comparación robusta: maneja inf, NaN y diferencias pequeñas
        if (std::isnan(sw) || std::isnan(hw)) {
            if (!(std::isnan(sw) && std::isnan(hw))) {
                std::cout << "[OP " << op << "] NaN mismatch en i=" << i
                          << " HW=" << hw << " SW=" << sw << std::endl;
                ok = false;
            }
            continue;
        }

        if (std::isinf(sw) || std::isinf(hw)) {
            if (!(std::isinf(sw) && std::isinf(hw))) {
                std::cout << "[OP " << op << "] INF mismatch en i=" << i
                          << " HW=" << hw << " SW=" << sw << std::endl;
                ok = false;
            }
            continue;
        }

        float abs_err = std::fabs(hw - sw);
        float rel_err = (std::fabs(sw) > 1e-6f) ? abs_err / std::fabs(sw) : abs_err;

        if (abs_err > 1e-4f && rel_err > 1e-4f) {
            std::cout << "[OP " << op << "] Mismatch en i=" << i
                      << " HW=" << hw
                      << " SW=" << sw
                      << " abs_err=" << abs_err
                      << " rel_err=" << rel_err
                      << std::endl;
            ok = false;
        }

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
        }
    }

    std::cout << "[OP " << op << "] Max abs err = " << max_abs_err
              << " -> " << (ok ? "OK" : "FAIL") << std::endl;

    return ok;
}

// =============================
// Ejecuta un test para una operación
// =============================
bool run_test(unsigned int op, unsigned int n_blocks) {
    std::cout << "====================================================\n";
    std::cout << "  RUN TEST  op=" << op << "  n_blocks=" << n_blocks << "\n";
    std::cout << "====================================================\n";

    // 1) Inicializar entradas
    init_inputs(n_blocks);

    // 2) Calcular referencia en C
    compute_reference(op, n_blocks);

    // 3) Llamar al kernel HLS
    fp32_kernel(op, n_blocks, A, B, Z_hw);

    // 4) Comparar HW vs SW
    return check_results(op, n_blocks);
}

// =============================
// main() del testbench
// =============================
int main() {
    bool all_ok = true;

    // Prueba con distintos nº de bloques (siempre <= MAX_BLOCKS)
    const unsigned int n_blocks1 = 1;
    const unsigned int n_blocks2 = 3;

    // ADD
    all_ok &= run_test(OP_ADD, n_blocks1);
    all_ok &= run_test(OP_ADD, n_blocks2);

    // SUB
    all_ok &= run_test(OP_SUB, n_blocks1);
    all_ok &= run_test(OP_SUB, n_blocks2);

    // MUL
    all_ok &= run_test(OP_MUL, n_blocks1);
    all_ok &= run_test(OP_MUL, n_blocks2);

    // DIV
    all_ok &= run_test(OP_DIV, n_blocks1);
    all_ok &= run_test(OP_DIV, n_blocks2);

    // RCP
    all_ok &= run_test(OP_RCP, n_blocks1);
    all_ok &= run_test(OP_RCP, n_blocks2);

    if (all_ok) {
        std::cout << "========================================\n";
        std::cout << "  TODOS LOS TESTS PASARON (HW == SW)\n";
        std::cout << "========================================\n";
        return 0;   // C TB OK → cosim OK
    } else {
        std::cout << "========================================\n";
        std::cout << "  ALGÚN TEST FALLÓ\n";
        std::cout << "========================================\n";
        return 1;   // C TB falla → cosim marca FAIL
    }
}
