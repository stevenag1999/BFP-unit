#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <type_traits>
#include "bfp_hls.h"
//#include "bfp_hls_opt.h"
#include "bfp_ops_hls.h"

// Top kernel
extern "C" void bfp_kernel(
    const unsigned int operation,
    const unsigned int n_blocks,
    const float* in_fp32_a,
    const unsigned int* in_exp_a,
    const unsigned int* in_sign_a,
    const unsigned int* in_mant_a,
    const unsigned int* in_exp_b,
    const unsigned int* in_sign_b,
    const unsigned int* in_mant_b,
    float* out_fp32,
    unsigned int* out_exp,
    unsigned int* out_sign,
    unsigned int* out_mant
);

// Config
#define WE 5
#define WM 7
#define N  16

enum : unsigned {
    OP_ENCODE = 0,
    OP_DECODE = 1,
    OP_ADD    = 2,
    OP_SUB    = 3,
    OP_MUL    = 4,
    OP_DIV    = 5,
    OP_RCP    = 6
};

// --------- Helpers de impresión y métricas ----------
template<typename T>
static void print_head(const char* tag, const T* x, unsigned len, unsigned k=8) {
    printf("%s [", tag);
    for (unsigned i = 0; i < std::min(k, len); ++i) {
        if constexpr (std::is_floating_point<T>::value)
            printf("%s%.6f", (i?", ":""), x[i]);
        else
            printf("%s%u", (i?", ":""), (unsigned)x[i]);
    }
    if (len > k) printf(", ...");
    printf("]\n");
}

// Métricas: MAE y MAPE (%). Evita dividir por cero en MAPE.
static void metrics(const float* ref, const float* got, unsigned len,
                    double& mae, double& mape){
    double abs_sum = 0.0, ape_sum = 0.0;
    unsigned mape_cnt = 0;
    for (unsigned i = 0; i < len; ++i) {
        const double r = ref[i], g = got[i];
        const double ae = std::fabs(g - r);
        abs_sum += ae;
        if (std::fabs(r) > 1e-12) { ape_sum += ae/std::fabs(r); ++mape_cnt; }
    }
    mae = abs_sum / double(len);
    mape = (mape_cnt ? (ape_sum / double(mape_cnt)) * 100.0 : 0.0);
}

int main() {
    const unsigned n_blocks = 2;
    const unsigned sz = N * n_blocks;

    // Buffers host
    std::vector<float>    in_fp32_a(sz, 0.f);
    std::vector<unsigned> in_exp_a(n_blocks, 0), in_sign_a(sz, 0), in_mant_a(sz, 0);

    std::vector<unsigned> in_exp_b(n_blocks, 0), in_sign_b(sz, 0), in_mant_b(sz, 0);

    std::vector<float>    out_fp32(sz, 0.f);
    std::vector<unsigned> out_exp(n_blocks, 0), out_sign(sz, 0), out_mant(sz, 0);

    // Dummies (para m_axi válidos)
    std::vector<float>    dummy_fp32(sz, 0.f);
    std::vector<unsigned> dummy_exp(n_blocks, 0), dummy_sign(sz, 0), dummy_mant(sz, 0);

    // Datos A y B en dos bloques
    float A0[N] = {
        12.35f,  6.50f, 10.20f,  6.60f,  8.80f,  2.56f, 11.11f,  8.00f,
         5.45f,  9.99f,  0.15f, 18.00f,  3.80f, 90.10f, 14.00f, 10.00f
    };
    float A1[N] = {
         0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
         8.0f, 9.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f
    };
    float B0[N] = {
        -2.0f, 0.0f, -2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
         3.0f, 3.0f,  5.0f, 3.0f, 6.0f, 3.0f, 8.0f, 2.0f
    };
    float B1[N] = {
        15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
         7.0f,  6.0f,  5.0f,  4.0f,  3.0f,  2.0f, 1.0f, 0.5f
    };

    // Carga A para ENCODE
    std::memcpy(&in_fp32_a[0], A0, sizeof(float)*N);
    std::memcpy(&in_fp32_a[N], A1, sizeof(float)*N);

    // ===== ENCODE A =====
    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32_a.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               out_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());

    in_exp_a = out_exp;
    for (unsigned i = 0; i < sz; ++i) {
        in_sign_a[i] = out_sign[i];
        in_mant_a[i] = out_mant[i];
    }

    // Carga B para ENCODE
    std::memcpy(&in_fp32_a[0], B0, sizeof(float)*N);
    std::memcpy(&in_fp32_a[N], B1, sizeof(float)*N);

    // ===== ENCODE B =====
    std::fill(out_exp.begin(), out_exp.end(), 0u);
    std::fill(out_sign.begin(), out_sign.end(), 0u);
    std::fill(out_mant.begin(), out_mant.end(), 0u);

    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32_a.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               out_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());

    in_exp_b = out_exp;
    for (unsigned i = 0; i < sz; ++i) {
        in_sign_b[i] = out_sign[i];
        in_mant_b[i] = out_mant[i];
    }

    // Referencias FP32 completas por operación
    std::vector<float> ref_add(sz), ref_sub(sz), ref_mul(sz), ref_div(sz), ref_rcp(sz);
    // Reconstruimos A y B originales en FP32 para referencia (usamos los arrays A0/A1/B0/B1)
    std::vector<float> A_fp(sz), B_fp(sz);
    std::memcpy(&A_fp[0], A0, sizeof(float)*N);
    std::memcpy(&A_fp[N], A1, sizeof(float)*N);
    std::memcpy(&B_fp[0], B0, sizeof(float)*N);
    std::memcpy(&B_fp[N], B1, sizeof(float)*N);

    for (unsigned i = 0; i < sz; ++i) {
        ref_add[i] = A_fp[i] + B_fp[i];
        ref_sub[i] = A_fp[i] - B_fp[i];
        ref_mul[i] = A_fp[i] * B_fp[i];
        ref_div[i] = (std::fabs(B_fp[i]) > 1e-30f) ? (A_fp[i] / B_fp[i]) : 0.f;
        ref_rcp[i] = (std::fabs(B_fp[i]) > 1e-30f) ? (1.0f / B_fp[i]) : 0.f;
    }

    auto run_and_decode_compare = [&](unsigned op, const char* name, const std::vector<float>& ref){
        std::fill(out_exp.begin(),  out_exp.end(),  0u);
        std::fill(out_sign.begin(), out_sign.end(), 0u);
        std::fill(out_mant.begin(), out_mant.end(), 0u);
        std::fill(out_fp32.begin(), out_fp32.end(), 0.f);

        // Ejecuta Z = op(A,B) (o RCP(B) si aplica)
        if (op == OP_RCP) {
            bfp_kernel(OP_RCP, n_blocks,
                       dummy_fp32.data(),
                       dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
                       in_exp_b.data(), in_sign_b.data(), in_mant_b.data(),
                       dummy_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());
        } else {
            bfp_kernel(op, n_blocks,
                       dummy_fp32.data(),
                       in_exp_a.data(), in_sign_a.data(), in_mant_a.data(),
                       in_exp_b.data(), in_sign_b.data(), in_mant_b.data(),
                       dummy_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());
        }

        printf("\n[%s] exp_shared (block0, block1) = (%u, %u)\n", name, out_exp[0], out_exp[1]);
        print_head("Z.mant[blk0]", &out_mant[0], N);
        print_head("Z.mant[blk1]", &out_mant[N], N);

        // Decodifica Z -> FP32
        bfp_kernel(OP_DECODE, n_blocks,
                   dummy_fp32.data(),
                   out_exp.data(), out_sign.data(), out_mant.data(),
                   dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
                   out_fp32.data(), dummy_exp.data(), dummy_sign.data(), dummy_mant.data());

        // ---- Imprime resultados (primeros 8 por bloque) ----
        print_head("Z.decoded[blk0]", &out_fp32[0], N);
        print_head("Z.decoded[blk1]", &out_fp32[N], N);

        // ---- Métricas por bloque ----
        double mae0=0.0, mape0=0.0, mae1=0.0, mape1=0.0;
        metrics(&ref[0],        &out_fp32[0], N, mae0, mape0);
        metrics(&ref[N], &out_fp32[N],        N, mae1, mape1);
        printf("[METRICS] %s -> blk0: MAE=%.6f  MAPE=%.3f%%   |   blk1: MAE=%.6f  MAPE=%.3f%%\n",
               name, mae0, mape0, mae1, mape1);

        // ---- Resumen de actividad ----
        unsigned nz0=0, nz1=0;
        for (unsigned i=0;i<N;++i){ nz0 += (out_mant[i]!=0); nz1 += (out_mant[N+i]!=0); }
        printf("[OK] %s -> nz(blk0)=%u, nz(blk1)=%u\n", name, nz0, nz1);
    };

    // Ejecuta y compara con referencia
    run_and_decode_compare(OP_ADD, "ADD", ref_add);
    run_and_decode_compare(OP_SUB, "SUB", ref_sub);
    run_and_decode_compare(OP_MUL, "MUL", ref_mul);
    run_and_decode_compare(OP_DIV, "DIV", ref_div);
    run_and_decode_compare(OP_RCP, "RCP", ref_rcp);

    // Verificación final: NaNs
    unsigned n_nan = 0;
    for (float v : out_fp32) n_nan += std::isnan(v);
    printf("\n[DECODE check] NaNs=%u\n", n_nan);

    printf("\nTB completed (n_blocks=%u, N=%u).\n", n_blocks, N);
    return 0;
}
