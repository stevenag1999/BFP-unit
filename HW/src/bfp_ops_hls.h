#ifndef BFP_OPS_H
#define BFP_OPS_H

#include <ap_int.h>
#include <cstdint>
#include <cstdlib>
#include "bfp_hls.h"

//*============================================================================
//* HELPER: CLAMP EXPONENTE A RANGO VALIDO
//*============================================================================
template<class Cfg>
static inline uint32_t clamp_exponent(int E_real) {
#pragma HLS INLINE
    int E_biased = E_real + Cfg::bias_bfp;
    if (E_biased < 0) E_biased = 0;
    if (E_biased > (1 << Cfg::we) - 1) E_biased = (1 << Cfg::we) - 1;
    return uint32_t(E_biased);
}

//*============================================================================
//* SUMA DE BLOQUES BFP: Z = A + B
//* - Alinea mantissas según diferencia de exponentes
//* - Suma con signo (complemento a 2)
//* - Normaliza si hay overflow
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> add_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Z{};
    
    //*========================================================================
    //* FASE 1: DETERMINAR EXPONENTE MAYOR Y DIFERENCIA
    //*========================================================================
    int Ea = int(A.exp_shared) - Cfg::bias_bfp;
    int Eb = int(B.exp_shared) - Cfg::bias_bfp;
    
    const BFP_Global<Cfg, Block_size>* X = &A;
    const BFP_Global<Cfg, Block_size>* Y = &B;
    
    if (Eb > Ea) {
        X = &B;
        Y = &A;
        int temp = Ea;
        Ea = Eb;
        Eb = temp;
    }
    
    int E = Ea;
    unsigned diff = unsigned(E - Eb);
    
    Z.exp_shared = clamp_exponent<Cfg>(E);
    
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
    
    //*========================================================================
    //* CASO ESPECIAL: Diferencia muy grande, copiar X directamente
    //*========================================================================
    if (diff > Cfg::wm) {
COPY_LARGER:
        for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            
            Z.sign[i] = (*X).sign[i];
            uint32_t Mx = (*X).mant[i];
            if (Mx > mant_max) Mx = mant_max;
            Z.mant[i] = Mx;
            if (Z.mant[i] == 0u) {
                Z.sign[i] = 0u;
            }
        }
        return Z;
    }
    
    //*========================================================================
    //* FASE 2: SUMA ELEMENTO POR ELEMENTO CON ALINEACION
    //*========================================================================
    bool overflow_flag = false;
    std::array<uint32_t, Block_size> M_temp;
    
ADD_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        // Alinear mantissa del exponente menor con RNE
        uint32_t Mb = (diff > 0) ? helper_rne((*Y).mant[i], int(diff)) : (*Y).mant[i];
        uint32_t Ma = (*X).mant[i];
        
        // Convertir a enteros con signo para la suma
        int32_t Sa = (*X).sign[i] ? -int32_t(Ma) : int32_t(Ma);
        int32_t Sb = (*Y).sign[i] ? -int32_t(Mb) : int32_t(Mb);
        int32_t S  = Sa + Sb;
        
        // Determinar signo y magnitud del resultado
        uint32_t sign_res = (S < 0) ? 1u : 0u;
        uint32_t Mag = uint32_t((S < 0) ? -S : S);
        
        if (Mag == 0u) {
            sign_res = 0u;  // Evitar -0
        }
        
        Z.sign[i] = sign_res;
        M_temp[i] = Mag;
        
        if (M_temp[i] > mant_max) {
            overflow_flag = true;
        }
    }
    
    //*========================================================================
    //* FASE 3: NORMALIZAR SI HAY OVERFLOW
    //*========================================================================
    if (overflow_flag) {
        E += 1;
        Z.exp_shared = clamp_exponent<Cfg>(E);
        
NORMALIZE_OVERFLOW:
        for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            
            uint32_t M_adj = helper_rne(M_temp[i], 1);
            if (M_adj > mant_max) {
                M_adj = mant_max;
            }
            M_temp[i] = M_adj;
            if (M_temp[i] == 0u) {
                Z.sign[i] = 0u;
            }
        }
    } else {
        // Sin overflow, asegurar saturación
SATURATE_MANT:
        for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            
            if (M_temp[i] > mant_max) {
                M_temp[i] = mant_max;
            }
            if (M_temp[i] == 0u) {
                Z.sign[i] = 0u;
            }
        }
    }
    
    //*========================================================================
    //* FASE 4: COPIAR MANTISSAS FINALES
    //*========================================================================
COPY_FINAL_MANT:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        Z.mant[i] = M_temp[i];
    }
    
    // Si todos son cero, exponente = 0
    bool all_zero = true;
CHECK_ALL_ZERO:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        if (M_temp[i] != 0u) {
            all_zero = false;
        }
    }
    
    if (all_zero) {
        Z.exp_shared = 0;
    }
    
    return Z;
}

//*============================================================================
//* RESTA DE BLOQUES BFP: Z = A - B
//* Implementada como A + (-B)
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> sub_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Bneg = B;
    
NEGATE_B:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        if (Bneg.mant[i] == 0u) {
            Bneg.sign[i] = 0u;
        } else {
            Bneg.sign[i] = Bneg.sign[i] ^ 1u;  // Invertir signo
        }
    }
    
    return add_blocks<Cfg, Block_size>(A, Bneg);
}

//*============================================================================
//* MULTIPLICACION DE BLOQUES BFP: Z = A * B
//* - Exponente: Ea + Eb
//* - Mantissas: producto con RNE para reducir a WM bits
//* - Signo: XOR
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> mul_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Z{};
    
    //*========================================================================
    //* EXPONENTE DE SALIDA: Suma de exponentes reales
    //*========================================================================
    int Ea = int(A.exp_shared) - Cfg::bias_bfp;
    int Eb = int(B.exp_shared) - Cfg::bias_bfp;
    int E = Ea + Eb;
    
    Z.exp_shared = clamp_exponent<Cfg>(E);
    
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1u;
    
MUL_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        // Signo = XOR
        uint32_t sign = A.sign[i] ^ B.sign[i];
        
        // Producto de mantissas (64 bits para evitar overflow)
        uint64_t P = uint64_t(A.mant[i]) * uint64_t(B.mant[i]);
        
        // Reducir a escala 2^(WM) con RNE
        uint64_t q = P >> Cfg::wm;
        uint64_t rem = P & ((uint64_t(1) << Cfg::wm) - 1);
        uint64_t half = uint64_t(1) << (Cfg::wm - 1);
        
        bool tie = (rem == half);
        bool gt = (rem > half);
        bool lsb_odd = (q & 1u) != 0;
        
        if (gt || (tie && lsb_odd)) {
            ++q;
        }
        
        uint32_t M = (q > mant_max) ? mant_max : uint32_t(q);
        
        // Normalizar si excede WM+1 bits
        if (M > mant_max) {
            M = helper_rne(M, 1);
            // E incrementado se manejará al final si es necesario
        }
        
        if (M == 0u) sign = 0u;
        
        Z.sign[i] = sign;
        Z.mant[i] = M;
    }
    
    return Z;
}

//*============================================================================
//* RECIPROCO DE BLOQUE BFP: R = 1/B
//* - Exponente: -Eb
//* - Mantissa: División (2^(2*WM)) / mant[i] con RNE
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> rcp_blocks(
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> R{};
    
    const int Eb = int(B.exp_shared) - Cfg::bias_bfp;
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1u;
    
    std::array<uint32_t, Block_size> q{};
    std::array<int, Block_size> Ei{};
    std::array<uint8_t, Block_size> is_zero_den{};
    bool any_nz = false;
    
    //*========================================================================
    //* FASE 1: CALCULAR RECIPROCO PARA CADA ELEMENTO
    //*========================================================================
RCP_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        if (B.mant[i] == 0u) {
            R.sign[i] = B.sign[i];
            q[i] = mant_max;  // Saturar a máximo (representa infinito)
            Ei[i] = 0;
            is_zero_den[i] = 1;
            continue;
        }
        
        is_zero_den[i] = 0;
        R.sign[i] = B.sign[i];  // 1/(+) = +, 1/(-) = -
        
        // División: (2^(2*WM)) / mant[i]
        const uint64_t Num = 1ull << (2 * Cfg::wm);
        const uint64_t Den = (uint64_t)B.mant[i];
        
        uint64_t qq  = Num / Den;
        uint64_t rem = Num % Den;
        
        // RNE para el cociente
        const bool gt = (rem << 1) > Den;
        const bool tie = (rem << 1) == Den;
        const bool lsb_odd = (qq & 1ull) != 0ull;
        
        if (gt || (tie && lsb_odd)) {
            ++qq;
        }
        
        int Erec = -Eb;  // Exponente del recíproco sin normalizar
        
        // Normalizar si la mantissa excede el máximo permitido
        for (int j = 0; j < (int)Cfg::wm + 1; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=15 avg=5
            if (qq <= mant_max) break;
            qq = helper_rne((uint32_t)qq, 1);
            ++Erec;
        }
        
        if (qq > mant_max) qq = mant_max;
        
        q[i] = (uint32_t)qq;
        Ei[i] = Erec;
        any_nz = true;
    }
    
    //*========================================================================
    //* FASE 2: ENCONTRAR EXPONENTE COMPARTIDO MAXIMO
    //*========================================================================
    int Eshared = 0;
    
    if (any_nz) {
        bool first = true;
FIND_MAX_EXP:
        for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            
            if (is_zero_den[i]) continue;
            
            if (first) {
                Eshared = Ei[i];
                first = false;
            } else if (Ei[i] > Eshared) {
                Eshared = Ei[i];
            }
        }
    } else {
        // Todos cero
        R.exp_shared = clamp_exponent<Cfg>(0);
        R.sign.fill(0u);
        R.mant.fill(0u);
        return R;
    }
    
    //*========================================================================
    //* FASE 3: ALINEAR MANTISSAS AL EXPONENTE COMPARTIDO
    //*========================================================================
ALIGN_MANTISSAS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        uint32_t M = q[i];
        
        if (!is_zero_den[i]) {
            int diff = Eshared - Ei[i];
            if (diff > 0) {
                M = helper_rne(M, diff);
            }
        }
        
        if (M > mant_max) M = mant_max;
        if (M == 0u) R.sign[i] = 0u;
        
        R.mant[i] = M;
    }
    
    R.exp_shared = clamp_exponent<Cfg>(Eshared);
    return R;
}

//*============================================================================
//* DIVISION DE BLOQUES BFP: Z = A / B
//* Implementada como A * (1/B) usando rcp_blocks y mul_blocks
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> div_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    auto R = rcp_blocks<Cfg, Block_size>(B);
    return mul_blocks<Cfg, Block_size>(A, R);
}

#endif // BFP_OPS_H