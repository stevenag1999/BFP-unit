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
//* HELPER: CALCULAR DELTA DESDE MANTISA
//* Delta = WM - posición_MSB
//*============================================================================
template<class Cfg>
static inline uint32_t calculate_delta_from_mant(uint32_t mant) {
#pragma HLS INLINE
    
    if (mant == 0) return 0;
    
    // Encontrar el bit más significativo (MSB)
    int msb_pos = -1;
    
    #pragma HLS UNROLL factor=8
    for (int b = Cfg::wm; b >= 0; --b) {
        if ((mant >> b) & 0x1) {
            msb_pos = b;
            break;
        }
    }
    
    if (msb_pos < 0) return 0;
    
    // Delta = cuántos bits "perdidos" desde la precisión completa
    int delta = Cfg::wm - msb_pos;
    
    return uint32_t(delta);
}

//*============================================================================
//* SUMA DE BLOQUES BFP: Z = A + B
//* - Alinea mantissas según diferencia de exponentes
//* - Suma con signo (complemento a 2)
//* - Normaliza si hay overflow
//* USA DELTAS PARA DETERMINAR EXPONENTES REALES
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> add_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Z{};
    
    //*========================================================================
    //* FASE 1: DETERMINAR EXPONENTE REALES DE LOS BLOQUES
    //*========================================================================
    int Ea_shared = int(A.exp_shared) - Cfg::bias_bfp;
    int Eb_shared = int(B.exp_shared) - Cfg::bias_bfp;
    
    //*========================================================================
    //* FASE 2: ENCONTRAR EXP MAX CONSIDERANDO DELTAS
    //*========================================================================
    int Emax = std::numeric_limits<int>::min();

FIND_EMAX:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        // Exponente real de elemento i en A
        if (A.mant[i] != 0u) {
            int exp_A_i = Ea_shared - int(A.delta[i]);
            if (exp_A_i > Emax) Emax = exp_A_i;
        }
        
        // Exponente real de elemento i en B
        if (B.mant[i] != 0u) {
            int exp_B_i = Eb_shared - int(B.delta[i]);
            if (exp_B_i > Emax) Emax = exp_B_i;
        }
    }
    
    if (Emax == std::numeric_limits<int>::min()) {
        //[Todos ceros]
        Z.exp_shared = 0;
        Z.sign.fill(0);
        Z.mant.fill(0);
        Z.delta.fill(0);
        return Z;
    }
    
    Z.exp_shared = clamp_exponent<Cfg>(Emax);
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;

    //*========================================================================
    //* FASE 3: SUMA CON ALINEACION POR ELEMENTO USANDO DELTAS
    //*========================================================================

    bool overflow_flag = false;
    std::array<uint32_t, Block_size> M_temp;
    
ADD_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16

        //*========================================================================
        //* MANEJO DE CASOS ESPECIALES
        //*========================================================================
        // PROPAGACIÓN DE NaN/Inf
        bool is_inf_A = (A.mant[i] == mant_max) && (A.delta[i] == 0);
        bool is_inf_B = (B.mant[i] == mant_max) && (B.delta[i] == 0);
        bool is_nan_A = (A.mant[i] == mant_max - 1) && (A.delta[i] == 0);
        bool is_nan_B = (B.mant[i] == mant_max - 1) && (B.delta[i] == 0);
        
        // Si alguno es NaN, el resultado es NaN
        if (is_nan_A || is_nan_B) {
            Z.sign[i] = 0u;
            M_temp[i] = mant_max - 1;  // NaN
            continue;
        }
        
        // Si ambos son Inf con signos opuestos: NaN
        if (is_inf_A && is_inf_B) {
            bool same_sign = (A.sign[i] == B.sign[i]);
            if (!same_sign) {
                Z.sign[i] = 0u;
                M_temp[i] = mant_max - 1;  // Inf - Inf = NaN
                continue;
            } else {
                Z.sign[i] = A.sign[i];
                M_temp[i] = mant_max;  // Inf + Inf = Inf
                continue;
            }
        }
        
        // Si uno es Inf, el resultado es Inf
        if (is_inf_A) {
            Z.sign[i] = A.sign[i];
            M_temp[i] = mant_max;
            continue;
        }
        if (is_inf_B) {
            Z.sign[i] = B.sign[i];
            M_temp[i] = mant_max;
            continue;
        }
        //*========================================================================
        //Unshift mantisas para recuperar valores completos
        int delta_A_i = int(A.delta[i]);
        int delta_B_i = int(B.delta[i]);
        
        uint32_t Ma_full = A.mant[i] << delta_A_i;
        uint32_t Mb_full = B.mant[i] << delta_B_i;

        // Exponente real del elemento i en A y B
        int exp_A_i = Ea_shared - delta_A_i;
        int exp_B_i = Eb_shared - delta_B_i;
        
        // Shift necesario para alinear cada elemento a Emax
        int shift_A = Emax - exp_A_i;
        int shift_B = Emax - exp_B_i;
        
        // Alinear mantisas
        uint32_t Ma = (shift_A > 0) ? helper_rne(Ma_full, shift_A) : Ma_full;
        uint32_t Mb = (shift_B > 0) ? helper_rne(Mb_full, shift_B) : Mb_full;
        
        // Convertir a enteros con signo para la suma
        int32_t Sa = A.sign[i] ? -int32_t(Ma) : int32_t(Ma);
        int32_t Sb = B.sign[i] ? -int32_t(Mb) : int32_t(Mb);
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
    //* FASE 4: NORMALIZAR SI HAY OVERFLOW
    //*========================================================================
    if (overflow_flag) {
        Emax += 1;
        Z.exp_shared = clamp_exponent<Cfg>(Emax);
        
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
    //* FASE 5: COPIAR MANTISSAS FINALES Y CALCULO DE DELTAS
    //*========================================================================
COPY_AND_SET_DELTA:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16

        Z.mant[i] = M_temp[i];
        Z.delta[i] = calculate_delta_from_mant<Cfg>(M_temp[i]);

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

//*============================================================================*/
//* MULTIPLICACION DE BLOQUES BFP: Z = A * B                                   */
//* - Usa exponentes reales (exp_shared - delta) de cada elemento              */
//* - Mantissas: producto con RNE para reducir a WM bits                       */
//* - Signo: XOR                                                               */
//* - Delta para salidas inline                                                */
//*============================================================================*/
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> mul_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Z{};
    
    //*========================================================================*/
    //* FASE 1: CALCULAR EXPONENTES REALES DE BLOQUES                          */
    //*========================================================================*/
    int Ea_shared = int(A.exp_shared) - Cfg::bias_bfp;
    int Eb_shared = int(B.exp_shared) - Cfg::bias_bfp;

    //*========================================================================*/
    //* FASE 2: ENCONTRAR EXPONENTE MÁXIMO DEL RESULTADO                       */
    //*========================================================================*/
    int Emax = std::numeric_limits<int>::min();

FIND_EMAX_MUL:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        if (A.mant[i] != 0u && B.mant[i] != 0u) {
            int exp_A_i = Ea_shared - int(A.delta[i]);
            int exp_B_i = Eb_shared - int(B.delta[i]);
            int exp_prod = exp_A_i + exp_B_i;
            
            if (exp_prod > Emax) {
                Emax = exp_prod;
            }
        }
    }
    
    if (Emax == std::numeric_limits<int>::min()) {
        Z.exp_shared = 0;
        Z.sign.fill(0);
        Z.mant.fill(0);
        Z.delta.fill(0);
        return Z;
    }
    
    Z.exp_shared = clamp_exponent<Cfg>(Emax);
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1u;

    //*========================================================================*/
    //* FASE 3: MULTIPLICACIÓN ELEMENTO POR ELEMENTO                           */
    //*========================================================================*/
    
MUL_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        // Signo = XOR
        uint32_t sign = A.sign[i] ^ B.sign[i];

        //*========================================================================
        //* MANEJO DE CASOS ESPECIALES
        //*========================================================================
        // PROPAGACIÓN DE NaN/Inf
        bool is_inf_A = (A.mant[i] == mant_max) && (A.delta[i] == 0);
        bool is_inf_B = (B.mant[i] == mant_max) && (B.delta[i] == 0);
        bool is_nan_A = (A.mant[i] == mant_max - 1) && (A.delta[i] == 0);
        bool is_nan_B = (B.mant[i] == mant_max - 1) && (B.delta[i] == 0);
        bool is_zero_A = (A.mant[i] == 0u);
        bool is_zero_B = (B.mant[i] == 0u);
        
        // Si alguno es NaN, resultado es NaN
        if (is_nan_A || is_nan_B) {
            Z.sign[i] = 0u;
            Z.mant[i] = mant_max - 1;
            Z.delta[i] = 0u;
            continue;
        }
        
        // Inf * 0 = NaN
        if ((is_inf_A && is_zero_B) || (is_zero_A && is_inf_B)) {
            Z.sign[i] = 0u;
            Z.mant[i] = mant_max - 1;
            Z.delta[i] = 0u;
            continue;
        }
        
        // Inf * (no-cero) = Inf
        if (is_inf_A || is_inf_B) {
            Z.sign[i] = sign;
            Z.mant[i] = mant_max;
            Z.delta[i] = 0u;
            continue;
        }

        // Cero normal
        if (is_zero_A || is_zero_B) {
            Z.sign[i] = 0u;
            Z.mant[i] = 0u;
            Z.delta[i] = 0u;
            continue;
        }
        //*========================================================================
        // Unshift las mantisas para recuperar valores completos
        int delta_A_i = int(A.delta[i]);
        int delta_B_i = int(B.delta[i]);
        uint32_t Ma_full = A.mant[i] << delta_A_i;
        uint32_t Mb_full = B.mant[i] << delta_B_i;
        
        // Producto de mantissas (64 bits para evitar overflow)
        uint64_t P = uint64_t(Ma_full) * uint64_t(Mb_full);
        
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
        
        //Calcular exponente del producto y alinear a Emax
        int exp_A_i = Ea_shared - delta_A_i;
        int exp_B_i = Eb_shared - delta_B_i;
        int exp_prod = exp_A_i + exp_B_i;

        // Delta = Emax - exponente del productox
        int shift = Emax - exp_prod;

        uint64_t M_shifted;

        if (shift > 0) {
            uint64_t q_shift = q >> shift;
            uint64_t rem_shift = q & ((1ull << shift) - 1);
            uint64_t half_shift = 1ull << (shift - 1);
            
            bool tie_shift = (rem_shift == half_shift);
            bool gt_shift = (rem_shift > half_shift);
            bool lsb_odd_shift = (q_shift & 1ull) != 0;
            
            if (gt_shift || (tie_shift && lsb_odd_shift)) {
                ++q_shift;
            }
            M_shifted = q_shift;

        } else if (shift < 0) {
            // Shift left (caso raro)
            M_shifted = q << (-shift);
        } else{
            M_shifted = q;
        }
        
        uint32_t M;
        if (M_shifted > mant_max) {
            M = mant_max;
        } else {
            M = uint32_t(M_shifted);
        }

        // Calculo de delta
        if (M == 0u) {
            sign = 0u;
        } 

        Z.sign[i] = sign;
        Z.mant[i] = M;
        Z.delta[i] = calculate_delta_from_mant<Cfg>(M);
    }
    
    return Z;
}

//*============================================================================
//* RECIPROCO DE BLOQUE BFP: R = 1/B CON DELTA
//* - Usa exponentes reales de cada elemento (exp_shared - delta)
//* - Calcula 1/B para cada elemento: exp_recip = -exp_B_i
//* - Mantissa: División (2^(2*WM)) / mant[i] con RNE
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> rcp_blocks(
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> R{};
    
    const int Eb_shared = int(B.exp_shared) - Cfg::bias_bfp;
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
            Ei[i] = (1 << Cfg::we) - 1 - Cfg::bias_bfp;;
            is_zero_den[i] = 1;
            continue;
        }
        
        is_zero_den[i] = 0;
        R.sign[i] = B.sign[i];  // 1/(+) = +, 1/(-) = -

        // Recuperar mantisa completa (unshift con delta)
        int delta_B_i = int(B.delta[i]);
        uint64_t Mb_full = (uint64_t)B.mant[i] << delta_B_i;
        
        // División: (2^(2*WM)) / mant[i]
        const uint64_t Num = 1ull << (2 * Cfg::wm);
        const uint64_t Den = Mb_full;
        
        uint64_t qq  = Num / Den;
        uint64_t rem = Num % Den;
        
        // RNE para el cociente
        const bool gt = (rem << 1) > Den;
        const bool tie = (rem << 1) == Den;
        const bool lsb_odd = (qq & 1ull) != 0ull;
        
        if (gt || (tie && lsb_odd)) {
            ++qq;
        }
        
        // Exponente del recíproco = -exponente_real_del_elemento
        int exp_B_i = Eb_shared - int(B.delta[i]);
        int Erec = -exp_B_i;
        
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
        // .Todos cero
        R.exp_shared = clamp_exponent<Cfg>(0);
        R.sign.fill(0u);
        R.mant.fill(0u);
        R.delta.fill(0u);
        return R;
    }
    
    //*========================================================================
    //* FASE 3: ALINEAR MANTISSAS AL EXPONENTE COMPARTIDO Y CALCULO DE DELTA
    //*========================================================================
ALIGN_AND_CALC_DELTA:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        uint32_t M = q[i];
        
        if (!is_zero_den[i]) {
            int diff = Eshared - Ei[i];
            // Delta = diferencia de exponentes
            //R.delta[i] = uint32_t(diff);
            if (diff > 0) {
                M = helper_rne(M, diff);
            }
        }/*else {
            R.delta[i] = 0u;
        }*/
        
        if (M > mant_max) M = mant_max;
        if (M == 0u) R.sign[i] = 0u;
        
        R.mant[i] = M;
        R.delta[i] = calculate_delta_from_mant<Cfg>(M);
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
