#ifndef BFP_H
#define BFP_H

#include <cstdint>
#include <limits>
#include <algorithm>
#include <array>
#include <cmath>
#include <climits> 


//*CONFIGURACION DE BIAS PARA FORMATO BFP 
template<int WE, int WM>
struct BFP_bias {

    static constexpr int we = WE;                      
    static constexpr int wm = WM;                      
    static constexpr int bias_bfp = (1 << (WE - 1)) -1;


};

//* ROUND TO NEAREST EVEN (SHIFT RIGHT)
static inline uint32_t helper_rne(uint32_t x, int shift) {
    // SHIT LEFT
    if (shift <= 0) {
        int s = -shift;
        if (s >= 32) return 0u;       // LO QUE SE SALE DE RANGO
        return (s == 0) ? x : (x << s);
    }

    //SHIFT RIGHT
    if (shift >= 32) return 0u;       // ES LO QUE TODO SE DESCARTA

    uint32_t q    = x >> shift;
    uint32_t rem  = x & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);

    // RNE: NEAREST / TIES-TO-EVEN
    if (rem > half || (rem == half && (q & 1u))) {++q; }
    return q;
}

//* ------------------------------------------------------------------------
//* GLOBLALIZAR EL EXPONENTE POR BLOQUE

template<class Cfg, std::size_t Block_size>
struct BFP_Global{

    uint32_t exp_shared; // EXPONENTE COMPARTIDO E CON BIAS
    std::array<uint32_t, Block_size> sign;  
    std::array<uint32_t, Block_size> mant;
    std::array<int, Block_size> delta; // DELTA


    // RECONSTRUIR VALORES A FP32 NUEVAMENTE PARA VALIDAR 
    float rebuild_FP32(std::size_t i) const {
        if (i >= Block_size) return 0.0f;

        if (exp_shared == 0 && mant[i] == 0) return 0.0f;
        
        int   exp_unbiased = int(exp_shared) - Cfg::bias_bfp;
        float mant_val     = float(mant[i]) / float(1u << Cfg::wm); //SIN 1 IMPLICITO
        float value        = std::ldexp(mant_val, exp_unbiased);     
        return sign[i] ? -value : value;
    }

    float rebuid_FP32(std::size_t i) const {  // no tocar tu TB
        return rebuild_FP32(i);
    }
     
};

//* CODIFICACION GLOBLAL CALCULANDO EMAX Y USANDO RNE

template< class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> encode_block(const std::array<float, Block_size>& xs){

    BFP_Global<Cfg, Block_size> out{};

    //* PARA HALLAR EL EXPONENTE MAYOR
    int Emax = std::numeric_limits<int>::min();
    for (std::size_t i = 0; i < Block_size; i++)
    {
        float num_fp32 = xs[i];
        //INTERPRETACION DE FLOAT A 32 BITS
        union {float f; uint32_t u; } u = {num_fp32};

        int exp_fp32 = int((u.u >> 23) & 0xFF); // EXPONENTE FP32
        if (exp_fp32 == 0) continue;
        
        // EXPONENTE REAL SIN BIAS
        int exp_unbiased = exp_fp32 - 127; 
        if (exp_unbiased > Emax) Emax = exp_unbiased; //ACTUALIZO EXPONENTE MAYOR
    }

    // VALIDAR SI EL BLOQUE SON TODOS CEROS
    if (Emax == std::numeric_limits<int>::min()) {
        out.exp_shared = 0;
        out.sign.fill(0);
        out.mant.fill(0);
        out.delta.fill(0); // DELTA
        return out; // devolver bloque "cero"
    }   
    
    //* CALCULO DEL EXPONENTE COMPARTIDO PARA BFP CON BIAS
    int exp_shared_bfp = Emax + Cfg::bias_bfp;
    // PARA ASEGURAR QUE CABE EN WE BITS
    if (exp_shared_bfp < 0 ) exp_shared_bfp = 0;
    if (exp_shared_bfp > (1 << Cfg::we) - 1) exp_shared_bfp = (1 << Cfg::we) - 1;
    out.exp_shared = uint32_t(exp_shared_bfp);

    //* CUANTIZAR Y CODIFICAR CADA ELEMENTO CON EL EXPONENTE MAX (SHIFT & RNE)
    // PARA LIMPIAR A WE
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;

    for (std::size_t i = 0; i < Block_size; i++)
    {
        float num_fp32 = xs[i];
        if (num_fp32 == 0.0f){
            out.sign[i] = 0; // PARA GUARDAR COMO CERO
            out.mant[i] = 0;
            out.delta[i] = 0; // DELTA
            continue;
        }

        
        //* EXTRAER LOS CAMPOS DE FP32
        union {float f; uint32_t u; } u = {num_fp32};
        // FLOAT COMO ENTEROS DE 32 BITS
        uint32_t s = (u.u >> 31) & 0x1;
        int exp_fp32 = int((u.u >> 23) & 0xFF); //EXP FP32
        if (exp_fp32 == 0) {
            out.sign[i]  = 0;
            out.mant[i]  = 0;
            out.delta[i] = 0;
            continue;
        }

        // EXPONENTE REAL Y DELTA_i    Δ_i = Emax - Ereal_i
        int Ereal_i = exp_fp32 - 127;
        long long diff = (long long)Emax - (long long)Ereal_i; // CALCULO DE DELTA        
        int delta_i;
        if      (diff > INT_MAX) delta_i = INT_MAX;
        else if (diff < INT_MIN) delta_i = INT_MIN;
        else                     delta_i = (int)diff;
        out.delta[i] = delta_i;
        
        //if (exp_fp32 == 0){out.sign[i] = 0; out.mant[i] = 0; continue; }
        
        //* CONSTRUIR MANT DE 24 BITS CON 1 IMPLICITO
        uint32_t mant24 = (u.u & 0x7FFFFF) | (1u << 23); // 1.MANT
        long long st64 = (long long)(23 - Cfg::wm) + (long long)delta_i;
        int shift_total;
        if (st64 > INT_MAX) shift_total = INT_MAX;
        else if (st64 < INT_MIN) shift_total = INT_MIN;
        else shift_total = (int)st64;
        
        //* REDUCCION DE MANTISA A WM BITS & ALINEAR EXPONENTE COMPART
        uint32_t mant_reduced;

        if (shift_total >= 31) mant_reduced = 0u; //CASO DE UNDERFLOW POR DESPLAZAR DEMASIADO
        else if (shift_total >= 0) mant_reduced = helper_rne(mant24, shift_total);
        else {
            int sleft = -shift_total;
            mant_reduced = (sleft >= 32) ? 0u : (mant24 << sleft);// CASO QUE NO DEBE PASAR
         }

        if(mant_reduced > mant_max) mant_reduced = mant_max;

        //* GUARDAR SIGN Y MANTISA WM
        out.sign[i] = s;
        out.mant[i] = mant_reduced;
        
    }
    
    return out;
    
}  



#endif 
