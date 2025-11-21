#ifndef COMMON_FP32_H
#define COMMON_FP32_H

#include <cstdint>

// FP32 Configuration - Same block structure
#define N 16

// For consistency with BFP interface
#define FP32_BLOCK_SIZE N

// Operation codes - Same as BFP for consistency
typedef enum : unsigned int {
    OP_ADD = 2,
    OP_SUB = 3,
    OP_MUL = 4,
    OP_DIV = 5,
    OP_RCP = 6
} fp32_op_t;

// Operation names for display
static const char* FP32_OP_NAMES[] = {
    "ADD",
    "SUB", 
    "MUL",
    "DIV",
    "RCP"
};

#endif // COMMON_FP32_H
