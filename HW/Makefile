# HW/Makefile - Build BFP kernel for Alveo U55C
# ECASLab configuration for xcu55c-fsvh2892-2L-e

#==============================================================================
# ALVEO U55C CONFIGURATION
#==============================================================================
PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1
DEVICE := xcu55c-fsvh2892-2L-e
TARGET := hw

# Use hw_emu for quick testing, hw for bitstream
# TARGET := hw_emu

#==============================================================================
# PATHS
#==============================================================================
SRC_DIR := src
BUILD_DIR := build
TCL_DIR := tcl

KERNEL_NAME := bfp_kernel
KERNEL_SRC := $(SRC_DIR)/$(KERNEL_NAME).cpp
KERNEL_XO := $(BUILD_DIR)/$(KERNEL_NAME).xo
XCLBIN := $(BUILD_DIR)/$(KERNEL_NAME).xclbin

#==============================================================================
# VITIS COMPILER
#==============================================================================
VPP := v++

# Compiler flags
VPP_COMMON_FLAGS := \
	-t $(TARGET) \
	--platform $(PLATFORM) \
	--save-temps \
	--temp_dir $(BUILD_DIR)/_x \
	--log_dir $(BUILD_DIR)/logs

# Compilation flags (C++ to XO)
VPP_COMPILE_FLAGS := $(VPP_COMMON_FLAGS) \
	-c \
	-I$(SRC_DIR) \
	-g \
	--optimize 3

# Link flags (XO to XCLBIN)
VPP_LINK_FLAGS := $(VPP_COMMON_FLAGS) \
	-l \
	--optimize 3 \
	--connectivity.nk $(KERNEL_NAME):1 \
	--kernel_frequency 250

#==============================================================================
# TARGETS
#==============================================================================
.PHONY: all kernel xclbin clean help

all: xclbin

help:
	@echo "Makefile for BFP Kernel on Alveo U55C"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build complete xclbin (default)"
	@echo "  kernel   - Compile kernel to .xo"
	@echo "  xclbin   - Link .xo to .xclbin"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Device:   $(DEVICE)"
	@echo "  Target:   $(TARGET)"
	@echo ""
	@echo "Usage:"
	@echo "  make            # Build hardware (slow, ~2-4 hours)"
	@echo "  make TARGET=hw_emu  # Build emulation (fast, ~10 min)"

#==============================================================================
# COMPILE: C++ -> XO
#==============================================================================
kernel: $(KERNEL_XO)

$(KERNEL_XO): $(KERNEL_SRC) $(SRC_DIR)/bfp_hls.h $(SRC_DIR)/bfp_ops_hls.h
	@echo ""
	@echo "===================================================================="
	@echo "Compiling BFP kernel to .xo"
	@echo "===================================================================="
	@mkdir -p $(BUILD_DIR)/logs
	$(VPP) $(VPP_COMPILE_FLAGS) -k $(KERNEL_NAME) -o $@ $(KERNEL_SRC)
	@echo "[OK] Kernel compiled: $@"

#==============================================================================
# LINK: XO -> XCLBIN
#==============================================================================
xclbin: $(XCLBIN)

$(XCLBIN): $(KERNEL_XO)
	@echo ""
	@echo "===================================================================="
	@echo "Linking to XCLBIN (this may take 2-4 hours for hw target)"
	@echo "===================================================================="
	$(VPP) $(VPP_LINK_FLAGS) -o $@ $(KERNEL_XO)
	@echo ""
	@echo "===================================================================="
	@echo "[SUCCESS] XCLBIN ready: $@"
	@echo "===================================================================="

#==============================================================================
# CLEAN
#==============================================================================
clean:
	rm -rf $(BUILD_DIR)
	@echo "[OK] Build directory cleaned"

#==============================================================================
# INFO
#==============================================================================
info:
	@echo "Build Configuration:"
	@echo "  Platform:    $(PLATFORM)"
	@echo "  Device:      $(DEVICE)"
	@echo "  Target:      $(TARGET)"
	@echo "  Kernel:      $(KERNEL_NAME)"
	@echo "  Source:      $(KERNEL_SRC)"
	@echo "  Output XO:   $(KERNEL_XO)"
	@echo "  Output XCLBIN: $(XCLBIN)"
