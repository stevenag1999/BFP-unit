# Makefile - BFP Alveo U55C Project
# ECASLab structure: HW/ and SW/

.PHONY: all hw sw clean help check test

#==============================================================================
# DEFAULT TARGET
#==============================================================================
all: check hw sw
	@echo ""
	@echo "===================================================================="
	@echo "Build complete!"
	@echo "===================================================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Program FPGA: make flash"
	@echo "  2. Run test:     make test"
	@echo ""

#==============================================================================
# BUILD HARDWARE
#==============================================================================
hw:
	@echo ""
	@echo "===================================================================="
	@echo "Building Hardware (Kernel)"
	@echo "===================================================================="
	$(MAKE) -C HW

hw-emu:
	@echo ""
	@echo "===================================================================="
	@echo "Building Hardware Emulation"
	@echo "===================================================================="
	$(MAKE) -C HW TARGET=hw_emu

#==============================================================================
# BUILD SOFTWARE
#==============================================================================
sw:
	@echo ""
	@echo "===================================================================="
	@echo "Building Software (Host)"
	@echo "===================================================================="
	$(MAKE) -C SW

#==============================================================================
# FLASH FPGA
#==============================================================================
flash:
	@echo ""
	@echo "===================================================================="
	@echo "Programming Alveo U55C with XCLBIN"
	@echo "===================================================================="
	@if [ ! -f HW/build/bfp_kernel.xclbin ]; then \
		echo "ERROR: XCLBIN not found. Run 'make hw' first."; \
		exit 1; \
	fi
	xbutil program -d 0 -u HW/build/bfp_kernel.xclbin
	@echo "[OK] FPGA programmed"

#==============================================================================
# RUN TEST
#==============================================================================
test: check-xclbin
	@echo ""
	@echo "===================================================================="
	@echo "Running BFP Test on Alveo U55C"
	@echo "===================================================================="
	cd SW && ./build/bfp_host ../HW/build/bfp_kernel.xclbin

#==============================================================================
# CLEAN
#==============================================================================
clean:
	@echo "Cleaning all build artifacts..."
	$(MAKE) -C HW clean
	$(MAKE) -C SW clean
	@echo "[OK] All clean"

clean-hw:
	$(MAKE) -C HW clean

clean-sw:
	$(MAKE) -C SW clean

#==============================================================================
# CHECKS
#==============================================================================
check:
	@echo "Checking environment..."
	@if [ -z "$$XILINX_VITIS" ]; then \
		echo "ERROR: XILINX_VITIS not set"; \
		echo "Please run: source /tools/Xilinx/Vitis/2024.2/settings64.sh"; \
		exit 1; \
	fi
	@if [ -z "$$XILINX_XRT" ]; then \
		echo "ERROR: XILINX_XRT not set"; \
		echo "Please run: source /opt/xilinx/xrt/setup.sh"; \
		exit 1; \
	fi
	@echo "[OK] Environment configured"

check-xclbin:
	@if [ ! -f HW/build/bfp_kernel.xclbin ]; then \
		echo "ERROR: XCLBIN not found. Run 'make hw' first."; \
		exit 1; \
	fi
	@if [ ! -f SW/build/bfp_host ]; then \
		echo "ERROR: Host executable not found. Run 'make sw' first."; \
		exit 1; \
	fi

#==============================================================================
# DEVICE INFO
#==============================================================================
device-info:
	@echo "Checking Alveo devices..."
	xbutil examine
	@echo ""
	xbutil validate -d 0

device-reset:
	@echo "Resetting Alveo device..."
	xbutil reset -d 0

#==============================================================================
# HELP
#==============================================================================
help:
	@echo ""
	@echo "BFP Alveo U55C - Makefile Help"
	@echo "===================================================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Source tools:   source /tools/Xilinx/Vitis/2024.2/settings64.sh"
	@echo "                     source /opt/xilinx/xrt/setup.sh"
	@echo "  2. Build all:      make"
	@echo "  3. Program FPGA:   make flash"
	@echo "  4. Run test:       make test"
	@echo ""
	@echo "Build Targets:"
	@echo "  all        - Build hardware and software (default)"
	@echo "  hw         - Build hardware kernel (slow, 2-4 hours)"
	@echo "  hw-emu     - Build hardware emulation (fast, ~10 min)"
	@echo "  sw         - Build host application"
	@echo ""
	@echo "Run Targets:"
	@echo "  flash      - Program FPGA with bitstream"
	@echo "  test       - Run BFP test suite"
	@echo ""
	@echo "Utility Targets:"
	@echo "  clean      - Clean all build artifacts"
	@echo "  check      - Verify environment setup"
	@echo "  device-info - Show Alveo device information"
	@echo "  device-reset - Reset Alveo device"
	@echo ""
	@echo "Device Configuration:"
	@echo "  Platform: xilinx_u55c_gen3x16_xdma_3_202210_1"
	@echo "  Part:     xcu55c-fsvh2892-2L-e"
	@echo ""
	@echo "For more details, see GUIDE.md"
	@echo ""
