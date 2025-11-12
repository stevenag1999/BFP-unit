#!/bin/bash
# scripts/setup_env.sh - Setup environment for BFP Alveo project

set -e

echo "================================================================="
echo "BFP Alveo U55C - Environment Setup"
echo "================================================================="
echo ""

#==============================================================================
# Check for required tools
#==============================================================================
echo "[1/4] Checking required tools..."

if ! command -v vivado &> /dev/null; then
    echo "ERROR: Vivado not found"
    echo "Please source: /tools/Xilinx/Vitis/2024.2/settings64.sh"
    exit 1
fi

if ! command -v v++ &> /dev/null; then
    echo "ERROR: Vitis compiler (v++) not found"
    echo "Please source: /tools/Xilinx/Vitis/2024.2/settings64.sh"
    exit 1
fi

if [ -z "$XILINX_XRT" ]; then
    echo "ERROR: XRT not configured"
    echo "Please source: /opt/xilinx/xrt/setup.sh"
    exit 1
fi

echo "[OK] All tools found"

#==============================================================================
# Check Alveo device
#==============================================================================
echo ""
echo "[2/4] Checking Alveo device..."

if ! command -v xbutil &> /dev/null; then
    echo "ERROR: xbutil not found"
    echo "XRT may not be properly installed"
    exit 1
fi

DEVICE_COUNT=$(xbutil examine 2>/dev/null | grep -c "xilinx_u55c" || echo "0")

if [ "$DEVICE_COUNT" -eq "0" ]; then
    echo "WARNING: No Alveo U55C device detected"
    echo "This is OK for build-only mode"
else
    echo "[OK] Found $DEVICE_COUNT Alveo device(s)"
    xbutil examine -d 0 | grep "Name" | head -1
fi

#==============================================================================
# Setup directories
#==============================================================================
echo ""
echo "[3/4] Setting up directories..."

mkdir -p HW/build/logs
mkdir -p SW/build

echo "[OK] Directories created"

#==============================================================================
# Environment summary
#==============================================================================
echo ""
echo "[4/4] Environment Summary"
echo "----------------------------------------------------------------"
echo "Vitis:     $XILINX_VITIS"
echo "XRT:       $XILINX_XRT"
echo "Platform:  xilinx_u55c_gen3x16_xdma_3_202210_1"
echo "Part:      xcu55c-fsvh2892-2L-e"
echo "----------------------------------------------------------------"

echo ""
echo "================================================================="
echo "Setup Complete!"
echo "================================================================="
echo ""
echo "Next steps:"
echo "  1. Build hardware:  make hw          (takes 2-4 hours)"
echo "  2. Build software:  make sw          (takes ~10 seconds)"
echo "  3. Program FPGA:    make flash"
echo "  4. Run test:        make test"
echo ""
echo "For quick testing, use emulation:"
echo "  make hw-emu    # Fast, ~10 minutes"
echo ""
echo "For help:"
echo "  make help"
echo "  cat GUIDE.md"
echo ""
