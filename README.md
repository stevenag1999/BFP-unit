# Block Floating-Point (BFP) Unit in FPGA for Hardware Acceleration using HLS

This repository implements a **Block Floating-Point (BFP)** arithmetic unit on an FPGA, using **Vitis HLS 2024.2**.  
The project includes the **HLS kernel**, a **C++ host application** using **XRT**, reference CPU models and **experimental results**.

> Developed as part of a Final Graduation Project in Electronic Engineering at Instituto Tecnológico de Costa Rica (TEC).

---

## 1. Overview

Block Floating-Point (BFP) is a numeric format where a **group of values (a block)** shares a single exponent, while each element stores its own fixed-point mantissa. This allows:

- Lower **memory footprint** than FP32.
- Better **throughput** and **resource efficiency** on FPGA.
- Controlled **loss of precision**, suitable for workloads that tolerate quantization.

This project provides a **hardware BFP unit** capable of performing basic arithmetic operations on blocks of values, and compares its behavior and performance against standard FP32 arithmetic.

---

## 2. Main Features

- **Parametric BFP format**
  - Shared exponent per block.
  - Configurable exponent/mantissa widths (default: `WE = 5` bits, `WM = 7` bits).
  - Default block size: `N = 16` elements.

- **Supported operations**
  - `ENCODE` – convert FP32 vectors to BFP representation.
  - `DECODE` – convert BFP vectors back to FP32.
  - `ADD`    – block-wise addition.
  - `SUB`    – block-wise subtraction.
  - `MUL`    – block-wise multiplication.
  - `DIV`    – block-wise division.
  - `RCP`    – block-wise reciprocal.

- **Target platform**
  - **FPGA:** Xilinx Alveo **U55C** (`xcu55c-fsvh2892-2L-e`).
  - **Toolchain:** Xilinx **Vitis 2024.2** + **XRT**.
  - **Host:** C++17 application using XRT.

- **HLS design techniques**
  - Use of **dataflow** and **pipelining** for high throughput.
  - **Memory access optimization** (burst transfers, stream interfaces).
  - Resource-aware design for LUTs, FFs, BRAMs and DSPs.

---

## 3. Repository Structure

```text
BFP-unit/
├── C++/                 # CPU-side reference models and tests (pure C++)
│   └── ...             
├── HW/                  # HLS kernel for the FPGA
│   ├── src/             # BFP implementation (encode, decode, add, sub, mul, div, rcp, etc.)
│   └── Makefile         # Build flow for the hardware kernel
├── SW/                  # Host application using XRT
│   ├── src/             # Host C++ code (buffer management, kernel invocation, checking)
│   └── Makefile         # Build flow for the host executable
├── Results/             # Synthesis reports and experimental results (timing, resources, error)
│   └── ...             
├── Documentation/       # Additional documentation (diagrams, reports, notes)
│   └── ...             
├── Makefile             # Top-level build script (HW + SW)
└── README.md            # Project description (this file)
```

---

## 4. Requirements

- **Software**
  - Xilinx Vitis 2024.2
  - XRT (Xilinx Runtime)
  - make
  - C++17 compiler (e.g. g++, clang++)

- **Hardware**
  - Xilinx Alveo U55C board with a compatible XDMA shell.
  - Host machine with PCIe and XRT drivers correctly installed.

---

## 5. Quick Start

From the root of the repository:

**5.1. Set up the environment**

Adjust these paths to your local installation:

```bash
# 1. Source Vitis
source /tools/Xilinx/Vitis/2024.2/settings64.sh

# 2. Source XRT
source /opt/xilinx/xrt/setup.sh
```

**5.2. Build and run**

```bash
# Build hardware + software
make

# Program the FPGA with the generated xclbin/bitstream
make flash

```

> **Note:** Hardware synthesis and bitstream generation can take several hours depending on your machine.

---

## 6. Build Targets

Common Makefile targets (names may vary slightly depending on your platform configuration):

| Command | Description |
|---------|-------------|
| `make` | Build hardware + software |
| `make hw` | HLS synthesis / implementation of the kernel |
| `make sw` | Build host application only |
| `make flash` | Program the Alveo U55C with the bitstream |
| `make clean` | Remove build artifacts |

Check the top-level Makefile and the `HW/` / `SW/` Makefiles if you need to adjust platform names or paths.

---

## 7. Usage and Flow

At a high level, the dataflow is:

```text
Input (FP32)
   └─→ ENCODE (FP32 → BFP)
          └─→ BFP OPS (ADD / SUB / MUL / DIV / RCP)
                 └─→ DECODE (BFP → FP32)
                        └─→ Output (FP32, compared vs FP32 reference)
```
---

## 8. Configuration

Default configuration (can be changed at compile time in the HLS code):

- **WE = 5 bits** (exponent width)
- **WM = 7 bits** (mantissa width)
- **N = 16 elements** per block

To experiment with other formats or block sizes:

1. Modify the corresponding template parameters / `#defines` in the HLS source (inside `HW/src/`)

2. Rebuild the hardware:
```bash
make hw
```

3. Rebuild the host (if needed) and rerun the tests:
```bash
make sw
make test
```
---

<!-- 
## 9. Documentation

Additional design documentation is stored under `Documentation/`, including:

- **BFP format description** and derivations
- **Block diagrams and flowcharts** for:
  - Encoding / decoding blocks
  - Arithmetic operations (`add_blocks`, `mul_blocks`, `rcp_block`, etc.)
- **Design decisions** and HLS optimization notes

If you add a main PDF/markdown guide (e.g., `Documentation/BFP-Unit-Guide.pdf`), reference it here for detailed explanations of the architecture and build flow.
-->

---

## Credits

- **Author:** Steven Arias Gutiérrez
- **Institution:** Instituto Tecnológico de Costa Rica (TEC)
- **Lab:** ECASLab – Efficient Computing Across the Stack

If you are interested in collaboration or have questions, feel free to open an issue or a pull request in this repository.

## License 
Block Floating-Point (BFP) Unit

Copyright (c) 2025 Steven Arias Gutiérrez

[Full text license]

