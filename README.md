# Block Floating-Point (BFP) Unit in FPGA for Hardware Acceleration using HLS

Implementación de una unidad **Block Floating-Point (BFP)** para operaciones aritméticas en FPGA, diseñada y evaluada usando **Vitis HLS 2024.2**.  
El proyecto incluye el **kernel hardware en HLS**, el **host en C++ con XRT**, scripts de síntesis y **resultados experimentales**.

> Proyecto desarrollado como parte de un Trabajo Final de Graduación en Ingeniería Electrónica (Instituto Tecnológico de Costa Rica).

---

## 1. Características principales

- **Formato BFP parametrizable**
  - Exponente compartido por bloque.
  - Mantisas enteras con configuración `WE = 5` bits (exponente) y `WM = 7` bits (mantisa).
  - Tamaño de bloque por defecto: `N = 16` elementos.

- **Operaciones soportadas**
  - `ENCODE`  – FP32 → BFP.
  - `DECODE`  – BFP → FP32.
  - `ADD`     – suma de bloques BFP.
  - `SUB`     – resta de bloques BFP.
  - `MUL`     – multiplicación de bloques BFP.
  - `DIV`     – división de bloques BFP.
  - `RCP`     – recíproco de bloque BFP.

- **Plataforma objetivo**
  - **FPGA:** Alveo **U55C** (`xcu55c-fsvh2892-2L-e`).
  - **Herramienta:** Xilinx **Vitis 2024.2** + **XRT**.
  - **Host:** C++17 + XRT (Xilinx Runtime).

- **Enfoque de diseño**
  - Kernels en **C/C++ HLS** con optimizaciones para:
    - Paralelismo a nivel de datos.
    - Uso eficiente de LUTs, FFs, BRAMs y DSPs.
    - Reducción de latencia mediante `PIPELINE`, `DATAFLOW` y optimización de accesos a memoria.

---

## 2. Estructura del repositorio

La organización actual del proyecto es:

```text
BFP-unit/
├── C++/                 # Modelos de referencia en C++ / pruebas de precisión en CPU
│   └── ...             
├── HW/                  # Kernels HLS (unidad BFP para FPGA)
│   ├── src/             # Implementaciones BFP (encode, decode, add, mul, div, rcp, etc.)
│   └── Makefile         # Síntesis del kernel y exportación
├── SW/                  # Aplicación host (XRT)
│   ├── src/             # Código host en C++ (manejo de buffers, comandos, profiling)
│   └── Makefile         # Compilación del ejecutable host
├── Results/             # Resultados de síntesis y experimentos (timing, recursos, etc.)
│   └── ...             
├── Documentation/       # Documentos de apoyo (memoria, diagramas, notas técnicas)
│   └── ...             
└── README.md            # Este archivo


