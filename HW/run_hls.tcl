# run_hls_2024.tcl - Versión mínima para Vitis 2024.2 (U55C, top=bfp_kernel)

#==============================================================================
# CONFIGURACION DEL PROYECTO
#==============================================================================
open_project -reset bfp_proj_fp32
set_top bfp_kernel

#==============================================================================
# ARCHIVOS FUENTE
#   (usa tus fuentes actuales del kernel)
#==============================================================================
add_files bfp_kernel.cpp
add_files bfp_ops_hls.h
add_files bfp_hls.h

#==============================================================================
# Testbench para C simulation (usa el TB que llama al kernel)
#==============================================================================
add_files -tb tb_kernel.cc

#==============================================================================
# CONFIGURACION DE LA SOLUCION
#   (sin flow_target vivado; objetivo: U55C)
#==============================================================================
open_solution -reset "sol1"
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 5.0 -name default  ;# 200 MHz

#==============================================================================
# CONFIGURACION (mínima, manteniendo tu estilo)
#==============================================================================
config_compile   -name_max_length 80
config_dataflow  -strict_mode warning
config_rtl       -deadlock_detection sim
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64
config_interface -m_axi_auto_max_ports=0
config_export    -format xo -ipname bfp_kernel

#==============================================================================
# EJECUCION DEL FLUJO (sin cosim, minimalista como el original)
#==============================================================================
puts "\n=========================================="
puts "Starting C Simulation (csim)"
puts "==========================================\n"
csim_design

puts "\n=========================================="
puts "Starting C Synthesis (csynth)"
puts "==========================================\n"
csynth_design

puts "\n=========================================="
puts "Exporting XO"
puts "==========================================\n"
export_design -format xo -rtl verilog -output bfp_kernel.xo

#==============================================================================
# REPORTE FINAL
#==============================================================================
puts "\n=========================================="
puts "HLS Flow Complete!"
puts "=========================================="
puts "Project: bfp_proj"
puts "Reports: bfp_proj/sol1/syn/report/"
puts "XO     : bfp_proj/sol1/bfp_kernel.xo (y copia local: ./bfp_kernel.xo)"
puts "\n"

exit
