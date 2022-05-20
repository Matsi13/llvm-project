add_mlir_conversion_library(MemRefAllocToGPU
  MemrefAllocToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefAllocToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRMemRef
  MLIRTransforms
  MLIRArithmetic
  MLIRGPUOps
  MLIRPass
  MLIRSupport
  MLIRIR
  )
