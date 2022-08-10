add_mlir_conversion_library(MemRefAllocToGPU
  MemRefAllocToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefAllocToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRMemRefDialect
  MLIRTransforms
  #MLIRArithmetic
  MLIRGPUOps
  MLIRGPUTransforms
  MLIRPass
  MLIRSupport
  MLIRIR
  MLIRFuncDialect
  )
