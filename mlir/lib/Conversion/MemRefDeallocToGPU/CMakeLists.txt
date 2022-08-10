add_mlir_conversion_library(MemRefDeallocToGPU
  MemRefDeallocToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefDeallocToGPU

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
  )
