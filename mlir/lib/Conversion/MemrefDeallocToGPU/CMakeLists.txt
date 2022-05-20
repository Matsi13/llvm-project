add_mlir_conversion_library(MemRefDeallocToGPU
  MemrefDeallocToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefDeallocToGPU

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
