add_mlir_conversion_library(MemRefAllocaToAlloc
  MemRefAllocaToAlloc.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefAllocaToAlloc

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRMemRef
  MLIRTransforms
  MLIRPass
  MLIRSupport
  MLIRIR
  )
