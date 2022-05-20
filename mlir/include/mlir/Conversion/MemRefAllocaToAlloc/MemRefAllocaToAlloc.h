#ifndef MLIR_CONVERSION_MEMREFALLOCATOALLOC_MEMREFALLOCATOALLOC_H
#define MLIR_CONVERSION_MEMREFALLOCATOALLOC_MEMREFALLOCATOALLOC_H

#include <memory>
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;

class RewritePatternSet;
class MLIRContext;


void populateMemRefAllocaToAllocConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertMemRefAllocaToAllocPass();
} // namespace mlir

#endif 