#ifndef MLIR_CONVERSION_MEMREFALLOCTOGPU_MEMREFALLOCTOGPU_H
#define MLIR_CONVERSION_MEMREFALLOCTOGPU_MEMREFALLOCTOGPU_H

#include <memory>
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;

class RewritePatternSet;
class MLIRContext;


void populateMemRefAllocToGPUConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertMemRefAllocToGPUPass();
} // namespace mlir

#endif 