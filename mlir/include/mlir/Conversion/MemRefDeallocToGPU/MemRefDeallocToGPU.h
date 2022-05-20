#ifndef MLIR_CONVERSION_MEMREFDEALLOCTOGPU_MEMREFALLOCTOGPU_H
#define MLIR_CONVERSION_MEMREFDEALLOCTOGPU_MEMREFALLOCTOGPU_H

#include <memory>
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;

class RewritePatternSet;
class MLIRContext;


void populateMemRefDeallocToGPUConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertMemRefDeallocToGPUPass();
} // namespace mlir

#endif 