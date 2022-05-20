#ifndef MLIR_CONVERSION_MEMREFDEALLOCTOGPU_MEMREFALLOCTOGPU_H
#define MLIR_CONVERSION_MEMREFDEALLOCTOGPU_MEMREFALLOCTOGPU_H

#include <memory>
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;
// class LLVMTypeConverter;
class RewritePatternSet;
class MLIRContext;

/// Collect a set of patterns to convert memory-related operations from the
/// MemRef dialect to the LLVM dialect.
void populateMemRefDeallocToGPUConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertMemrefDeallocToGPUPass();
} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLLVM_H