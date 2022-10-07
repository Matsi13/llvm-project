#ifndef MLIR_CONVERSION_MEMREFALLOCTOGPU_MEMREFALLOCTOGPU_H
#define MLIR_CONVERSION_MEMREFALLOCTOGPU_MEMREFALLOCTOGPU_H

#include <memory>
#include "mlir/Support/LLVM.h"
//#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Location;
struct LogicalResult;
class OpBuilder;
class Pass;
class RewritePattern;
class Value;
class ValueRange;

class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTAFFINETOSTANDARD
#include "mlir/Conversion/Passes.h.inc"


void populateMemRefAllocToGPUConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertMemRefAllocToGPUPass();
} // namespace mlir

#endif 