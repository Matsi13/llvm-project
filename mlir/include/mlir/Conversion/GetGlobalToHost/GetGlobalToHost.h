#ifndef MLIR_CONVERSION_GETGLOBALTOHOST_GETGLOBALTOHOST_H
#define MLIR_CONVERSION_GETGLOBALTOHOST_GETGLOBALTOHOST_H

#include <memory>
#include "mlir/IR/PatternMatch.h"
// This pass will run on FuncOp, not memref.get_globalOp

namespace mlir {
class Pass;

class RewritePatternSet;
class MLIRContext;


void populateGetGlobalToHostConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertGetGlobalToHostPass();
} // namespace mlir

#endif 