#include <type_traits>

#include "../PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/MemRefAllocaToAlloc/MemRefAllocaToAlloc.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"


#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/BlockAndValueMapping.h"


#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
class MemRefAllocaToAllocPattern
    : public OpRewritePattern<memref::AllocaOp> {
  public:
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    auto allocaOp = cast<memref::AllocaOp>(op);
    memref::AllocaOpAdaptor allocaOpAdaptor(allocaOp);
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op,allocaOp.getType(), allocaOpAdaptor.dynamicSizes(),allocaOp.alignmentAttr());
    return success();
  }
};   
}

void mlir::populateMemRefAllocaToAllocConversionPatterns(RewritePatternSet &patterns){
    patterns.add<MemRefAllocaToAllocPattern>(patterns.getContext());
}

namespace {

class ConvertMemRefAllocaToAllocPass
    : public ConvertMemRefAllocaToAllocBase<ConvertMemRefAllocaToAllocPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMemRefAllocaToAllocConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addIllegalOp<memref::AllocaOp>();
    target.addLegalOp<memref::AllocOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertMemRefAllocaToAllocPass() {
  return std::make_unique<ConvertMemRefAllocaToAllocPass>();
}