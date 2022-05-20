#include <type_traits>

#include "../PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/MemrefAllocToGPU/MemrefAllocToGPU.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"


#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"



using namespace mlir;

namespace {
class MemrefAllocToGPUPattern
    : public OpRewritePattern<memref::AllocOp> {
  public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    memref::AllocOpAdaptor operandAdaptor = memref::AllocOpAdaptor(op);
    Operation* baseop(op);
    auto allocOp = cast<memref::AllocOp>(op);
    // auto launchOp = op.getParentOfType<gpu::LaunchOp>();
    if (!baseop->getParentOfType<gpu::LaunchOp>())
    {
       ValueRange voidValue = {};
       rewriter.replaceOpWithNewOp<gpu::AllocOp>(op,allocOp.memref().getType(), Type(),voidValue,operandAdaptor.dynamicSizes(),operandAdaptor.symbolOperands());
    }
    return success();
  }
};   
}

void mlir::populateMemRefAllocToGPUConversionPatterns(RewritePatternSet &patterns){
    patterns.add<MemrefAllocToGPUPattern>(patterns.getContext());
}

namespace {

class ConvertMemrefAllocToGPUPass
    : public ConvertMemrefAllocToGPUBase<ConvertMemrefAllocToGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMemRefAllocToGPUConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<memref::AllocOp>([](memref:: AllocOp op){
      Operation *baseop(op);
        if (baseop->getParentOfType<gpu::LaunchOp>())
        {
            return true;
        }
        return false;
    });
    target.addLegalOp<gpu::AllocOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertMemrefAllocToGPUPass() {
  return std::make_unique<ConvertMemrefAllocToGPUPass>();
}