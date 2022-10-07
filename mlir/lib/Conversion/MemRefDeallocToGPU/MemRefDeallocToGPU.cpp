#include <type_traits>

//#include "../PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/MemRefDeallocToGPU/MemRefDeallocToGPU.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/SmallBitVector.h"
//#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
//#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"


#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFDEALLOCTOGPU
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
class MemRefDeallocToGPUPattern
    : public OpRewritePattern<memref::DeallocOp> {
  public:
  using OpRewritePattern<memref::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    memref::DeallocOpAdaptor operandAdaptor = memref::DeallocOpAdaptor(op);
    Operation* baseop(op);
    //auto deallocOp = cast<memref::DeallocOp>(op);
    // auto launchOp = op.getParentOfType<gpu::LaunchOp>();
    if (!baseop->getParentOfType<gpu::LaunchOp>())
    {
       //ValueRange voidValue = {};
       rewriter.replaceOpWithNewOp<gpu::DeallocOp>(op, Type(), ValueRange(),operandAdaptor.getMemref());
    }
    return success();
  }
};   
}

void mlir::populateMemRefDeallocToGPUConversionPatterns(RewritePatternSet &patterns){
    patterns.add<MemRefDeallocToGPUPattern>(patterns.getContext());
}

namespace {

class ConvertMemRefDeallocToGPUPass
    : public impl::ConvertMemRefDeallocToGPUBase<ConvertMemRefDeallocToGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMemRefDeallocToGPUConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<memref::DeallocOp>([](memref::DeallocOp op){
      Operation *baseop(op);
        if (baseop->getParentOfType<gpu::LaunchOp>())
        {
            return true;
        }
        return false;
    });
    target.addLegalOp<gpu::DeallocOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertMemRefDeallocToGPUPass() {
  return std::make_unique<ConvertMemRefDeallocToGPUPass>();
}