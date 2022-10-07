#include <type_traits>

//#include "../PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/MemRefAllocToGPU/MemRefAllocToGPU.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/SmallBitVector.h"
//#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
//#include "mlir/Dialect/SCF/SCF.h"
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

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFALLOCTOGPU
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
class MemRefAllocToGPUPattern
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
       rewriter.replaceOpWithNewOp<gpu::AllocOp>(op,allocOp.getMemref().getType(), Type(),voidValue,operandAdaptor.getDynamicSizes(),operandAdaptor.getSymbolOperands());
    }
    return success();
  }
};   
}

void mlir::populateMemRefAllocToGPUConversionPatterns(RewritePatternSet &patterns){
    patterns.add<MemRefAllocToGPUPattern>(patterns.getContext());
}

namespace {

class ConvertMemRefAllocToGPUPass
    : public impl::ConvertMemRefAllocToGPUBase<ConvertMemRefAllocToGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMemRefAllocToGPUConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<memref::AllocOp>([](memref::AllocOp op){
      memref::AllocOp allocOp = cast<memref::AllocOp>(op);
      Operation *baseOp(op);
      if (baseOp->getParentOfType<gpu::LaunchOp>()) return true;
      bool legal = false;
      func::FuncOp funcOp = baseOp->getParentOfType<func::FuncOp>();
        for (auto returnOp : funcOp.getBody().getOps<func::ReturnOp>()){
          func::ReturnOp returnOpBase = cast<func::ReturnOp>(returnOp);
          func::ReturnOpAdaptor returnOpAdaptor(returnOpBase);
          for (auto result : returnOpAdaptor.getOperands()){
              if (result == allocOp.getMemref()) {
                legal = true; break;
              }
          }
        }
        return legal;
    });
    target.addLegalOp<gpu::AllocOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertMemRefAllocToGPUPass() {
  return std::make_unique<ConvertMemRefAllocToGPUPass>();
}