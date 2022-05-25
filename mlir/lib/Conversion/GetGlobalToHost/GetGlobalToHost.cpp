#include "../PassDetail.h"
#include "mlir/Conversion/GetGlobalToHost/GetGlobalToHost.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <functional>

using namespace mlir;

namespace {
class GetGlobalToHostPattern
    : public OpRewritePattern<func::FuncOp> {
  public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = cast<func::FuncOp>(op);
    Operation* baseOp(op);
    OpBuilder builder(funcOp);
    // Builder valueBuilder;
    // ArrayRef<int64_t> destshape;
    // MemRefType dest;
    // Float32Type flt;
    // MemRefType destType =  dest.get({}, Float32Type());
   
    // only deal with the main function
    if (!baseOp->getParentOfType<func::FuncOp>()){
        for (auto getGlobalOp : funcOp.getRegion().getOps<memref::GetGlobalOp>()){           
            builder.setInsertionPointAfter(getGlobalOp);
            auto castOp = builder.create<memref::CastOp>(getGlobalOp.getLoc(), UnrankedMemRefType::get(Float32Type(), 1), getGlobalOp.result());
            // auto castOp = builder.create<memref::CastOp>(getGlobalOp.getLoc(), getGlobalOp.result().getType(), getGlobalOp.result());
            builder.setInsertionPointAfter(castOp);
            builder.create<gpu::HostRegisterOp>(castOp.getLoc(), castOp.dest());
        }

    }
    // void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
    //                FunctionType type, ArrayRef<NamedAttribute> attrs,
    //                ArrayRef<DictionaryAttr> argAttrs)
    // func::FuncOp newFuncOp = rewriter.replaceOpWithNewOp<func::FuncOp>(op, op->getLoc(), funcOp.getName(), funcOp.getFunctionType(), funcOp.getResultAttrs(), {});
    // for(int i = 0; i < funcOp.getNumArguments(); i++){
    //   newfuncOp.setArgAttrs(funcOp.getArgAttrDict(i));
    // }
    // rewriter.replaceOp(op,funcOp);
    return success();
  }
};   
}

void mlir::populateGetGlobalToHostConversionPatterns(RewritePatternSet &patterns){
    patterns.add<GetGlobalToHostPattern>(patterns.getContext());
}

namespace {

class ConvertGetGlobalToHostPass
    : public ConvertGetGlobalToHostBase<ConvertGetGlobalToHostPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGetGlobalToHostConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op){
      auto funcOp = cast<func::FuncOp>(op);
      Operation *baseop(op);
      int getGlobalCount = 0, registerCount = 0;
      // we only deal with the main function
        if (baseop->getParentOfType<func::FuncOp>()){
            return true;
        }
        else {
            for (auto getGlobalOp : funcOp.getRegion().getOps<memref::GetGlobalOp>()){
              getGlobalCount++;
            }
            for (auto registerOp : funcOp.getRegion().getOps<gpu::HostRegisterOp>()){
              registerCount++;
            }
        }
        return getGlobalCount == registerCount;
        
        
    });
    target.addLegalOp<gpu::HostRegisterOp>();
    target.addLegalOp<memref::CastOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertGetGlobalToHostPass() {
  return std::make_unique<ConvertGetGlobalToHostPass>();
}






