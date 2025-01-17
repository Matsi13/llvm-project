; RUN: opt -S -inferattrs -basic-aa -licm < %s | FileCheck %s

define void @test(ptr noalias %loc, ptr noalias %a) {
; CHECK-LABEL: @test
; CHECK: @strlen
; CHECK-LABEL: loop:
  br label %loop

loop:
  %res = call i64 @strlen(ptr %a)
  store i64 %res, ptr %loc
  br label %loop
}

; CHECK: declare i64 @strlen(ptr nocapture) #0
; CHECK: attributes #0 = { argmemonly mustprogress nofree nounwind readonly willreturn }
declare i64 @strlen(ptr)


