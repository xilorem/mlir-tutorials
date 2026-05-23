module {
  func.func @z_add_example(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = arith.addi %x, %c0 : i32
    return %0 : i32
  }
}
