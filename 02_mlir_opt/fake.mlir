module {
  func.func @matmul(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>, %Cinit: tensor<2x4xf32>) -> tensor<2x4xf32> { 
    %C = "fake.fake"(%A, %B): (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %C : tensor<2x4xf32>
  }
}