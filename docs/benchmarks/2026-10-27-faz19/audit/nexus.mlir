module @nexus_spmd {
  func.func @main(%arg0: tensor<8x16xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<16x32xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<32xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<32x8xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<8x8xf32> {mhlo.sharding = "{replicated}"}) {
    %v4 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
    %v5_b = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<8x32xf32>
    %v5 = stablehlo.add %v4, %v5_b : tensor<8x32xf32>
    %v6 = stablehlo.tanh %v5 : tensor<8x32xf32>
    %v7 = "stablehlo.dot_general"(%v6, %arg3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
    return %v7 : tensor<8x8xf32>
  }
}
