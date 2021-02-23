package ml.network

object api extends GradientClippingApi with ActivationFuncApi with LossApi with MetricApi:  
  final type SimpleGD = ml.network.SimpleGD
  final type Adam = ml.network.Adam
  export ml.network.Dense
  export ml.network.optimizers.given
  export ml.network.Layer
  export ml.network.Sequential
  import ml.network.RandomGen.given