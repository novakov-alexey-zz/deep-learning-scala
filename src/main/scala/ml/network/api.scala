package ml.network

object api:  
  final type SimpleGD = ml.network.SimpleGD
  final type Adam = ml.network.Adam
  export ml.network.Dense
  export ml.network.optimizers.given
  export ml.network.Layer
  export ml.network.Sequential
  import ml.network.RandomGen.given
  export ml.network.GradientClippingApi._
  export ml.network.MetricApi._
  export ml.network.LossApi._
  export ml.network.ActivationFuncApi._