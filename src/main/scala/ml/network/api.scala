package ml.network

object api:  
  final type StandardGD = ml.network.StandardGD
  final type Adam = ml.network.Adam
  final type Stub = ml.network.Stub

  import ml.network.RandomGen.given
  
  export ml.network.Dense
  export ml.network.optimizers.given
  export ml.network.Layer
  export ml.network.Sequential
  export ml.network.GradientClippingApi.*
  export ml.network.MetricApi.*
  export ml.network.Metric
  export ml.network.LossApi.*
  export ml.network.ActivationFuncApi.*