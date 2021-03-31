package ml.network

object api:  
  final type StandardGD = ml.network.StandardGD
  final type Adam = ml.network.Adam
  final type Stub = ml.network.Stub
  
  final type MetricValues[T] = ml.network.MetricValues[T]
  
  final type RandomUniform = ml.network.RandomUniform
  final type HeNormal = ml.network.HeNormal
  
  export ml.network.Dense
  export ml.network.Conv2D
  export ml.network.MaxPool
  export ml.network.Flatten2D
  export ml.network.Layer
  export ml.network.optimizers.given
  export ml.network.Optimizable
  export ml.network.Sequential
  export ml.network.Model
  export ml.network.GradientClippingApi.*
  export ml.network.GradientClippingApi
  export ml.network.GradientClipping
  export ml.network.MetricApi.*
  export ml.network.Metric
  export ml.network.LossApi.*
  export ml.network.ActivationFuncApi.*
  export ml.network.ParamsInitializer
  export ml.network.inits.given
  export ml.network.inits