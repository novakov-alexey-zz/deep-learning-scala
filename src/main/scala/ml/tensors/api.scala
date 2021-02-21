package ml.tensors

object api extends ops:
  export ml.tensors.{Tensor, Tensor0D, Tensor1D, Tensor2D}
  export scala.math.Numeric.Implicits._
  export math.Ordering.Implicits.infixOrderingOps