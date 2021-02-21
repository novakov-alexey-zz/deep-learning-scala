package ml.network

import scala.reflect.ClassTag
import ml.transformation.transformAny
import ml.tensors.api._

trait RandomGen[T]:
  def gen: T

object RandomGen:

  def random2D[T: ClassTag](rows: Int, cols: Int)(using rng: RandomGen[T]): Tensor[T] =
    Tensor2D(Array.fill(rows)(Array.fill[T](cols)(rng.gen)))

  def zeros[T: Numeric: ClassTag](length: Int)(using n: Numeric[T]): Tensor[T] =    
    Tensor1D(Array.fill(length)(n.zero))

  given uniform[T: Numeric: ClassTag]: RandomGen[T] with
    override def gen: T = 
      transformAny[Double, T](math.random().toDouble + 0.001d)
