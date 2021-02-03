import scala.reflect.ClassTag

trait RandomGen[T]:
  def gen: T

object RandomGen:

  def random2D[T: ClassTag](rows: Int, cols: Int)(using rng: RandomGen[T]): Tensor2D[T] =
    Tensor2D(Array.fill(rows)(Array.fill[T](cols)(rng.gen)))

  def zeros[T: Numeric: ClassTag](length: Int): Tensor1D[T] =
    val zero = summon[Numeric[T]].zero
    Tensor1D(Array.fill(length)(zero))

given uniform: RandomGen[Float] with
   override def gen: Float = 
     math.random().toFloat + 0.001f
