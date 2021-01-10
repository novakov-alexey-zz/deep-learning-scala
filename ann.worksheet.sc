import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

sealed trait Activation[T] {
  def apply(x: T): T
}

trait Relu[T] extends Activation[T]
trait Sigmoid[T] extends Activation[T]

implicit val relu = new Relu[Double] {
  override def apply(x: Double): Double = math.max(0, x)
}

implicit val sigmoid = new Sigmoid[Double] {
  override def apply(x: Double): Double = 1 / (1 + math.exp(-x))
}

sealed trait Layer[T] {
  def units: Int
  def f: Activation[T]
}
case class Dense[T](units: Int = 1)(implicit val f: Activation[T])
    extends Layer[T]

//case class LayerState[T](data: Array[T])

sealed trait Model[T] {
  def layers: List[Layer[T]]
  def train(x: Tensor[T]): TrainedModel[T]
}

sealed trait Tensor[T] {
  type A
  def data: Array[A]
  def length: Int = data.length
}
case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T] {
  type A = T

  override def toString(): String = {
    val name = s"Tensor1D[${implicitly[ClassTag[T]]}]"
    s"$name:\n[" + data.mkString(",") + "]"
  }
}
case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T] {
  type A = Array[T]

  override def toString(): String = {
    val name = s"Tensor2D[${implicitly[ClassTag[T]]}]"
    s"$name:\n[" + data
      .map(a => a.mkString("[", ",", "]"))
      .mkString("\n ") + "]"
  }
}

object Tensor {
  def of[T: ClassTag](size: Int): Tensor1D[T] =
    new Tensor1D[T](Array.ofDim(size))

  def of[T: ClassTag](size: Int, size2: Int): Tensor2D[T] =
    new Tensor2D[T](Array.fill(size)(of(size2).data))

  def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match {
      case (Tensor1D(data), Tensor2D(data2)) =>
        new Tensor2D[T](matMul[T](Array(data), data2))
      case (Tensor2D(data), Tensor1D(data2)) =>
        new Tensor2D[T](matMul[T](data, Array(data2)))
      case (Tensor1D(data), Tensor1D(data2)) =>
        new Tensor2D[T](matMul[T](Array(data), Array(data2)))
      case (Tensor2D(data), Tensor2D(data2)) =>
        new Tensor2D[T](matMul[T](data, data2))
    }

  def activate[T: ClassTag](t: Tensor[T], f: Activation[T]): Tensor[T] =
    t match {
      case Tensor1D(data) => Tensor1D(data.map(f(_)))
      case Tensor2D(data) => Tensor2D(data.map(d => d.map(f(_))))
    }

  private def matMul[T: ClassTag: Numeric](
      a: Array[Array[T]],
      b: Array[Array[T]]
  ): Array[Array[T]] = {
    val cols = b.headOption.map(_.length).getOrElse(0)
    val rows = a.length
    val res = Array.ofDim(rows, cols)
    val numeric = implicitly[Numeric[T]]
    for (i <- (0 until rows).indices) {
      for (j <- (0 until cols).indices) {
        var sum = numeric.zero
        for (k <- b.indices) {
          sum = sum + (a(i)(k) * b(k)(j))
        }
        res(i)(j) = sum
      }
    }
    res
  }
}

sealed trait TrainedModel[T] {
  def predict[T](x: Array[Array[T]]): Array[T]

  def loss: Double
}

class SequentialTrainedModel[T](val data: Tensor[T]) extends TrainedModel[T] {

  override def predict[T](x: Array[Array[T]]): Array[T] = ???

  override def loss: Double = ???

  override def toString(): String = data.toString()
}

trait RandomGen[T] {
  def gen: T
}

implicit val randomUniform = new RandomGen[Double] {
  def gen: Double = math.random() + 0.001
}

def random2D[T: ClassTag](rows: Int, cols: Int)(implicit
    rng: RandomGen[T]
): Tensor2D[T] = {
  val rnd = implicitly[RandomGen[T]]
  Tensor2D(Array.fill(rows)(Array.fill[T](cols)(rnd.gen)))
}

case class Sequential[T: ClassTag: RandomGen: Numeric](
    layers: List[Layer[T]] = Nil
) extends Model[T] {
  self =>
  def add(l: Layer[T]) =
    self.copy(layers = layers :+ l)

  def train(x: Tensor[T]): TrainedModel[T] = {
    val initialWeights = layers.map(l => random2D(l.units, x.length))
    val res = initialWeights.foldLeft(x) { case (acc, w) =>
      println(s"w = $w")
      println(s"acc = $acc")
      Tensor.mul(w, acc)
    }

    new SequentialTrainedModel[T](res)
  }
}

val ann =
  Sequential()
    .add(Dense(6)(relu))
//.add(Dense(6)(relu))
//.add(Dense(1)(sigmoid))

val x = Tensor2D(Array(Array(0.0), Array(1.0), Array(0.0), Array(1.0)))
val model = ann.train(x)

Tensor.mul(
  Tensor2D(
    Array(Array(9, 8, 7, 6), Array(6, 5, 4, 3), Array(3, 2, 1, 0))
  ),
  Tensor2D(Array(Array(1), Array(2), Array(3), Array(4)))
)
