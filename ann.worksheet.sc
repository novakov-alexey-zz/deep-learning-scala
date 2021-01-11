import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

sealed trait Activation[T] {
  def apply(x: T): T
}

trait Relu[T] extends Activation[T]
trait Sigmoid[T] extends Activation[T]

implicit val relu = new Relu[Float] {
  override def apply(x: Float): Float = math.max(0, x)
}

implicit val sigmoid = new Sigmoid[Float] {
  override def apply(x: Float): Float = 1 / (1 + math.exp(-x).toFloat)
}

sealed trait Layer[T] {
  def units: Int
  def f: Activation[T]
}
case class Dense[T](units: Int = 1)(implicit val f: Activation[T])
    extends Layer[T]

sealed trait Model[T] {
  def layers: List[Layer[T]]
  def train(x: Tensor[T]): TrainedModel[T]
}

sealed trait Tensor[T] {
  type A
  def data: Array[A]
  def length: Int = data.length
  def sizes: List[Int]
}
case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T] {
  type A = T

  override def sizes: List[Int] = List(data.length)

  override def toString(): String = {
    val meta = s"sizes: ${sizes.head}, Tensor1D[${implicitly[ClassTag[T]]}]"
    s"$meta:\n[" + data.mkString(",") + "]\n"
  }
}
case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T] {
  type A = Array[T]

  override def sizes: List[Int] =
    List(data.length, data.headOption.map(_.length).getOrElse(0))

  override def toString(): String = {
    val meta =
      s"sizes: ${sizes.mkString("x")}, Tensor2D[${implicitly[ClassTag[T]]}]"
    s"$meta:\n[" + data
      .map(a => a.mkString("[", ",", "]"))
      .mkString("\n ") + "]\n"
  }
}

implicit class TensorOps[T: ClassTag: Numeric](val t: Tensor[T]) {
  def *(that: Tensor[T]): Tensor[T] = Tensor.mul(t, that)
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
        new Tensor2D[T](matMul[T](Array(data), Array(data2))) //TODO: return 1D?
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
    val rows = a.length
    val cols = b.headOption.map(_.length).getOrElse(0)
    val res = Array.ofDim(rows, cols)

    for (i <- (0 until rows).indices) {
      for (j <- (0 until cols).indices) {
        var sum = implicitly[Numeric[T]].zero
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
  def predict[T](x: Tensor[T]): Tensor[T]

  def loss: T
}

class SequentialTrainedModel[T](val data: Tensor[T]) extends TrainedModel[T] {

  override def predict[T](x: Tensor[T]): Tensor[T] = ???

  override def loss: T = ???

  override def toString(): String = data.toString()
}

trait RandomGen[T] {
  def gen: T
}

implicit val randomUniform = new RandomGen[Float] {
  def gen: Float = math.random().toFloat + 0.001f
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
    val inputs = x.length
    val (initialWeights, _) =
      layers.foldLeft(List.empty[(Tensor[T], Activation[T])], inputs) {
        case ((acc, inputs), l) =>
          (acc :+ (random2D(l.units, inputs), l.f), l.units)
      }
    val res = initialWeights.foldLeft(x) { case (a, (w, activation)) =>
      // println(s"w = $w")
      // println(s"a = $a")
      // println(s"res = ${w * a}")
      Tensor.activate(w * a, activation) //TODO: add bias
    }
    new SequentialTrainedModel[T](res)
  }
}

val ann =
  Sequential()
    .add(Dense(6)(relu))
    .add(Dense(6)(relu))
    .add(Dense(1)(sigmoid))

val x = Tensor2D(Array(Array(0.0f), Array(1.0f), Array(0.0f), Array(1.0f)))
val model = ann.train(x)
println(model)
