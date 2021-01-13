import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

sealed trait Activation[T] {
  def apply(x: T): T //TODO: change input & output param to Tensor[T]

  // a.k.a gradient
  def derivative(x: T): T //TODO: change input & output param to Tensor[T]
}

trait Relu[T] extends Activation[T]
trait Sigmoid[T] extends Activation[T]

implicit val relu = new Relu[Float] {

  override def derivative(x: Float): Float = if (x >= 0) 1 else 0

  override def apply(x: Float): Float = math.max(0, x)
}

implicit val sigmoid = new Sigmoid[Float] {

  override def derivative(x: Float): Float = {
    val s = apply(x)
    s * (1 - s)
  }

  override def apply(x: Float): Float = 1 / (1 + math.exp(-x).toFloat)
}

sealed trait Loss[T] {
  def apply(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): T
}

implicit val binaryCrossEntropy = new Loss[Float] {
  /*
  TODO: implement second case for multi-class prediction

  def categorical_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
      for j in range(len(actual[i])):
        sum_score += actual[i][j] * log(1e-15 + predicted[i][j])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score
   */
  override def apply(
      actual: Tensor1D[Float],
      predicted: Tensor1D[Float]
  ): Float = {
    var sumScore = 0.0
    for {
      y1 <- actual.data
      y2 <- predicted.data
    } {
      sumScore += y1 * math.log(1e-15 + y2.toDouble)
    }
    val meanSumScore = 1.0 / actual.length * sumScore
    -meanSumScore.toFloat
  }
}

sealed trait Optimizer[T] {
  def apply(t: Tensor2D[T]): Tensor2D[T]
}

implicit val adam = new Optimizer[Float] {
  override def apply(t: Tensor2D[Float]): Tensor2D[Float] = ???
}

implicit val stochasticGradientDescent = new Optimizer[Float] {
  //TODO: move initial algorithm here
  override def apply(t: Tensor2D[Float]): Tensor2D[Float] = ???
}

sealed trait Layer[T] {
  def units: Int
  def f: Activation[T]
}
case class Dense[T](units: Int = 1)(implicit val f: Activation[T])
    extends Layer[T]

sealed trait Model[T] {
  def layers: List[Layer[T]]
  def train(x: Tensor2D[T], y: Tensor1D[T], epocs: Int): Model[T]
  def currentWeights: List[Tensor[T]]
  def predict[T](x: Tensor[T]): Tensor[T]
  def loss: Tensor[T]
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

object Tensor1D {
  def apply[T: ClassTag](data: T*): Tensor1D[T] = Tensor1D[T](data.toArray)
}

case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T] {
  type A = Array[T]

  override def sizes: List[Int] =
    List(_sizes._1, _sizes._2)

  private def _sizes: (Int, Int) =
    (data.length, data.headOption.map(_.length).getOrElse(0))

  override def toString(): String = {
    val meta =
      s"sizes: ${sizes.mkString("x")}, Tensor2D[${implicitly[ClassTag[T]]}]"
    s"$meta:\n[" + data
      .map(a => a.mkString("[", ",", "]"))
      .mkString("\n ") + "]\n"
  }

  def cols: Int = _sizes._2
}

object Tensor2D {
  def apply[T: ClassTag](rows: Array[T]*): Tensor2D[T] =
    Tensor2D[T](rows.toArray)
}

implicit class TensorOps[T: ClassTag: Numeric](val t: Tensor[T]) {
  def *(that: Tensor[T]): Tensor[T] = Tensor.mul(t, that)
  def activate(f: Activation[T]) = Tensor.activate(t, f)
  def gradient(f: Activation[T]) = Tensor.gradient(t, f)
}

implicit class TensorOps2[T: ClassTag: Numeric](val t: Array[Tensor[T]]) {
  def combineAllAs1D = Tensor.combineAllAs1D(t)
}

object Tensor {
  def of[T: ClassTag](size: Int): Tensor1D[T] =
    Tensor1D[T](Array.ofDim(size))

  def of[T: ClassTag](size: Int, size2: Int): Tensor2D[T] =
    Tensor2D[T](Array.fill(size)(of(size2).data))

  def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match {
      case (Tensor1D(data), Tensor2D(data2)) =>
        Tensor2D[T](matMul[T](asColumn(data), data2))
      case (Tensor2D(data), Tensor1D(data2)) =>
        Tensor2D[T](matMul[T](data, asColumn(data2)))
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor1D[T](matMul[T](asColumn(data), Array(data2)).head)
      case (Tensor2D(data), Tensor2D(data2)) =>
        Tensor2D[T](matMul[T](data, data2))
    }

  private def asColumn[T: ClassTag](a: Array[T]) = a.map(Array(_))

  def activate[T: ClassTag](t: Tensor[T], f: Activation[T]): Tensor[T] =
    t match {
      case Tensor1D(data) => Tensor1D(data.map(f(_)))
      case Tensor2D(data) => Tensor2D(data.map(d => d.map(f(_))))
    }

  def gradient[T: ClassTag](t: Tensor[T], f: Activation[T]): Tensor[T] =
    t match {
      case Tensor1D(data) => Tensor1D(data.map(f.derivative(_)))
      case Tensor2D(data) => Tensor2D(data.map(d => d.map(f.derivative(_))))
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

  def combineAll[T: ClassTag](ts: List[Tensor1D[T]]): Tensor1D[T] =
    ts.reduce[Tensor1D[T]] { case (a, b) => Tensor.combine(a, b) }

  def combine[T: ClassTag](a: Tensor1D[T], b: Tensor1D[T]): Tensor1D[T] =
    Tensor1D(a.data ++ b.data)

  def combineAllAs1D[T: ClassTag](ts: Iterable[Tensor[T]]): Tensor1D[T] =
    ts.foldLeft(Tensor1D()) { case (a, b) => combineAs1D(a, b) }

  def combineAs1D[T: ClassTag](a: Tensor[T], b: Tensor[T]): Tensor1D[T] =
    (a, b) match {
      case (t1 @ Tensor1D(data), t2 @ Tensor1D(data2)) => combine(t1, t2)
      case (t1 @ Tensor1D(data), t2 @ Tensor2D(data2)) =>
        combine(t1, Tensor1D(data2.flatten))
      case (t1 @ Tensor2D(data), t2 @ Tensor1D(data2)) =>
        combine(Tensor1D(data.flatten), t2)
      case (t1 @ Tensor2D(data), t2 @ Tensor2D(data2)) =>
        combine(Tensor1D(data.flatten), Tensor1D(data2.flatten))
    }
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
    lossFunc: Loss[T],
    optimizer: Optimizer[T],
    learningRate: Float = 0.00001f,
    layers: List[Layer[T]] = Nil,
    weights: List[(Tensor[T], Activation[T])] = Nil
) extends Model[T] {
  self =>

  def currentWeights: List[Tensor[T]] = weights.map(_._1)

  def predict[T](x: Tensor[T]): Tensor[T] = ???

  def loss: Tensor[T] = ???

  def add(l: Layer[T]) =
    self.copy(layers = layers :+ l)

  def initialWeights(inputs: Int): List[(Tensor[T], Activation[T])] = {
    val (weights, _) =
      layers.foldLeft(List.empty[(Tensor[T], Activation[T])], inputs) {
        case ((acc, inputs), layer) =>
          (acc :+ (random2D(layer.units, inputs), layer.f), layer.units)
      }
    weights
  }

  private def _predict(
      x: Tensor[T],
      weights: List[(Tensor[T], Activation[T])]
  ): Tensor[T] =
    weights.foldLeft(x) { case (a, (w, activation)) =>
      println(s"w = $w")
      println(s"a = $a")
      println(s"res = ${w * a}")
      (w * a).activate(activation) //TODO: add bias
    }

  def train(x: Tensor2D[T], y: Tensor1D[T], epocs: Int): Model[T] = {
    val inputs = x.cols
    val w = if (weights == Nil) initialWeights(inputs) else weights

    val (updatedWeights, epochLosses) =
      (0 until epocs).foldLeft(w, List.empty[T]) { case ((w, losses), epoch) =>
        val predicted =
          x.data.map(row => _predict(Tensor1D(row), w)).combineAllAs1D
        // println("x = " + x)
        // println("actual =  " + y)
        println(s"\npredicted: $predicted")
        val loss = lossFunc(y, predicted)
        println(s"loss: $loss")
        // TODO: update weights
        val newWeights = w.foldLeft(List.empty[(Tensor[T], Activation[T])]) {
          case (acc, (t, activation)) =>
            val gradient = t.gradient(activation)
            acc :+ (gradient -> activation)
        }
        (newWeights, losses :+ loss)
      }
    copy(weights = updatedWeights)
  }
}

val ann =
  Sequential(binaryCrossEntropy, stochasticGradientDescent)
    .add(Dense(6)(relu))
    .add(Dense(6)(relu))
    .add(Dense(1)(sigmoid))

val x = Tensor2D(
  Array(0.2f, 1.0f, 0.4f, 1.0f),
  Array(0.12f, 0.1f, 0.44f, 0.202f)
)
val y = Tensor1D(0.8f, 0.3f)

val model = ann.train(x, y, 1)
//println("weights:\n " + model.currentWeights.mkString("\n"))
