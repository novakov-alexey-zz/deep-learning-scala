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

sealed trait Loss[T] {
  def apply(actual: Array[T], predicted: Array[T]): T //TODO: change input args to Tensor[T]
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
  override def apply(actual: Array[Float], predicted: Array[Float]): Float = {
    var sumScore = 0.0
    for {
      y1 <- actual
      y2 <- predicted
    } {
      sumScore += y1 * math.log(1e-15 + y2.toDouble)
    }
    val meanSumScore = 1.0 / actual.length * sumScore
    -meanSumScore.toFloat
  }
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
  def weights: List[Tensor[T]]
  def predict[T](x: Tensor[T]): Tensor[T]
  def loss: T
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
}

object Tensor {
  def of[T: ClassTag](size: Int): Tensor1D[T] =
    new Tensor1D[T](Array.ofDim(size))

  def of[T: ClassTag](size: Int, size2: Int): Tensor2D[T] =
    new Tensor2D[T](Array.fill(size)(of(size2).data))

  def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match {
      case (Tensor1D(data), Tensor2D(data2)) =>
        new Tensor2D[T](matMul[T](asColumn(data), data2))
      case (Tensor2D(data), Tensor1D(data2)) =>
        new Tensor2D[T](matMul[T](data, asColumn(data2)))
      case (Tensor1D(data), Tensor1D(data2)) =>
        new Tensor1D[T](matMul[T](asColumn(data), Array(data2)).head)
      case (Tensor2D(data), Tensor2D(data2)) =>
        new Tensor2D[T](matMul[T](data, data2))
    }

  private def asColumn[T: ClassTag](a: Array[T]) = a.map(Array(_))

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
    layers: List[Layer[T]] = Nil,
    var _weights: List[(Tensor[T], Activation[T])] = Nil
) extends Model[T] {
  self =>

  def weights: List[Tensor[T]] = _weights.map(_._1)

  def predict[T](x: Tensor[T]): Tensor[T] = ???

  def loss: T = ???

  def add(l: Layer[T]) =
    self.copy(layers = layers :+ l)

  def initialWeights(inputs: Int): List[(Tensor[T], Activation[T])] = {
    val (initialWeights, _) =
      layers.foldLeft(List.empty[(Tensor[T], Activation[T])], inputs) {
        case ((acc, inputs), l) =>
          (acc :+ (random2D(l.units, inputs), l.f), l.units)
      }

    initialWeights
  }

  private def _predict(
      x: Tensor[T],
      weights: List[(Tensor[T], Activation[T])]
  ): Tensor[T] = {
    weights.foldLeft(x) { case (a, (w, activation)) =>
      println(s"w = $w")
      println(s"a = $a")
      println(s"res = ${w * a}")
      Tensor.activate(w * a, activation) //TODO: add bias
    }
  }

  def train(x: Tensor2D[T], y: Tensor1D[T], epocs: Int): Model[T] = {
    val inputs = x.cols
    val weights = if (_weights == Nil) initialWeights(inputs) else _weights
    val predictedBatch = x.data.foldLeft(List.empty[Tensor[T]]) { case (acc, row) =>
      println("PREDICT >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
      acc :+ _predict(Tensor1D(row), weights)
    }
    println("x = " + x)
    println("predictedBatch =  " + predictedBatch)
    println("actual =  " + y)
    val predicted = predictedBatch.map { t => 
      t match {
        case Tensor1D(data) => data.head
        case Tensor2D(data) => data.head.head
      }
    }
    println("predicted =  " + predicted)
    val loss = lossFunc(y.data, predicted.toArray)
    println("loss = " + loss)
    Sequential[T](lossFunc, layers, weights)
  }
}

val ann =
  Sequential(binaryCrossEntropy)
    .add(Dense(6)(relu))
    .add(Dense(6)(relu))
    .add(Dense(1)(sigmoid))

val x = Tensor2D(
  Array(0.2f, 1.0f, 0.4f, 1.0f),
  Array(0.12f, 0.1f, 0.44f, 0.202f)
)
val y = Tensor1D(0.8f, 0.3f)

val model = ann.train(x, y, 100)
//println("weights:\n " + model.weights.mkString("\n"))
