// scala 2.13.3

import $file.loader
import $file.tensor
import $file.encoders
import $file.converter

import Model.{batches, getAvgLoss}
import loader._
import tensor._
import encoders._
import converter._

import java.nio.file.Path
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

sealed trait Activation[T] extends (T => T) {
  def apply(x: T): T //TODO: change input & output param to Tensor[T]
}

trait Relu[T] extends Activation[T]
trait Sigmoid[T] extends Activation[T]

implicit val relu = new Relu[Float] {
  override def apply(x: Float): Float = math.max(0, x)
}

implicit val sigmoid = new Sigmoid[Float] {
  override def apply(x: Float): Float = 1 / (1 + math.exp(-x).toFloat)
}

sealed trait Derivative[T] {
  def derivative(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): T
}
sealed trait Loss[T] extends Derivative[T] {
  def apply(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): T
}

implicit val mse = new Loss[Float] {
  override def apply(
      actual: Tensor1D[Float],
      predicted: Tensor1D[Float]
  ): Float = {    
    var sumScore = 0.0
    for (i <- actual.data.indices) {
      val y1 = actual.data(i)
      val y2 = predicted.data(i)
      sumScore += math.pow(y1 - y2, 2)
    }
    val meanSumScore = 1.0 / actual.length * sumScore
    meanSumScore.toFloat
  }

  override def derivative(
      actual: Tensor1D[Float],
      predicted: Tensor1D[Float]
  ): Float = {    
    var sumScore = 0.0
    for (i <- actual.data.indices) {
      val y1 = actual.data(i)
      val y2 = predicted.data(i)
      sumScore += y1 - y2
    }
    val meanSumScore = 2 / actual.length * sumScore
    meanSumScore.toFloat
  }
}

// implicit val binaryCrossEntropy = new Loss[Float] {
//   /*
//   TODO: implement second case for multi-class prediction

//   def categorical_cross_entropy(actual, predicted):
//     sum_score = 0.0
//     for i in range(len(actual)):
//       for j in range(len(actual[i])):
//         sum_score += actual[i][j] * log(1e-15 + predicted[i][j])
//     mean_sum_score = 1.0 / len(actual) * sum_score
//     return -mean_sum_score
//    */
//   override def apply(
//       actual: Tensor1D[Float],
//       predicted: Tensor1D[Float]
//   ): Float = {
//     var sumScore = 0.0
//     for {
//       y1 <- actual.data
//       y2 <- predicted.data
//     } {
//       sumScore += y1 * math.log(1e-15 + y2.toDouble)
//     }
//     val meanSumScore = 1.0 / actual.length * sumScore
//     -meanSumScore.toFloat
//   }
// }

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
  def predict(x: Tensor[T]): Tensor[T]
  def loss: Tensor[T]
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

object Model {
  def batches[T](t: Tensor2D[T], batchSize: Int): Iterator[Array[Array[T]]] =
     t.data.grouped(batchSize)

  def batches[T](t: Tensor1D[T], batchSize: Int): Iterator[Array[T]] =
     t.data.grouped(batchSize)

  def getAvgLoss[T: Numeric](losses: List[T]) =
    implicitly[Numeric[T]].toFloat(losses.sum) / losses.length
}

case class Sequential[T: ClassTag: RandomGen: Numeric](
    lossFunc: Loss[T],
    optimizer: Optimizer[T],
    learningRate: T,
    batchSize: Int = 1,
    layers: List[Layer[T]] = Nil,
    weights: List[(Tensor[T], Activation[T])] = Nil
) extends Model[T] {
  self =>

  def currentWeights: List[Tensor[T]] = weights.map(_._1)

  def predict(x: Tensor[T]): Tensor[T] = activate(x, weights)

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

  private def activate(
      x: Tensor[T],
      weights: List[(Tensor[T], Activation[T])]
  ): Tensor[T] =
    weights.foldLeft(x) { case (a, (w, activation)) =>
      // println(s"w = $w")
      // println(s"a = $a")
      // println(s"res = ${w * a}")
      (w * a).map(activation) //TODO: add bias, i.e. w * a + b
    }

  def updateWeights(weights: List[(Tensor[T], Activation[T])], actual: Tensor1D[T], predicted: Tensor1D[T]) = 
    weights.foldLeft(List.empty[(Tensor[T], Activation[T])]) {
      case (acc, (w, activation)) =>
        val gradient = lossFunc.derivative(actual, predicted)
        val newWeight =
          w - learningRate * gradient //TODO: multiply by activation value on wi
        // println(s"newWeight:\n$newWeight")
        acc :+ (newWeight -> activation)
    }
  

  def train(x: Tensor2D[T], y: Tensor1D[T], epochs: Int): Model[T] = {
    val inputs = x.cols
    lazy val actualBatches = batches(y, batchSize).toArray
    lazy val xBatches = batches(x, batchSize).zip(actualBatches)
    
    val (updatedWeights, epochLosses) =
      (0 until epochs).foldLeft(getWeights(inputs), List.empty[T]) {
        case ((weights, losses), epoch) =>        
          val epochRes = xBatches.foldLeft(weights, losses) {
            case ((weights, losses), (batch, actualBatch)) =>
              val predicted = batch
                .map(row => activate(Tensor1D(row), weights))
                .combineAllAs1D              
              val actual = Tensor1D(actualBatch)
              val loss = lossFunc(actual, predicted)              
              val updated = updateWeights(weights, actual, predicted)                
              (updated, losses :+ loss)
          }
          println(s"epoch: $epoch, avg loss: ${getAvgLoss(epochRes._2)}")
          epochRes
      }
    println(s"losses count: ${epochLosses.length}")
    println(s"losses: ${epochLosses.mkString(",")}")

    copy(weights = updatedWeights)
  }

  private def getWeights(inputs: Int) =
    if (weights == Nil) initialWeights(inputs) else weights
}

val ann =
  Sequential(mse, stochasticGradientDescent, learningRate = 0.00001f, 32)
    .add(Dense(6)(relu))
    .add(Dense(6)(relu))
    .add(Dense()(sigmoid))

//1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1 = length 14
val data = TextLoader(Path.of("data", "Churn_Modelling.csv")).load()
val y = data.col[Float](-1)
// println(y.data.mkString(", "))

var x = data.cols[String]((3, -1))
// println(x.data.head.mkString(","))
val encoder = LabelEncoder[String]().fit(x.col(2))
x = encoder.transform(x, 2)
val hotEncoder = OneHotEncoder[String, Float]().fit(x.col(1))
// println(hotEncoder.classes)
x = hotEncoder.transform(x, 1)
val xFloat = transform[Float](x.data)
// println(xFloat.data.head.mkString(","))

//val x = Tensor2D(
//  Array(0.2f, 0.3f, 0.4f, 0.5f),
//  Array(0.1f, 0.1f, 0.4f, 0.2f)
//)
//val y = Tensor1D(0.8f, 0.3f)

val model = ann.train(xFloat, y, 100)
println("weights:\n " + model.currentWeights.mkString("\n"))
// print(x.data.map(_.mkString(", ")).mkString("\n"))
