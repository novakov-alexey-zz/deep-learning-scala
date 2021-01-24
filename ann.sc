// scala 2.13.3

import $file.encoders
import $file.loader
import $file.tensor
import $file.converter

import Model.{batches, getAvgLoss, batchColumn}
import loader._
import tensor._
import encoders._
import converter._
import tensor.randoms.uniform

import java.nio.file.Path
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag
import scala.collection.mutable.ArrayBuffer

sealed trait Activation[T] extends (Tensor[T] => Tensor[T]) {
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]
}

implicit val relu = new Activation[Float] {

  override def derivative(x: tensor.Tensor[Float]): tensor.Tensor[Float] =
    Tensor.map(x, t => if (t < 0) 0 else 1)

  override def apply(x: Tensor[Float]): Tensor[Float] =
    Tensor.map(x, t => math.max(0, t))
}

implicit val sigmoid = new Activation[Float] {

  override def apply(x: Tensor[Float]): Tensor[Float] =
    Tensor.map(x, t => 1 / (1 + math.exp(-t).toFloat))

  override def derivative(x: Tensor[Float]): Tensor[Float] =
    Tensor.map(
      x,
      t => math.exp(-t).toFloat / math.pow(1 + math.exp(-t).toFloat, 2).toFloat
    )
}

sealed trait Gradient[T] {
  type Delta = (Tensor[T], Tensor[T])

  def gradient(
      x: Tensor[T],
      error: Tensor[T]
  ): Delta
}
sealed trait Loss[T] extends Gradient[T] {
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

  override def gradient(
      x: Tensor[Float],
      error: Tensor[Float]
  ): Delta = {
    // println(s"samples = ${error.length}")
    println(s"x = ${x}")
    println(s"error = $error")
    val n = 1f / error.length
    val weightGradient = n * (x * error)
    val biasGradient = n * error
    weightGradient -> biasGradient
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
  def apply(
      t: Tensor2D[T]
  ): Tensor2D[T] //TODO: change this to def train(x: Tensor[T], y: Tensor1D[T], epochs: Int): (List[Weight], List[T])
}

implicit val adam = new Optimizer[Float] {
  override def apply(t: Tensor2D[Float]): Tensor2D[Float] = ???
}

implicit val miniBatchGradientDescent = new Optimizer[Float] {
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
  def train(x: Tensor[T], y: Tensor1D[T], epocs: Int): Model[T]
  def currentWeights: List[(Tensor[T], Tensor[T])]
  def predict(x: Tensor[T]): Tensor[T]
  def losses: List[T]
}

object Model {
  def batches[T: ClassTag](
      t: Tensor2D[T],
      batchSize: Int
  ): Iterator[Array[Array[T]]] =
    t.data.grouped(batchSize)

  def batchColumn[T: ClassTag](
      t: Tensor1D[T],
      batchSize: Int
  ): Iterator[Array[T]] =
    t.data.grouped(batchSize)

  def batches[T: ClassTag: Numeric](
      t: Tensor[T],
      batchSize: Int
  ): Iterator[Array[Array[T]]] =
    t match {
      case Tensor1D(data) => t.as2D.data.grouped(batchSize)
      case Tensor2D(data) => data.grouped(batchSize)
    }

  def getAvgLoss[T: Numeric](losses: List[T]) =
    implicitly[Numeric[T]].toFloat(losses.sum) / losses.length
}

case class Weight[T](w: Tensor[T], b: Tensor[T], f: Activation[T])

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Neuron[T](
    x: Tensor[T],
    z: Tensor[T],
    a: Tensor[T]
) //TODO: add error property for backpropagation

case class Sequential[T: ClassTag: RandomGen: Numeric: TypeTag, U](
    lossFunc: Loss[T],
    optimizer: Optimizer[T],
    learningRate: T,
    batchSize: Int = 32,
    layers: List[Layer[T]] = Nil,
    weights: List[Weight[T]] = Nil,
    losses: List[T] = Nil
) extends Model[T] {
  self =>

  def currentWeights: List[(Tensor[T], Tensor[T])] =
    weights.map(w => w.w -> w.b)

  def predict(x: Tensor[T]): Tensor[T] = activate(x, weights).last.a

  def add(l: Layer[T]) =
    self.copy(layers = layers :+ l)

  def initialWeights(inputs: Int): List[Weight[T]] = {
    val (weights, _) =
      layers.foldLeft(List.empty[Weight[T]], inputs) {
        case ((acc, inputs), layer) =>
          val w = random2D(inputs, layer.units)
          val b = zeros(layer.units)
          (acc :+ Weight(w, b, layer.f), layer.units)
      }
    weights
  }

  private def activate(
      input: Tensor[T],
      weights: List[Weight[T]]
  ): Array[Neuron[T]] =
    weights
      .foldLeft(input, ArrayBuffer.empty[Neuron[T]]) {
        case ((x, acc), Weight(w, b, activation)) =>
          // println(s"w = $w")
          // println(s"x = ${x}")
          // println(s"b = $b")
          // println(s"res = ${x * w}")
          val z = (x * w) + b
          // println(s"res2 = $z")
          val a = activation(z)
          (a, acc :+ Neuron(x, z, a))
      }
      ._2
      .toArray

  def updateWeights(
      weights: List[Weight[T]],
      activations: Array[Neuron[T]],
      error: Tensor[T]
  ) = {
    // println(s"initial error = ${error}")
    weights
      .zip(activations)
      .foldRight(
        List.empty[Weight[T]],
        error,
        None: Option[Tensor[T]]
      ) {
        case (
              (Weight(w, b, actFunc), neuron),
              (ws, prevDelta, prevWeight)
            ) =>
          // println(s"prevDelta = $prevDelta")
          // println(s"prevWeight = $prevWeight")
          // println(s"z.T = ${neuron.z.T}")
          // println(s"w = $w")
          val delta = prevWeight match {
            case Some(pw) =>
              (prevDelta * pw.T) multiply actFunc.derivative(neuron.z)
            case None => prevDelta
          }
          // println(s"delta = $delta")
          // println(s"neuron.x.T = ${neuron.x.T}")

          val wPartialDerivative = neuron.x.T * delta
          // println(s"wPartialDerivative = $wPartialDerivative")
          // println(s"b = $b")
          val newWeight = w - (learningRate * wPartialDerivative)
          // println(s"newWeight = $newWeight")
          val newBias = b - (learningRate * delta.sum)
          // println(s"newBias = $newBias")
          val updated = Weight(newWeight, newBias, actFunc) +: ws
          (updated, delta, Some(w))
      }
      ._1
  }

  private def correctPredictions(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): Int =
    actual.data.zip(predicted.data).foldLeft(0) { case (acc, (y, yhat)) =>
      val normalized = if (yhat.toFloat > 0.5) 1 else 0
      acc + (if (y == normalized) 1 else 0)
    }

  private def trainEpoch(
      xBatches: List[(Array[Array[T]], Array[T])],
      weights: List[Weight[T]],
      epoch: Int
  ) = {
    val (w, l, positives) =
      xBatches.foldLeft(weights, List.empty[T], 0f) { // mini-batch SGD
        case ((weights, batchLoss, positives), (batch, actualBatch)) =>
          // forward
          // println(s"batch size = ${batch.length}")
          val activations = activate(Tensor2D(batch), weights)
          //println(s"activations = ${activations.mkString("\n---------\n")}")
          val actual = Tensor1D(actualBatch)
          val predicted = activations.last.a.as1D
          if (epoch % 20 == 0) {
            val positive = predicted.data.filter(_.toFloat > 0.5)
            // if (positive.nonEmpty) println(s"positive = ${positive.mkString(",")}")
            // println(s"predicted = $predicted")
            // println(s"actual = $actual")
          }
          val error = predicted - actual
          //println(s"error = $error")
          val loss = lossFunc(actual, predicted)
          // println(s"loss = $loss")
          // backward
          val updated = updateWeights(weights, activations, error)
          val correct = correctPredictions(actual, predicted)
          (updated, batchLoss :+ loss, positives + correct)
      }
    val avgLoss = getAvgLoss(l)
    (w, transformAny[Float, T](avgLoss), positives)
  }

  def train(x: Tensor[T], y: Tensor1D[T], epochs: Int): Model[T] = {
    val inputs = x.cols
    lazy val actualBatches = batchColumn(y, batchSize).toArray
    lazy val xBatches = batches(x, batchSize).zip(actualBatches).toList

    val w = getWeights(inputs)
    val (updatedWeights, epochLosses) =
      (0 until epochs).foldLeft((w, List.empty[T])) {
        case ((weights, losses), epoch) =>
          val (w, l, positives) = trainEpoch(xBatches, weights, epoch)
          val accuracy = positives / x.length
          println(
            s"epoch: ${epoch + 1}/$epochs, avg. loss: $l, accuracy: $accuracy"
          )
          (w, losses :+ l)
      }
    // println(s"losses count: ${epochLosses.length}")
    // println(s"losses:\n${epochLosses.mkString("\n")}")

    copy(weights = updatedWeights, losses = epochLosses)
  }

  private def getWeights(inputs: Int) =
    if (weights == Nil) initialWeights(inputs) else weights
}

def prepareData() = {
  //1,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1 = length 14
  val data = TextLoader(Path.of("data", "Churn_Modelling.csv")).load()
  val y = data.col[Float](-1)

  var xRaw = data.cols[String](3, -1)
  val encoder = LabelEncoder[String]().fit(xRaw.col(2))
  xRaw = encoder.transform(xRaw, 2)
  val hotEncoder = OneHotEncoder[String, Float]().fit(xRaw.col(1))
  xRaw = hotEncoder.transform(xRaw, 1)
  val xFloat = transform[Float](xRaw.data)

  val scaler = StandardScaler[Float]().fit(xFloat)
  val x = scaler.transform(xFloat)
  (x, y)
}

val ann =
  Sequential(mse, miniBatchGradientDescent, learningRate = 0.001f)
    .add(Dense(6)(relu))
    .add(Dense(6)(relu))
    .add(Dense()(sigmoid))

val (x, y) = prepareData()
val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)

val model = ann.train(xTrain, yTrain.as1D, epochs = 100)
println("\nweights:\n\n" + model.currentWeights.mkString("\n\n"))
