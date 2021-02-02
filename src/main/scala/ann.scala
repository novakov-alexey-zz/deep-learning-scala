import Model._
import RandomGen._
import Sequential._
import converter.transformAny
import ops._

import scala.collection.mutable.ArrayBuffer
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

sealed trait Activation[T] extends (Tensor[T] => Tensor[T]) {
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]
}

object Activation {
  implicit val relu: Activation[Float] = new Activation[Float] {

    override def apply(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(x, (t: Float) => math.max(0, t))

    override def derivative(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(x, (t: Float) => if (t < 0) 0 else 1)
  }

  implicit val sigmoid: Activation[Float] = new Activation[Float] {

    override def apply(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(x, (t: Float) => 1 / (1 + math.exp(-t).toFloat))

    override def derivative(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(
        x,
        (t: Float) =>
          math.exp(-t).toFloat / math.pow(1 + math.exp(-t).toFloat, 2).toFloat
      )
  }
}
sealed trait Loss[T] {
  def apply(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): T
}

object Loss {
  implicit val meanSquaredError: Loss[Float] = new Loss[Float] {
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
  }

  implicit val binaryCrossEntropy: Loss[Float] = new Loss[Float] {
    override def apply(
        actual: Tensor1D[Float],
        predicted: Tensor1D[Float]
    ): Float = {
      var sumScore = 0.0
      for (i <- actual.data.indices) {
        val y1 = actual.data(i)
        val y2 = predicted.data(i)
        sumScore += y1 * math.log(1e-15 + y2.toDouble)
      }
      val meanSumScore = 1.0 / actual.length * sumScore
      -meanSumScore.toFloat
    }
  }
}
sealed trait Optimizer[U] {

  def updateWeights[T: Numeric: ClassTag](
      weights: List[Weight[T]],
      neurons: List[Neuron[T]],
      error: Tensor[T],
      learningRate: T
  ): List[Weight[T]]
}

sealed trait MiniBatchGD

object optimizers {
  implicit val miniBatchGradientDescent: Optimizer[MiniBatchGD] =
    new Optimizer[MiniBatchGD] {

      override def updateWeights[T: Numeric: ClassTag](
          weights: List[Weight[T]],
          neurons: List[Neuron[T]],
          error: Tensor[T],
          learningRate: T
      ): List[Weight[T]] =
        weights
          .zip(neurons)
          .foldRight(
            List.empty[Weight[T]],
            error,
            None: Option[Tensor[T]]
          ) {
            case (
                  (Weight(w, b, f), Neuron(x, z, _)),
                  (ws, prevDelta, prevWeight)
                ) =>
              val delta = (prevWeight match {
                case Some(pw) => prevDelta * pw.T
                case None     => prevDelta
              }) multiply f.derivative(z)

              val partialDerivative = x.T * delta
              val newWeight = w - (learningRate * partialDerivative)
              val newBias = b - (learningRate * delta.sum)
              val updated = Weight(newWeight, newBias, f) +: ws
              (updated, delta, Some(w))
          }
          ._1
    }
}

sealed trait Layer[T] {
  def units: Int
  def f: Activation[T]
}

case class Dense[T](
    f: Activation[T],
    units: Int = 1
) extends Layer[T]

sealed trait Model[T] {
  def layers: List[Layer[T]]
  def train(x: Tensor[T], y: Tensor1D[T], epochs: Int): Model[T]
  def currentWeights: List[(Tensor[T], Tensor[T])]
  def predict(x: Tensor[T]): Tensor[T]
  def losses: List[T]
}

object Model {
  def getAvgLoss[T: Numeric](losses: List[T]): Float =
    implicitly[Numeric[T]].toFloat(losses.sum) / losses.length
}

case class Weight[T](w: Tensor[T], b: Tensor[T], f: Activation[T])

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Neuron[T](x: Tensor[T], z: Tensor[T], a: Tensor[T])

object Sequential {
  def activate[T: Numeric: ClassTag](
      input: Tensor[T],
      weights: List[Weight[T]]
  ): List[Neuron[T]] =
    weights
      .foldLeft(input, ArrayBuffer.empty[Neuron[T]]) {
        case ((x, acc), Weight(w, b, f)) =>
          val z = x * w + b
          val a = f(z)
          (a, acc :+ Neuron(x, z, a))
      }
      ._2
      .toList

  def initialWeights[T: RandomGen: Numeric: ClassTag](
      layers: List[Layer[T]],
      inputs: Int
  ): List[Weight[T]] =
    layers
      .foldLeft(List.empty[Weight[T]], inputs) { case ((acc, inputs), layer) =>
        val w = random2D(inputs, layer.units)
        val b = zeros(layer.units)
        (acc :+ Weight(w, b, layer.f), layer.units)
      }
      ._1
}

case class Sequential[T: ClassTag: RandomGen: Numeric, U: Optimizer](
    lossFunc: Loss[T],
    learningRate: T,
    metric: Metric[T],
    batchSize: Int = 16,
    layers: List[Layer[T]] = Nil,
    weights: List[Weight[T]] = Nil,
    losses: List[T] = Nil
) extends Model[T] {

  def currentWeights: List[(Tensor[T], Tensor[T])] =
    weights.map(w => w.w -> w.b)

  def predict(x: Tensor[T]): Tensor[T] =
    activate(x, weights).last.a

  def add(l: Layer[T]): Sequential[T, U] =
    copy(layers = layers :+ l)

  private def trainEpoch(
      xBatches: List[(Array[Array[T]], Array[T])],
      weights: List[Weight[T]]
  ) = {
    val (w, l, positives) =
      xBatches.foldLeft(weights, List.empty[T], 0) {
        case ((weights, batchLoss, positives), (batch, actualBatch)) =>
          // forward
          val neurons = activate(Tensor2D(batch), weights)
          val actual = Tensor1D(actualBatch)
          val predicted = neurons.last.a.as1D
          val error = predicted - actual
          val loss = lossFunc(actual, predicted)

          // backward
          val updated = implicitly[Optimizer[U]].updateWeights(
            weights,
            neurons,
            error.T,
            learningRate
          )
          val correct = metric.correctPredictions(actual, predicted)
          (updated, batchLoss :+ loss, positives + correct)
      }
    // println(l)
    val avgLoss = transformAny[Float, T](getAvgLoss(l))
    (w, avgLoss, positives)
  }

  def train(x: Tensor[T], y: Tensor1D[T], epochs: Int): Model[T] = {
    lazy val inputs = x.cols
    lazy val actualBatches = y.batchColumn(batchSize).toArray
    lazy val xBatches = x.batches(batchSize).zip(actualBatches).toList
    lazy val w = getWeights(inputs)

    val (updatedWeights, epochLosses) =
      (0 until epochs).foldLeft((w, List.empty[T])) {
        case ((weights, losses), epoch) =>
          val (w, l, positives) = trainEpoch(xBatches, weights)
          val metricValue = metric.result(x.length, positives)
          println(
            s"epoch: ${epoch + 1}/$epochs, avg. loss: $l, ${metric.name}: $metricValue"
          )
          (w, losses :+ l)
      }
    copy(weights = updatedWeights, losses = epochLosses)
  }

  private def getWeights(inputs: Int) =
    if (weights == Nil) initialWeights(layers, inputs) else weights
}

trait Metric[T] {
  val name: String

  def correctPredictions(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): Int

  def result(count: Int, correct: Int): Double

  def apply(actual: Tensor1D[T], predicted: Tensor1D[T]): Double = {
    val correct = correctPredictions(actual, predicted)
    result(actual.length, correct)
  }
}

object Metric {
  def predictedToBinary[T: Numeric](v: T): Byte =
    if (implicitly[Numeric[T]].toDouble(v) > 0.5) 1 else 0

  def accuracyMetric[T: Numeric]: Metric[T] = new Metric[T] {
    val name = "accuracy"

    def correctPredictions(
        actual: Tensor1D[T],
        predicted: Tensor1D[T]
    ): Int =
      actual.data.zip(predicted.data).foldLeft(0) { case (acc, (y, yHat)) =>
        val normalized = predictedToBinary(yHat)
        acc + (if (y == implicitly[Numeric[T]].fromInt(normalized)) 1 else 0)
      }

    def result(count: Int, correct: Int): Double =
      correct.toDouble / count
  }
}
