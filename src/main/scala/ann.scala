import Model._
import RandomGen._
import Sequential._
import converter.transformAny
import ops._

import scala.collection.mutable.ArrayBuffer
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

sealed trait ActivationFunc[T] extends (Tensor[T] => Tensor[T]):
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

object ActivationFunc:
  val relu: ActivationFunc[Float] = new ActivationFunc[Float] {

    override def apply(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(x, (t: Float) => math.max(0, t))

    override def derivative(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(x, (t: Float) => if t < 0 then 0 else 1)
  }

  val sigmoid: ActivationFunc[Float] = new ActivationFunc[Float] {

    override def apply(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(x, (t: Float) => 1 / (1 + math.exp(-t).toFloat))

    override def derivative(x: Tensor[Float]): Tensor[Float] =
      Tensor.map(
        x,
        (t: Float) =>
          math.exp(-t).toFloat / math.pow(1 + math.exp(-t).toFloat, 2).toFloat
      )
  }
sealed trait Loss[T]:
  def apply(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): T

object Loss:
  val meanSquaredError: Loss[Float] = new Loss[Float] {
    override def apply(
        actual: Tensor1D[Float],
        predicted: Tensor1D[Float]
    ): Float =
      var sumScore = 0.0
      for i <- actual.data.indices do
        val y1 = actual.data(i)
        val y2 = predicted.data(i)
        sumScore += math.pow(y1 - y2, 2)
      val meanSumScore = 1.0 / actual.length * sumScore
      meanSumScore.toFloat
  }

  val binaryCrossEntropy: Loss[Float] = new Loss[Float] {
    override def apply(
        actual: Tensor1D[Float],
        predicted: Tensor1D[Float]
    ): Float =
      var sumScore = 0.0
      for i <- actual.data.indices do
        val y1 = actual.data(i)
        val y2 = predicted.data(i)
        sumScore += y1 * math.log(1e-15 + y2.toDouble)
      val meanSumScore = 1.0 / actual.length * sumScore
      -meanSumScore.toFloat
  }
  
sealed trait Optimizer[U]:

  def updateWeights[T: Numeric: ClassTag](
      weights: List[Weight[T]],
      activations: List[Activation[T]],
      error: Tensor[T],
      learningRate: T
  ): List[Weight[T]]

sealed trait MiniBatchGD

object optimizers:
  given miniBatchGradientDescent: Optimizer[MiniBatchGD] with
    override def updateWeights[T: Numeric: ClassTag](
        weights: List[Weight[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        learningRate: T
    ): List[Weight[T]] =
      weights
        .zip(activations)
        .foldRight(
          List.empty[Weight[T]],
          error,
          None: Option[Tensor[T]]
        ) {
          case (
                (Weight(w, b, f, u), Activation(x, z, _)),
                (ws, prevDelta, prevWeight)
              ) =>
            val delta = (prevWeight match {
              case Some(pw) => prevDelta * pw.T
              case None     => prevDelta
            }) multiply f.derivative(z)

            val partialDerivative = x.T * delta
            val newWeight = w - (learningRate * partialDerivative)
            val newBias = b - (learningRate * delta.sum)
            val updated = Weight(newWeight, newBias, f, u) +: ws
            (updated, delta, Some(w))
        }
        ._1    

sealed trait Layer[T]:
  def units: Int
  def f: ActivationFunc[T]

case class Dense[T](
    f: ActivationFunc[T],
    units: Int = 1
) extends Layer[T]

case class Weight[T](
    w: Tensor[T],
    b: Tensor[T],
    f: ActivationFunc[T],
    units: Int
)

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Activation[T](x: Tensor[T], z: Tensor[T], a: Tensor[T])

sealed trait Model[T]:
  def reset(): Model[T]
  def train(x: Tensor[T], y: Tensor1D[T], epochs: Int): Model[T]
  def currentWeights: List[(Tensor[T], Tensor[T])]
  def predict(x: Tensor[T]): Tensor[T]
  def losses: List[T]

object Model:
  def getAvgLoss[T](losses: List[T])(using num: Numeric[T]): Float =
    num.toFloat(losses.sum) / losses.length

object Sequential:
  def activate[T: Numeric: ClassTag](
      input: Tensor[T],
      weights: List[Weight[T]]
  ): List[Activation[T]] =
    weights
      .foldLeft(input, ArrayBuffer.empty[Activation[T]]) {
        case ((x, acc), Weight(w, b, f, _)) =>
          val z = x * w + b
          val a = f(z)
          (a, acc :+ Activation(x, z, a))
      }
      ._2
      .toList

case class Sequential[T: ClassTag: RandomGen: Numeric, U: Optimizer](
    lossFunc: Loss[T],
    learningRate: T,
    metric: Metric[T],
    batchSize: Int = 16,
    layerStack: Int => List[Weight[T]] = (_: Int) => List.empty[Weight[T]],    
    weights: List[Weight[T]] = Nil,
    losses: List[T] = Nil
) extends Model[T]:

  def currentWeights: List[(Tensor[T], Tensor[T])] =
    weights.map(w => w.w -> w.b)

  def predict(x: Tensor[T]): Tensor[T] =
    activate(x, weights).last.a

  def add(layer: Layer[T]): Sequential[T, U] =
    copy(layerStack = (inputs) => {
      val currentWeights = layerStack(inputs)
      val prevInput =
        currentWeights.reverse.headOption.map(_.units).getOrElse(inputs)
      val w = random2D(prevInput, layer.units)
      val b = zeros(layer.units)
      (currentWeights :+ Weight(w, b, layer.f, layer.units))
    })

  private def trainEpoch(
      xBatches: Array[(Array[Array[T]], Array[T])],
      weights: List[Weight[T]]
  ) =
    val (w, l, metricValue) =
      xBatches.foldLeft(weights, List.empty[T], 0) {
        case ((weights, batchLoss, metricAcc), (batch, actualBatch)) =>
          // forward
          val activations = activate(batch.as2D, weights)
          val actual = actualBatch.as1D
          val predicted = activations.last.a.as1D
          val error = predicted - actual
          val loss = lossFunc(actual, predicted)

          // backward
          val updated = implicitly[Optimizer[U]].updateWeights(
            weights,
            activations,
            error.T,
            learningRate
          )
          val metricValue = metric.calculate(actual, predicted)
          (updated, batchLoss :+ loss, metricAcc + metricValue)
      }
    val avgLoss = transformAny[Float, T](getAvgLoss(l))
    (w, avgLoss, metricValue)

  def train(x: Tensor[T], y: Tensor1D[T], epochs: Int): Model[T] =
    lazy val inputs = x.cols
    lazy val actualBatches = y.batchColumn(batchSize).toArray
    lazy val xBatches = x.batches(batchSize).zip(actualBatches).toArray
    lazy val w = getWeights(inputs)

    val (updatedWeights, epochLosses) =
      (1 to epochs).foldLeft((w, List.empty[T])) {
        case ((weights, losses), epoch) =>
          val (w, avgLoss, metricValue) = trainEpoch(xBatches, weights)
          val metricAvg = metric.average(x.length, metricValue)
          println(
            s"epoch: $epoch/$epochs, avg. loss: $avgLoss, ${metric.name}: $metricAvg"
          )
          (w, losses :+ avgLoss)
      }
    copy(weights = updatedWeights, losses = epochLosses)

  def reset(): Model[T] =
    copy(weights = Nil)

  private def getWeights(inputs: Int) =
    if weights == Nil then layerStack(inputs)
    else weights

trait Metric[T]:
  val name: String

  def calculate(
      actual: Tensor1D[T],
      predicted: Tensor1D[T]
  ): Int

  def average(count: Int, correct: Int): Double

  def apply(actual: Tensor1D[T], predicted: Tensor1D[T]): Double =
    val correct = calculate(actual, predicted)
    average(actual.length, correct)

object Metric:
  def predictedToBinary[T](v: T)(using num: Numeric[T]): Byte =
    if num.toDouble(v) > 0.5 then 1 else 0

  def accuracyMetric[T: Numeric]: Metric[T] = new Metric[T] {
    val name = "accuracy"

    def calculate(
        actual: Tensor1D[T],
        predicted: Tensor1D[T]
    ): Int =
      actual.data.zip(predicted.data).foldLeft(0) { case (acc, (y, yHat)) =>
        val normalized = predictedToBinary(yHat)
        acc + (if y == implicitly[Numeric[T]].fromInt(normalized) then 1 else 0)
      }

    def average(count: Int, correct: Int): Double =
      correct.toDouble / count
  }
