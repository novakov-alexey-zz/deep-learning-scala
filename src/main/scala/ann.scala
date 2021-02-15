import Model._
import RandomGen._
import Sequential._
import converter.transformAny
import ops._

import scala.collection.mutable.ArrayBuffer
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

trait ActivationFunc[T] extends (Tensor[T] => Tensor[T]):
  val name: String
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

object ActivationFunc:
  def relu[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T] {

    override def apply(x: Tensor[T]): Tensor[T] =
      Tensor.map(x, (t: T) => transformAny[Double, T](math.max(0, n.toDouble(t))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      Tensor.map(x, (t: T) => transformAny[Double, T](if n.toDouble(t) < 0 then 0 else 1))

    override val name = "relu"
  }

  def sigmoid[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T] {

    override def apply(x: Tensor[T]): Tensor[T] =
      Tensor.map(x, (t: T) => transformAny[Double,T](1 / (1 + math.exp(-n.toDouble(t)))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      Tensor.map(
        x,
        (t: T) =>
          transformAny[Double, T](math.exp(-n.toDouble(t)) / math.pow(1 + math.exp(-n.toDouble(t)), 2))
      )
    
    override val name = "sigmoid"
  }

  def noActivation[T] = new ActivationFunc[T] {
    override def apply(x: Tensor[T]): Tensor[T] = x
    override def derivative(x: Tensor[T]): Tensor[T] = x
    override val name = "no-activation"    
  }


trait Loss[T]:
  def apply(
      actual: Tensor[T],
      predicted: Tensor[T]
  ): T

object Loss:
  private def calcMetric[T: Numeric: ClassTag](
    t1: Tensor[T], t2: Tensor[T], f: (T, T) => T
  ) = 
    (t1, t2) match
      case (Tensor1D(a), Tensor1D(b)) => 
        val size = t1.length
        val sum = (t1, t2).map2(f).sum
        (sum, size)
      case (Tensor2D(a), Tensor2D(b)) =>
        val size = t1.length * t1.cols        
        val sum = (t1, t2).map2(f).sum
        (sum, size)
      case (Tensor0D(a), Tensor0D(b)) =>        
        (f(a, b), 1)
      case _ => sys.error(s"Both tensors must be the same shape: ${t1.sizes} != ${t2.sizes}")

  def meanSquareError[T: ClassTag: Numeric]: Loss[T] = new Loss[T] {
    def calc(a: T, b: T): T =      
      transformAny[Double, T](math.pow(transformAny[T, Double](a - b), 2)) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =      
      val (sumScore, count) = calcMetric(actual, predicted, calc)      
      val meanSumScore = 1.0 / count * transformAny[T, Double](sumScore)
      transformAny(meanSumScore)
  }

  def binaryCrossEntropy[T: ClassTag](using n: Numeric[T]): Loss[T] = new Loss[T] {
    def calc(a: T, b: T): T = 
      transformAny[Double, T](n.toDouble(a) * math.log(1e-15 + n.toDouble(b))) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =
      val (sumScore, count) = calcMetric(actual, predicted, calc)        
      val meanSumScore = 1.0 / count * transformAny[T, Double](sumScore)
      transformAny(-meanSumScore)      
  }
  
sealed trait Optimizer[U]:

  def updateWeights[T: Numeric: ClassTag](
      weights: List[Weight[T]],
      activations: List[Activation[T]],
      error: Tensor[T],
      learningRate: T
  ): List[Weight[T]]

type SimpleGD

object optimizers:
  given Optimizer[SimpleGD] with
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
    f: ActivationFunc[T] = ActivationFunc.noActivation[T],
    units: Int = 1
) extends Layer[T]

case class Weight[T](
    w: Tensor[T],
    b: Tensor[T],
    f: ActivationFunc[T],
    units: Int
) {
  override def toString() = 
    s"\n(\nweight = $w,\nbias = $b,\nf = ${f.name},\nunits = $units)"
}

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Activation[T](x: Tensor[T], z: Tensor[T], a: Tensor[T])

sealed trait Model[T]:
  def reset(): Model[T]
  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T]
  def currentWeights: List[Weight[T]]
  def predict(x: Tensor[T]): Tensor[T]
  def losses: List[T]
  def metricValues: Map[String, List[Double]]

object Model:
  def getAvgLoss[T: ClassTag](losses: List[T])(using num: Numeric[T]): T =
    transformAny[Float, T](num.toFloat(losses.sum) / losses.length)

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
    metrics: List[Metric[T]] = Nil,
    batchSize: Int = 16,
    weightStack: Int => List[Weight[T]] = (_: Int) => List.empty[Weight[T]],    
    weights: List[Weight[T]] = Nil,
    losses: List[T] = Nil,
    metricValues: Map[String, List[Double]] = Map.empty
) extends Model[T]:

  def currentWeights: List[Weight[T]] = weights

  def predict(x: Tensor[T]): Tensor[T] =
    activate(x, weights).last.a

  def add(layer: Layer[T]): Sequential[T, U] =
    copy(weightStack = (inputs) => {
      val currentWeights = weightStack(inputs)
      val prevInput =
        currentWeights.reverse.headOption.map(_.units).getOrElse(inputs)
      val w = random2D(prevInput, layer.units)
      val b = zeros(layer.units)
      (currentWeights :+ Weight(w, b, layer.f, layer.units))
    })

  private def trainEpoch(
      batches: Array[(Array[Array[T]], Array[Array[T]])],
      weights: List[Weight[T]]
  ) =
    val (w, l, metricValue) =
      batches.foldLeft(weights, List.empty[T], List.fill(metrics.length)(0)) {
        case ((weights, batchLoss, metricAcc), (xBatch, yBatch)) =>
          // forward
          val activations = activate(xBatch.as2D, weights)
          val actual = yBatch.as2D          
          val predicted = activations.last.a          
          val error = predicted - actual          
          val loss = lossFunc(actual, predicted)

          // backward
          val updated = summon[Optimizer[U]].updateWeights(
            weights,
            activations,
            error,
            learningRate
          )
          val metricValues = metrics.map(m => m.calculate(actual, predicted))
          (updated, batchLoss :+ loss, metricAcc.zip(metricValues).map(_ + _))
      }    
    (w, getAvgLoss(l), metricValue)

  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T] =
    lazy val inputs = x.cols
    lazy val actualBatches = y.batches(batchSize).toArray
    lazy val xBatches = x.batches(batchSize).zip(actualBatches).toArray
    lazy val w = getWeights(inputs)

    val (updatedWeights, epochLosses, metricValues) =
      (1 to epochs).foldLeft((w, List.empty[T], Map.empty[String, List[Double]])) {
        case ((weights, losses, metricsMap), epoch) =>
          val (w, avgLoss, metricValue) = trainEpoch(xBatches, weights)
          val metricAvg = metrics.zip(metricValue).map((m, value) => m -> m.average(x.length, value))
          val metricsStat = metricAvg.map((m, avg) => s"${m.name}: $avg").mkString(", metrics: [", ";", "]")
          println(
            s"epoch: $epoch/$epochs, avg. loss: $avgLoss${if (metrics.nonEmpty) metricsStat else ""}"
          )
          val epochMetrics = metricAvg.foldLeft(Map.empty[String, List[Double]]){ case (acc, (m, v)) => 
            val updated = metricsMap.getOrElse(m.name, List.empty[Double]) :+ v
            acc + (m.name -> updated)
          }
          (w, losses :+ avgLoss, epochMetrics)
      }
    copy(weights = updatedWeights, losses = epochLosses, metricValues = metricValues)

  def reset(): Model[T] =
    copy(weights = Nil)

  private def getWeights(inputs: Int) =
    if weights == Nil then weightStack(inputs)
    else weights

trait Metric[T]:
  val name: String

  def calculate(
      actual: Tensor[T],
      predicted: Tensor[T]
  ): Int

  def average(count: Int, correct: Int): Double =
    correct.toDouble / count

  def apply(actual: Tensor[T], predicted: Tensor[T]): Double =
    val correct = calculate(actual, predicted)
    average(actual.length, correct)

object Metric:
  def predictedToBinary[T](v: T)(using n: Numeric[T]): T =
    if n.toDouble(v) > 0.5 then n.one else n.zero

  def accuracyMetric[T: ClassTag: Numeric]: Metric[T] = new Metric[T] {
    val name = "accuracy"

    def calculate(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): Int =      
        val predictedNormalized = predicted.map(predictedToBinary)
        actual.equalRows(predictedNormalized)      
  }
