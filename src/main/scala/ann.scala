import Model._
import RandomGen._
import Sequential._
import converter.transformAny
import ops._

import scala.collection.mutable.ArrayBuffer
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

trait GradientClipping[T] extends (Tensor[T] => Tensor[T]) 

object GradientClipping:
  def clipByValue[T: Numeric: ClassTag](value: T) = new GradientClipping[T] {
    def apply(t: Tensor[T]): Tensor[T] = t.clipInRange(-value, value)
  }
  
  def noClipping[T]: GradientClipping[T] = t => t

trait ActivationFunc[T] extends (Tensor[T] => Tensor[T]):
  val name: String
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

object ActivationFunc:
  def relu[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T] {

    override def apply(x: Tensor[T]): Tensor[T] =
      Tensor.map(x, t => transformAny[Double, T](math.max(0, n.toDouble(t))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      Tensor.map(x, t => transformAny[Double, T](if n.toDouble(t) < 0 then 0 else 1))

    override val name = "relu"
  }

  def sigmoid[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T] {

    override def apply(x: Tensor[T]): Tensor[T] =
      Tensor.map(x, t => transformAny[Double,T](1 / (1 + math.exp(-n.toDouble(t)))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      Tensor.map(
        x,
        t =>
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
        val sum = (t1, t2).map2(f).sum
        (sum, t1.length)
      case (Tensor2D(a), Tensor2D(b)) =>
        val size = t1.length * t1.cols        
        val sum = (t1, t2).map2(f).sum
        (sum, size)
      case (Tensor0D(a), Tensor0D(b)) =>        
        (f(a, b), 1)
      case _ => 
        sys.error(s"Both tensors must be the same shape: ${t1.sizes} != ${t2.sizes}")

  def meanSquareError[T: ClassTag: Numeric] = new Loss[T] {
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

  def binaryCrossEntropy[T: ClassTag](using n: Numeric[T]) = new Loss[T] {
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
      learningRate: T,
      clip: GradientClipping[T]
  ): List[Weight[T]]

type SimpleGD

object optimizers:
  given Optimizer[SimpleGD] with
    override def updateWeights[T: Numeric: ClassTag](
        weights: List[Weight[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        learningRate: T,
        clip: GradientClipping[T]
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

            val partialDerivative = clip(x.T * delta)
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
    f: ActivationFunc[T] = ActivationFunc.noActivation[T],
    units: Int = 1
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
  def weights: List[Weight[T]]
  def predict(x: Tensor[T], weightList: List[Weight[T]] = weights): Tensor[T]
  def losses: List[T]
  def metricValues: List[(Metric[T], List[Double])]

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

case class Sequential[T: ClassTag: RandomGen: Numeric, U](
    lossFunc: Loss[T],
    learningRate: T,
    metrics: List[Metric[T]] = Nil,
    batchSize: Int = 16,
    weightStack: Int => List[Weight[T]] = (_: Int) => List.empty[Weight[T]],    
    weights: List[Weight[T]] = Nil,
    losses: List[T] = Nil,
    metricValues: List[(Metric[T], List[Double])] = Nil,
    gradientClipping: GradientClipping[T] = GradientClipping.noClipping[T]
)(using optimizer: Optimizer[U]) extends Model[T]:

  def predict(x: Tensor[T], w: List[Weight[T]] = weights): Tensor[T] =
    activate(x, w).last.a

  def loss(x: Tensor[T], y: Tensor[T], w: List[Weight[T]]): T =
    val predicted = predict(x, w)    
    lossFunc(y, predicted)  

  def add(layer: Layer[T]): Sequential[T, U] =
    copy(weightStack = (inputs) => {
      val currentWeights = weightStack(inputs)
      val prevInput = currentWeights.lastOption.map(_.units).getOrElse(inputs)
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
        case ((weights, batchLoss, epochMetrics), (xBatch, yBatch)) =>
          // forward
          val activations = activate(xBatch.as2D, weights)
          val actual = yBatch.as2D          
          val predicted = activations.last.a          
          val error = predicted - actual          
          val loss = lossFunc(actual, predicted)

          // backward
          val updated = optimizer.updateWeights(
            weights,
            activations,
            error,
            learningRate,
            gradientClipping
          )
          val metricValues = metrics
            .map(_.calculate(actual, predicted))
            .zip(epochMetrics).map(_ + _)
          (updated, batchLoss :+ loss, metricValues)
      }    
    (w, getAvgLoss(l), metricValue)

  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T] =
    lazy val inputs = x.cols
    lazy val actualBatches = y.batches(batchSize).toArray
    lazy val xBatches = x.batches(batchSize).zip(actualBatches).toArray
    lazy val w = getOrInitWeights(inputs)
    
    val emptyMetrics = metrics.map(_ -> List.empty[Double])
    val (updatedWeights, epochLosses, metricValues) =
      (1 to epochs).foldLeft(w, List.empty[T], emptyMetrics) {
        case ((weights, losses, metricsList), epoch) =>
          val (w, avgLoss, epochMetric) = trainEpoch(xBatches, weights)
          
          val epochMetricAvg = metrics.zip(epochMetric).map((m, value) => m -> m.average(x.length, value))
          printMetrics(epoch, epochs, avgLoss, epochMetricAvg)          
          val epochMetrics = epochMetricAvg.zip(metricsList).map { 
            case ((epochMetric, v), (trainingMetric, values)) => trainingMetric -> (values :+ v)
          }

          (w, losses :+ avgLoss, epochMetrics)
      }
    copy(weights = updatedWeights, losses = epochLosses, metricValues = metricValues)

  private def printMetrics(epoch: Int, epochs: Int, avgLoss: T, values: List[(Metric[T], Double)]) = 
    val metricsStat = values
      .map((m, avg) => s"${m.name}: $avg")
      .mkString(", metrics: [", ";", "]")
    println(
      s"epoch: $epoch/$epochs, avg. loss: $avgLoss${if (metrics.nonEmpty) metricsStat else ""}"
    )

  def reset(): Model[T] =
    copy(weights = Nil)

  private def getOrInitWeights(inputs: Int) =
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

  def accuracyBinaryClassification[T: ClassTag: Numeric]: Metric[T] = new Metric[T] {
    val name = "accuracy"

    def calculate(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): Int =      
        val predictedNormalized = predicted.map(predictedToBinary)
        actual.equalRows(predictedNormalized)      
  }
