package ml.network 

import ml.network.RandomGen._
import ml.transformation.transformAny
import ml.tensors.api._
import ml.tensors.ops._

import Model._
import Sequential._
import ActivationFuncApi._
import GradientClippingApi._

import scala.collection.mutable.ArrayBuffer
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

trait GradientClipping[T] extends (Tensor[T] => Tensor[T]) 

trait GradientClippingApi:
  def clipByValue[T: Fractional: ClassTag](value: T): GradientClipping[T] = 
    _.clipInRange(-value, value)

  def noClipping[T]: GradientClipping[T] = t => t

object GradientClippingApi extends GradientClippingApi  

trait ActivationFunc[T] extends (Tensor[T] => Tensor[T]):
  val name: String
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

trait ActivationFuncApi:
  def relu[T: ClassTag](using n: Fractional[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => transformAny[Double, T](math.max(0, n.toDouble(t))))      

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => transformAny[Double, T](if n.toDouble(t) < 0 then 0 else 1))

    override val name = "relu"
  
  def sigmoid[T: ClassTag](using n: Fractional[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => transformAny[Double,T](1 / (1 + math.exp(-n.toDouble(t)))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => transformAny[Double, T](
          math.exp(-n.toDouble(t)) / math.pow(1 + math.exp(-n.toDouble(t)), 2)
        ))
    
    override val name = "sigmoid"  

  def noActivation[T] = new ActivationFunc[T]:
    override def apply(x: Tensor[T]): Tensor[T] = x
    override def derivative(x: Tensor[T]): Tensor[T] = x
    override val name = "no-activation"  

object ActivationFuncApi extends ActivationFuncApi

trait Loss[T]:
  def apply(
      actual: Tensor[T],
      predicted: Tensor[T]
  ): T

trait LossApi:
  private def calcMetric[T: Fractional: ClassTag](
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

  def meanSquareError[T: ClassTag: Fractional] = new Loss[T]:
    def calc(a: T, b: T): T =      
      transformAny[Double, T](math.pow(transformAny[T, Double](a - b), 2)) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =      
      val (sumScore, count) = calcMetric(actual, predicted, calc)      
      val meanSumScore = 1.0 / count * transformAny[T, Double](sumScore)
      transformAny(meanSumScore) 

  def binaryCrossEntropy[T: ClassTag](using n: Fractional[T]) = new Loss[T]:
    def calc(a: T, b: T): T = 
      transformAny[Double, T](n.toDouble(a) * math.log(1e-15 + n.toDouble(b))) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =
      val (sumScore, count) = calcMetric(actual, predicted, calc)        
      val meanSumScore = 1.0 / count * transformAny[T, Double](sumScore)
      transformAny(-meanSumScore)        

object LossApi extends LossApi  
  
case class OptimizerContext[T: ClassTag: Fractional](
  learningRate: T,
  clip: GradientClipping[T],
  b1: T,
  b2: T,
  eps: T
)

sealed trait Optimizer[U]:

  def updateWeights[T: ClassTag](
      weights: List[Layer[T]],
      activations: List[Activation[T]],
      error: Tensor[T],
      ctx: OptimizerContext[T]
  )(using n: Fractional[T]): List[Layer[T]]


type Adam
type SimpleGD

object optimizers:
  given Optimizer[Adam] with        
    override def updateWeights[T: ClassTag](
        layers: List[Layer[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        c: OptimizerContext[T]
    )(using n: Fractional[T]): List[Layer[T]] =            
      val iteration = 1 //TOOD use batch id from train methid
      layers
        .zip(activations)
        .foldRight(
          List.empty[Layer[T]],
          error,
          None: Option[Tensor[T]],
          (n.zero.asT, n.zero.asT)
        ) {
          case (
                (Layer(w, b, f, u), Activation(x, z, _)),
                (ls, prevDelta, prevWeight, (m, v))
              ) =>            
            val delta = (prevWeight match 
              case Some(pw) => prevDelta * pw.T
              case None     => prevDelta
            ) multiply f.derivative(z)        
            val wGradient = c.clip(x.T * delta)
            val bGradient = c.clip(delta).sum
            
            // Adam            
            val mt = (c.b1 * m) + (n.one - c.b1) * wGradient
            val vt = (c.b2 * v) + (n.one - c.b2) * wGradient.sqr

            val mHat = mt :/ (n.one - (c.b1 ** iteration))
            val vHat = vt :/ (n.one - (c.b2 ** iteration))            

            val newWeight = w - ((c.learningRate *: mHat) / (vHat.sqrt + c.eps))

            val newBias = b - (c.learningRate * bGradient)
            val updated = Layer(newWeight, newBias, f, u) +: ls
            (updated, delta, Some(w), (mt, vt))
        }
        ._1    

  given Optimizer[SimpleGD] with
    override def updateWeights[T: ClassTag](
        layers: List[Layer[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        ctx: OptimizerContext[T]
    )(using n: Fractional[T]): List[Layer[T]] =      
      layers
        .zip(activations)
        .foldRight(
          List.empty[Layer[T]],
          error,
          None: Option[Tensor[T]]
        ) {
          case (
                (Layer(w, b, f, u), Activation(x, z, _)),
                (ls, prevDelta, prevWeight)
              ) =>            
            val delta = (prevWeight match {
              case Some(pw) => prevDelta * pw.T
              case None     => prevDelta
            }) multiply f.derivative(z)

            val wGradient = ctx.clip(x.T * delta)
            val bGradient = ctx.clip(delta).sum
            val newWeight = w - (ctx.learningRate * wGradient)
            val newBias = b - (ctx.learningRate * bGradient)
            val updated = Layer(newWeight, newBias, f, u) +: ls
            (updated, delta, Some(w))
        }
        ._1    

sealed trait LayerCfg[T]:
  def units: Int
  def f: ActivationFunc[T]

case class Dense[T](
    f: ActivationFunc[T] = ActivationFuncApi.noActivation[T],
    units: Int = 1
) extends LayerCfg[T]

case class Layer[T](
    w: Tensor[T],
    b: Tensor[T],
    f: ActivationFunc[T] = ActivationFuncApi.noActivation[T],
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
  def layers: List[Layer[T]]
  def predict(x: Tensor[T], weightList: List[Layer[T]] = layers): Tensor[T]
  def history: TrainHistory[T]
  def metricValues: List[(Metric[T], List[Double])]

object Model:
  def getAvgLoss[T: ClassTag](losses: List[T])(using num: Fractional[T]): T =
    transformAny[Float, T](num.toFloat(losses.sum) / losses.length)

object Sequential:
  def activate[T: Fractional: ClassTag](
      input: Tensor[T],
      layers: List[Layer[T]]
  ): List[Activation[T]] =
    layers
      .foldLeft(input, ArrayBuffer.empty[Activation[T]]) {
        case ((x, acc), Layer(w, b, f, _)) =>
          val z = x * w + b
          val a = f(z)
          (a, acc :+ Activation(x, z, a))
      }
      ._2
      .toList

case class TrainHistory[T](layers: List[List[Layer[T]]] = Nil, losses: List[T] = Nil)

case class Sequential[T: ClassTag: RandomGen: Fractional, U](
    lossFunc: Loss[T],
    learningRate: T,
    metrics: List[Metric[T]] = Nil,
    batchSize: Int = 16,
    layerStack: Int => List[Layer[T]] = (_: Int) => List.empty[Layer[T]],    
    layers: List[Layer[T]] = Nil,
    history: TrainHistory[T] = TrainHistory[T](),    
    metricValues: List[(Metric[T], List[Double])] = Nil,
    gradientClipping: GradientClipping[T] = GradientClippingApi.noClipping[T]
)(using optimizer: Optimizer[U]) extends Model[T]:
  
  val eps = transformAny[Double, T](10E-8)
  val b1 = transformAny[Double, T](0.9)
  val b2 = transformAny[Double, T](0.999)
  private val ctx = OptimizerContext[T](learningRate, gradientClipping, b1, b2, eps)

  def predict(x: Tensor[T], l: List[Layer[T]] = layers): Tensor[T] =
    activate(x, l).last.a

  def loss(x: Tensor[T], y: Tensor[T], w: List[Layer[T]]): T =
    val predicted = predict(x, w)    
    lossFunc(y, predicted)  

  def add(layer: LayerCfg[T]): Sequential[T, U] =
    copy(layerStack = (inputs) => {
      val currentLayers = layerStack(inputs)
      val prevInput = currentLayers.lastOption.map(_.units).getOrElse(inputs)
      val w = random2D(prevInput, layer.units)
      val b = zeros(layer.units)
      (currentLayers :+ Layer(w, b, layer.f, layer.units))
    })

  private def trainEpoch(
      batches: Array[(Array[Array[T]], Array[Array[T]])],
      layers: List[Layer[T]]
  ) =
    val (w, l, metricValue) =
      batches.foldLeft(layers, List.empty[T], List.fill(metrics.length)(0)) {
        case ((layers, batchLoss, epochMetrics), (xBatch, yBatch)) =>
          // forward
          val activations = activate(xBatch.as2D, layers)
          val actual = yBatch.as2D          
          val predicted = activations.last.a          
          val error = predicted - actual          
          val loss = lossFunc(actual, predicted)

          // backward
          val updated = optimizer.updateWeights(
            layers,
            activations,
            error,
            ctx
          )
          val metricValues = metrics
            .map(_.calculate(actual, predicted))
            .zip(epochMetrics).map(_ + _)
          (updated, batchLoss :+ loss, metricValues)
      }    
    (w, getAvgLoss(l), metricValue)

  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T] =
    lazy val actualBatches = y.batches(batchSize).toArray
    lazy val xBatches = x.batches(batchSize).zip(actualBatches).toArray
    val inputs = x.cols
    val l = getOrInitLayers(inputs)
    val emptyMetrics = metrics.map(_ -> List.empty[Double])

    val (updatedLayers, lHistory, epochLosses, metricValues) =
      (1 to epochs).foldLeft(l, List.empty[List[Layer[T]]], List.empty[T], emptyMetrics) {
        case ((weights, lHistory, losses, metricsList), epoch) =>
          val (l, avgLoss, epochMetric) = trainEpoch(xBatches, weights)
          
          val epochMetricAvg = metrics.zip(epochMetric).map((m, value) => m -> m.average(x.length, value))
          printMetrics(epoch, epochs, avgLoss, epochMetricAvg)          
          val epochMetrics = epochMetricAvg.zip(metricsList).map { 
            case ((epochMetric, v), (trainingMetric, values)) => trainingMetric -> (values :+ v)
          }

          (l, lHistory :+ l, losses :+ avgLoss, epochMetrics)
      }

    copy(
      layers = updatedLayers, 
      history = history.copy(losses = epochLosses, layers = lHistory), 
      metricValues = metricValues
    )

  private def printMetrics(epoch: Int, epochs: Int, avgLoss: T, values: List[(Metric[T], Double)]) = 
    val metricsStat = values
      .map((m, avg) => s"${m.name}: $avg")
      .mkString(", metrics: [", ";", "]")
    println(
      s"epoch: $epoch/$epochs, avg. loss: $avgLoss${if (metrics.nonEmpty) metricsStat else ""}"
    )

  def reset(): Model[T] =
    copy(layers = Nil)

  private def getOrInitLayers(inputs: Int) =
    if layers == Nil then layerStack(inputs)
    else layers

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

trait MetricApi:
  def predictedToBinary[T](v: T)(using n: Fractional[T]): T =
    if n.toDouble(v) > 0.5 then n.one else n.zero

  def accuracyBinaryClassification[T: ClassTag: Fractional] = new Metric[T]:
    val name = "accuracy"

    def calculate(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): Int =      
        val predictedNormalized = predicted.map(predictedToBinary)
        actual.equalRows(predictedNormalized)      

object MetricApi extends MetricApi