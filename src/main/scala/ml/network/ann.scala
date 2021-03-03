package ml.network 

import ml.network.RandomGen._
import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._

import Model._
import Sequential._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

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
    units: Int = 1,
    state: Option[OptimizerState[T]] = None
): 
  override def toString() = 
    s"\n(\nweight = $w,\nbias = $b,\nf = ${f.name},\nunits = $units)"

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Activation[T](x: Tensor[T], z: Tensor[T], a: Tensor[T])

sealed trait Model[T]:
  def reset(): Model[T]
  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T]
  def layers: List[Layer[T]]
  def predict(x: Tensor[T], customLayers: List[Layer[T]] = layers): Tensor[T]
  def apply(x: Tensor[T], customLayers: List[Layer[T]] = layers): Tensor[T] = 
    predict(x, customLayers)
  def history: TrainHistory[T]
  def metricValues: List[(Metric[T], List[Double])]

object Model:
  def getAvgLoss[T: ClassTag](losses: List[T])(using n: Fractional[T]): T =
    castFromTo[Double, T](n.toDouble(losses.sum) / losses.length)

object Sequential:
  def activate[T: Numeric: ClassTag](
      input: Tensor[T],
      layers: List[Layer[T]]
  ): List[Activation[T]] =
    layers
      .foldLeft(input, ArrayBuffer.empty[Activation[T]]) {
        case ((x, acc), Layer(w, b, f, _, _)) =>
          val z = x * w + b
          val a = f(z)
          (a, acc :+ Activation(x, z, a))
      }
      ._2
      .toList

case class TrainHistory[T](layers: List[List[Layer[T]]] = Nil, losses: List[T] = Nil)

type MetricValues[T] = List[(Metric[T], List[Double])]

case class Sequential[T: ClassTag: RandomGen: Fractional, U](
    lossFunc: Loss[T],    
    learningRate: T,
    metrics: List[Metric[T]] = Nil,
    batchSize: Int = 16,
    layerStack: Int => List[Layer[T]] = _ => List.empty[Layer[T]],    
    layers: List[Layer[T]] = Nil,
    history: TrainHistory[T] = TrainHistory[T](),    
    metricValues: MetricValues[T] = Nil,
    gradientClipping: GradientClipping[T] = GradientClippingApi.noClipping[T],
    cfg: Option[OptimizerCfg[T]] = None
)(using optimizer: Optimizer[U]) extends Model[T]:

  private val optimizerCfg = 
    cfg.getOrElse(OptimizerCfg(learningRate = learningRate, gradientClipping, AdamCfg.default))

  def withCfg(cfg: OptimizerCfg[T]) =
    copy(cfg = Some(cfg))

  def predict(x: Tensor[T], inputLayers: List[Layer[T]] = layers): Tensor[T] =
    activate(x, inputLayers).last.a

  def loss(x: Tensor[T], y: Tensor[T], w: List[Layer[T]]): T =
    val predicted = predict(x, w)    
    lossFunc(y, predicted)  

  def add(layer: LayerCfg[T]): Sequential[T, U] =
    copy(layerStack = (inputs) => 
      val currentLayers = layerStack(inputs)
      val prevInput = currentLayers.lastOption.map(_.units).getOrElse(inputs)
      val w = random2D(prevInput, layer.units)
      val b = zeros(layer.units)
      val optimizerState = optimizer.initState(w, b)
      (currentLayers :+ Layer(w, b, layer.f, layer.units, optimizerState))
    )

  private def trainEpoch(
      batches: Array[(Array[Array[T]], Array[Array[T]])],
      layers: List[Layer[T]],
      epoch: Int
  ) =
    val index = (1 to batches.length)
    val (trained, losses, metricValue) =
      batches.zip(index).foldLeft(layers, List.empty[T], List.fill(metrics.length)(0)) {
        case ((layers, batchLoss, epochMetrics), ((xBatch, yBatch), i)) =>
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
            optimizerCfg,
            i * epoch
          )
          val matches = metrics
            .map(_.matches(actual, predicted))
            .zip(epochMetrics).map(_ + _)
          (updated, batchLoss :+ loss, matches)
      }    
    (trained, getAvgLoss(losses), metricValue)

  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T] =
    lazy val actualBatches = y.batches(batchSize).toArray
    lazy val batches = x.batches(batchSize).zip(actualBatches).toArray
    val inputs = x.cols
    val currentLayers = getOrInitLayers(inputs)
    val initialMetrics = metrics.map(_ -> List.empty[Double])

    val (updatedLayers, lHistory, epochLosses, metricValues) =
      (1 to epochs).foldLeft(currentLayers, List.empty[List[Layer[T]]], List.empty[T], initialMetrics) {
        case ((layers, lHistory, losses, trainingMetrics), epoch) =>
          val (trainedLayers, avgLoss, epochMatches) = trainEpoch(batches, layers, epoch)
          
          val (epochMetrics, epochMetricAvg) = updateMetrics(epochMatches, trainingMetrics, x.length)
          printMetrics(epoch, epochs, avgLoss, epochMetricAvg)          

          (trainedLayers, lHistory :+ trainedLayers, losses :+ avgLoss, epochMetrics)
      }

    copy(
      layers = updatedLayers, 
      history = history.copy(losses = epochLosses, layers = lHistory), 
      metricValues = metricValues
    )

  private def updateMetrics(    
    observedMatches: List[Int],     
    currentMetrics: MetricValues[T],
    samples: Int
  ) =
    val observedAvg = metrics.zip(observedMatches).map((m, matches) => m -> m.average(samples, matches))    
    val updatedMetrics = observedAvg.zip(currentMetrics).map { 
      case ((_, v), (currentMetric, values)) => currentMetric -> (values :+ v)
    }
    (updatedMetrics, observedAvg)

  private def printMetrics(epoch: Int, epochs: Int, avgLoss: T, values: List[(Metric[T], Double)]) = 
    val metricsStat = values
      .map((m, avg) => s"${m.name}: $avg")
      .mkString(", metrics: [", ";", "]")
    println(
      s"epoch: $epoch/$epochs, avg. loss: $avgLoss${if metrics.nonEmpty then metricsStat else ""}"
    )

  def reset(): Model[T] =
    copy(layers = Nil)

  private def getOrInitLayers(inputs: Int) =
    if layers.isEmpty then layerStack(inputs)
    else layers