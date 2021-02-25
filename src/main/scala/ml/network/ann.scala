package ml.network 

import ml.network.RandomGen._
import ml.transformation.transformAny
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
  def predict(x: Tensor[T], weightList: List[Layer[T]] = layers): Tensor[T]
  def history: TrainHistory[T]
  def metricValues: List[(Metric[T], List[Double])]

object Model:
  def getAvgLoss[T: ClassTag](losses: List[T])(using n: Fractional[T]): T =
    transformAny[Double, T](n.toDouble(losses.sum) / losses.length)

object Sequential:
  def activate[T: Fractional: ClassTag](
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

case class Sequential[T: ClassTag: RandomGen: Fractional, U](
    lossFunc: Loss[T],    
    learningRate: T,
    metrics: List[Metric[T]] = Nil,
    batchSize: Int = 16,
    layerStack: Int => List[Layer[T]] = (_: Int) => List.empty[Layer[T]],    
    layers: List[Layer[T]] = Nil,
    history: TrainHistory[T] = TrainHistory[T](),    
    metricValues: List[(Metric[T], List[Double])] = Nil,
    gradientClipping: GradientClipping[T] = GradientClippingApi.noClipping[T],
    cfg: Option[OptimizerCfg[T]] = None
)(using optimizer: Optimizer[U]) extends Model[T]:

  private val optimizerCfg = 
    cfg.getOrElse(OptimizerCfg(learningRate = learningRate, gradientClipping, AdamCfg.default))

  def withCfg(cfg: OptimizerCfg[T]) =
    copy(cfg = Some(cfg))

  def predict(x: Tensor[T], l: List[Layer[T]] = layers): Tensor[T] =
    activate(x, l).last.a

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
          val metricValues = metrics
            .map(_.calculate(actual, predicted))
            .zip(epochMetrics).map(_ + _)
          (updated, batchLoss :+ loss, metricValues)
      }    
    (trained, getAvgLoss(losses), metricValue)

  def train(x: Tensor[T], y: Tensor[T], epochs: Int): Model[T] =
    lazy val actualBatches = y.batches(batchSize).toArray
    lazy val xBatches = x.batches(batchSize).zip(actualBatches).toArray
    val inputs = x.cols
    val l = getOrInitLayers(inputs)
    val emptyMetrics = metrics.map(_ -> List.empty[Double])

    val (updatedLayers, lHistory, epochLosses, metricValues) =
      (1 to epochs).foldLeft(l, List.empty[List[Layer[T]]], List.empty[T], emptyMetrics) {
        case ((weights, lHistory, losses, metricsList), epoch) =>
          val (l, avgLoss, epochMetric) = trainEpoch(xBatches, weights, epoch)
          
          val epochMetricAvg = metrics.zip(epochMetric).map((m, value) => m -> m.average(x.length, value))
          printMetrics(epoch, epochs, avgLoss, epochMetricAvg)          
          val epochMetrics = epochMetricAvg.zip(metricsList).map { 
            case ((_, v), (trainingMetric, values)) => trainingMetric -> (values :+ v)
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
      s"epoch: $epoch/$epochs, avg. loss: $avgLoss${if metrics.nonEmpty then metricsStat else ""}"
    )

  def reset(): Model[T] =
    copy(layers = Nil)

  private def getOrInitLayers(inputs: Int) =
    if layers == Nil then layerStack(inputs)
    else layers