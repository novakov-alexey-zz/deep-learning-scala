package ml.network 

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._

import Model._
import Sequential._

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag
import scala.util.Random

/*
 * z - before activation = w * x
 * a - activation value
 */
case class Activation[T](x: Tensor[T], z: Tensor[T], a: Tensor[T])

sealed trait Model[T]:
  def reset(): Model[T]
  def train(x: Tensor[T], y: Tensor[T], epochs: Int, shuffle: Boolean = true): Model[T]
  def layers: List[Layer[T]]
  def predict(x: Tensor[T], customLayers: List[Layer[T]] = layers): Tensor[T]
  def apply(x: Tensor[T], customLayers: List[Layer[T]] = layers): Tensor[T] = 
    predict(x, customLayers)
  def history: TrainHistory[T]
  def metricValues: MetricValues[T]

object Model:
  def getAvgLoss[T: ClassTag](losses: List[T])(using n: Fractional[T]): T =
    castFromTo[Double, T](n.toDouble(losses.sum) / losses.length)

object Sequential:
  def activate[T: Numeric: ClassTag](
      input: Tensor[T],
      layers: List[Layer[T]]
  ): List[Activation[T]] =
    layers
      .foldLeft(input, ListBuffer.empty[Activation[T]]) {
        case ((x, acc), layer) =>
          val act = layer(x)          
          (act.a, acc :+ act)
      }
      ._2
      .toList

case class TrainHistory[T](layers: List[List[Layer[T]]] = Nil, losses: List[T] = Nil)

type MetricValues[T] = List[(Metric[T], List[Double])]

case class Sequential[T: ClassTag: Fractional, U, V](
    lossFunc: Loss[T],    
    learningRate: T,
    metrics: List[Metric[T]] = Nil,
    batchSize: Int = 16,
    layerStack: List[Int] => List[Layer[T]] = _ => List.empty[Layer[T]],    
    layers: List[Layer[T]] = Nil,
    history: TrainHistory[T] = TrainHistory[T](),    
    metricValues: MetricValues[T] = Nil,
    gradientClipping: GradientClipping[T] = GradientClippingApi.noClipping[T],
    cfg: Option[OptimizerCfg[T]] = None
)(using optimizer: Optimizer[U], initializer: ParamsInitializer[T, V]) extends Model[T]:

  private val optimizerCfg = 
    cfg.getOrElse(OptimizerCfg(learningRate = learningRate, gradientClipping, AdamCfg.default))

  def withCfg(cfg: OptimizerCfg[T]) =
    copy(cfg = Some(cfg))

  def predict(x: Tensor[T], inputLayers: List[Layer[T]] = layers): Tensor[T] =
    activate(x, inputLayers).last.a

  def loss(x: Tensor[T], y: Tensor[T], w: List[Layer[T]]): T =
    val predicted = predict(x, w)    
    lossFunc(y, predicted)  

  def add(layer: Layer[T]): Sequential[T, U, V] =
    copy(layerStack = inputShape => 
      val currentLayers = layerStack(inputShape)
      val prevShape = currentLayers.lastOption.map(_.shape).getOrElse(inputShape)
      val initialized = layer match
        case o: Optimizable[_] => o.init(prevShape, initializer, optimizer)
        case _ => layer.init(prevShape)
      (currentLayers :+ initialized)
    )

  private def trainEpoch(
      batches: Array[(Tensor[T], Tensor[T])],
      layers: List[Layer[T]],
      epoch: Int
  ) =    
    val (trained, losses, metricValue, _) =
      batches.zipWithIndex.foldLeft(
        layers, 
        ListBuffer.empty[T], 
        ListBuffer.fill(metrics.length)(0),
        0L) {
        case ((layers, batchLoss, epochMetrics, stepDuration), ((x, y), i)) =>
          // forward
          val start = System.currentTimeMillis()
          val activations = activate(x, layers)             
          val predicted = activations.last.a
          val error = predicted - y          
          val loss = lossFunc(y, predicted)

          // backward          
          val updated = optimizer.updateWeights(
            layers,
            activations,
            error,
            optimizerCfg,
            (i + 1) * epoch
          )

          // update metrics
          val matches = metrics
            .map(_.matches(y, predicted))
            .zip(epochMetrics).map(_ + _)
          val duration = stepDuration + (System.currentTimeMillis() - start)
          printEpochPerformance(i + 1, duration)

          (updated, batchLoss :+ loss, matches.to(ListBuffer), duration)
      }    
    (trained, getAvgLoss(losses.toList), metricValue)
  
  inline private def printEpochPerformance(step: Int, duration: Long) = 
    if step % 50 == 0 then
      println(s"${step.toDouble / (duration / 1000d)} steps/sec")

  def train(x: Tensor[T], y: Tensor[T], epochs: Int, shuffle: Boolean = true): Model[T] =
    lazy val actualBatches = y.batches(batchSize).toArray
    lazy val batches = x.batches(batchSize).zip(actualBatches).toArray
    def getBatches = if shuffle then Random.shuffle(batches).toArray else batches
    val currentLayers = getOrInitLayers(x.shape)
    val initialMetrics = metrics.map(_ -> List.empty[Double])
    println(s"Running $epochs epochs")

    val (updatedLayers, lHistory, epochLosses, metricValues) =
      (1 to epochs).foldLeft(currentLayers, ListBuffer.empty[List[Layer[T]]], ListBuffer.empty[T], initialMetrics) {
        case ((layers, lHistory, losses, trainingMetrics), epoch) =>
          val start = System.currentTimeMillis()
          val (trainedLayers, avgLoss, epochMatches) = trainEpoch(getBatches, layers, epoch)
          val duration = System.currentTimeMillis() - start
          
          val (epochMetrics, epochMetricAvg) = updateMetrics(epochMatches.toList, trainingMetrics, x.length)
          printMetrics(epoch, epochs, avgLoss, epochMetricAvg, duration)          

          (trainedLayers, lHistory :+ trainedLayers, losses :+ avgLoss, epochMetrics)
      }

    copy(
      layers = updatedLayers, 
      history = history.copy(losses = epochLosses.toList, layers = lHistory.toList), 
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

  private def printMetrics(epoch: Int, epochs: Int, avgLoss: T, values: List[(Metric[T], Double)], duration: Long) = 
    val metricsStat = values
      .map((m, avg) => s"${m.name}: $avg")
      .mkString(", metrics: [", ";", "]")
    println(
      s"epoch: $epoch/$epochs, duration: ${duration/1000} sec, avg. loss: $avgLoss${if metrics.nonEmpty then metricsStat else ""}"
    )

  def reset(): Model[T] =
    copy(layers = Nil)

  private def getOrInitLayers(inputShape: List[Int]) =
    if layers.isEmpty then layerStack(inputShape)
    else layers