package examples

import ml.network.api._
import ml.tensors.api._
import ml.tensors.ops._
import ml.preprocessing._

import scala.reflect.ClassTag

object mnistCommon:
  def accuracyMnist[T: ClassTag: Ordering](using n: Numeric[T]) = new Metric[T]:
    val name = "accuracy"

    def matches(actual: Tensor[T], predicted: Tensor[T]): Int =      
        val predictedArgMax = predicted.argMax      
        actual.argMax.equalRows(predictedArgMax)
      
  val accuracy = accuracyMnist[Double]

  val encoder = OneHotEncoder(
    classes = (0 to 9).map(i => (i.toDouble, i.toDouble)).toMap)  

  def prepareData(x: Tensor[Double], y: Tensor[Double]) =
    val xData = x.map(_ / 255d) // normalize to [0,1] range
    val yData = encoder.transform(y.as1D)
    (xData, yData)