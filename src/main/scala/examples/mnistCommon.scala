package examples

import ml.network.api._
import ml.tensors.api._
import ml.tensors.ops._
import ml.preprocessing._

import scala.reflect.ClassTag

object mnistCommon:
  val imageDir = "images"
  
  def accuracyMnist[T: ClassTag: Ordering](using n: Numeric[T]) = new Metric[T]:
    val name = "accuracy"

    def matches(actual: Tensor[T], predicted: Tensor[T]): Int =      
        val predictedArgMax = predicted.argMax      
        actual.argMax.equalRows(predictedArgMax)     

  def prepareData[T: ClassTag](x: Tensor[T], y: Tensor[T])(using n: Fractional[T]) =
    val encoder = OneHotEncoder(
      classes = (0 to 9).map(i => (n.fromInt(i), n.fromInt(i))).toMap
    )  
    val max = n.fromInt(255)
    val xData = x.map(v =>  n.div(v, max)) // normalize to [0,1] range
    val yData = encoder.transform(y.as1D)
    (xData, yData)