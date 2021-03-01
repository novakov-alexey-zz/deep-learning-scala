package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._

import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

trait Loss[T]:
  def apply(
      actual: Tensor[T],
      predicted: Tensor[T]
  ): T

object LossApi:
  private def calcMetric[T: Numeric: ClassTag](
    t1: Tensor[T], t2: Tensor[T], f: (T, T) => T
  ) = 
    (t1, t2) match
      case (Tensor1D(a), Tensor1D(b)) =>         
        val sum = (t1, t2).map2(f).sum //TODO: sum and then apply f ?
        (sum, t1.length)
      case (Tensor2D(a), Tensor2D(b)) =>
        val size = t1.length * t1.cols        
        val sum = (t1, t2).map2(f).sum //TODO: sum and then apply f ?
        (sum, size)
      case (Tensor0D(a), Tensor0D(b)) =>        
        (f(a, b), 1)
      case _ => 
        sys.error(s"Both tensors must be the same shape: ${t1.shape} != ${t2.shape}")

  private def mean[T: Numeric: ClassTag](count: Int, sum: T): Double =
    1.0 / count * castFromTo[T, Double](sum)

  def meanSquareError[T: ClassTag: Numeric] = new Loss[T]:
    def calc(a: T, b: T): T =      
      castFromTo[Double, T](math.pow(castFromTo[T, Double](a - b), 2)) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =      
      val (sumScore, count) = calcMetric(actual, predicted, calc)      
      val meanSumScore = mean(count, sumScore)
      castFromTo(meanSumScore) 

  def crossEntropy[T: ClassTag](using n: Numeric[T]) = new Loss[T]:
    def calc(y: T, yHat: T): T = 
      castFromTo[Double, T](n.toDouble(y) * math.log(1e-15 + n.toDouble(yHat))) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =
      val (sumScore, count) = calcMetric(actual, predicted, calc)        
      val meanSumScore = mean(count, sumScore)
      castFromTo(-meanSumScore)
  
  def binaryCrossEntropy[T: ClassTag](using n: Numeric[T]) = new Loss[T]:
    def calc(y: T, yHat: T): T = 
      castFromTo[Double, T](
        n.toDouble(y) * math.log(1e-15 + n.toDouble(yHat)) + (1 - n.toDouble(y)) * math.log(1 - n.toDouble(yHat))
      ) 

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =
      val (sumScore, count) = calcMetric(actual, predicted, calc)        
      val meanSumScore = mean(count, sumScore)
      castFromTo(-meanSumScore)