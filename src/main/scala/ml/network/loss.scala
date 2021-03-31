package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._
import ml.math.generic._

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
      case (t @ Tensor2D(a), Tensor2D(b)) =>
        val (rows, cols) = t.shape2D
        val sum = (t1, t2).map2(f).sum //TODO: sum and then apply f ?
        (sum, rows * cols)
      case (Tensor0D(a), Tensor0D(b)) =>        
        (f(a, b), 1)
      case _ => 
        sys.error(s"Both tensors must be the same shape: ${t1.shape} != ${t2.shape}")

  private def mean[T: Numeric: ClassTag](count: Int, sum: T): Double =
    castFromTo[T, Double](sum) / count

  def meanSquareError[T: ClassTag](using n: Numeric[T]) = new Loss[T]:
    def calc(a: T, b: T): T =
      pow(a - b, n.fromInt(2))

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =      
      val (sumScore, count) = calcMetric(actual, predicted, calc)      
      val meanSumScore = mean(count, sumScore)
      castFromTo(meanSumScore) 

  def crossEntropy[T: ClassTag: Numeric] = new Loss[T]:
    def calc(y: T, yHat: T): T = 
      y * log(yHat)

    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =
      val (sumScore, count) = calcMetric(actual, predicted, calc)        
      val meanSumScore = mean(count, sumScore)
      castFromTo(-meanSumScore)
  
  def binaryCrossEntropy[T: ClassTag](using n: Numeric[T]) = new Loss[T]:
    def calc(y: T, yHat: T): T =      
      y * log(yHat) + (n.one - y) * log(n.one - yHat)
       
    override def apply(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): T =
      val (sumScore, count) = calcMetric(actual, predicted, calc)        
      val meanSumScore = mean(count, sumScore)
      castFromTo(-meanSumScore)