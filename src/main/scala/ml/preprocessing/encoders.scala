package ml.preprocessing

import Encoder._
import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops.T

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object Encoder:
  def toClasses[T: ClassTag: Ordering, U: ClassTag](
      samples: Tensor1D[T]
  ): Map[T, U] =
    samples.data.distinct.sorted.zipWithIndex.toMap.view
      .mapValues(castFromTo[Int, U])
      .toMap
      
case class LabelEncoder[T: ClassTag: Ordering](
    classes: Map[T, T] = Map.empty[T, T]
):
  def fit(samples: Tensor1D[T]): LabelEncoder[T] =
    LabelEncoder(toClasses[T, T](samples))

  def transform(t: Tensor2D[T], col: Int): Tensor2D[T] =
    val data = t.data.map(
      _.zipWithIndex.map { (d, i) =>
        if i == col then classes.getOrElse(d, d) else d
      }
    )
    Tensor2D(data)

/**
 * T - key type
 * U - numeric value type for the key type
 */
case class OneHotEncoder[
    T: Ordering: ClassTag,
    U: Ordering: ClassTag
](
    classes: Map[T, U] = Map.empty[T, U],
    notFound: Int = -1
)(using n: Numeric[U]):
  def fit(samples: Tensor1D[T]): OneHotEncoder[T, U] =
    OneHotEncoder[T, U](toClasses[T, U](samples))

  def transform(t: Tensor1D[T]): Tensor2D[T] =
    Tensor2D(t.data.map(encode))

  private def encode(v: T) = 
    val zero = castFromTo[Int, T](0)
    val array = Array.fill[T](classes.size)(zero)
    val pos = classes.get(v)
    pos match
      case Some(p) =>
        array(n.toInt(p)) = castFromTo[U, T](n.one)
      case None =>
        array(0) = castFromTo[U, T](n.fromInt(notFound))
    array

  def transform(t: Tensor2D[T], col: Int): Tensor2D[T] =    
    val data = t.data.map { row =>
      row.zipWithIndex
        .foldLeft(ArrayBuffer.empty[T]) { case (acc, (v, i)) =>
          if i == col then acc ++ encode(v)
          else acc :+ v
        }
        .toArray[T]
    }
    Tensor2D(data)

case class ColumnStat(mean: Double, stdDev: Double)

case class StandardScaler[T: Numeric: ClassTag](
    stats: Array[ColumnStat] = Array.empty
):
  def fit(samples: Tensor[T]): StandardScaler[T] =
    samples match
      case Tensor1D(data) =>
        StandardScaler(Array(fitColumn(data)))
      case t @ Tensor2D(_) =>
        StandardScaler(t.T.data.map(fitColumn))
      case Tensor0D(_) => StandardScaler()
      case _ => 
        sys.error(s"Not implemented for: $samples")

  private def fitColumn(data: Array[T]) =
    val nums = data.map(castFromTo[T, Double])
    val mean = nums.sum / data.length
    val stdDev = math.sqrt(
      nums.map(n => math.pow(n - mean, 2)).sum / (data.length - 1)
    )
    ColumnStat(mean, stdDev)

  def transform(t: Tensor[T]): Tensor[T] =
    t match
      case Tensor1D(data) =>
        val stat = stats.headOption.getOrElse(
          sys.error(s"There is no statistics for $t")
        )
        val res = data.map(n =>
          castFromTo[Double, T](scale(castFromTo[T, Double](n), stat))
        )
        Tensor1D(res)
      case t2 @ Tensor2D(data) =>
        val (rows, cols) = t2.shape2D
        val res = Array.ofDim[T](rows, cols)

        for i <- 0 until rows do
          for j <- 0 until cols do
            val stat = stats(j)
            val n = castFromTo[T, Double](data(i)(j))
            res(i)(j) = castFromTo[Double, T](scale(n, stat))
        Tensor2D(res)
      case Tensor0D(_) => t // scaling is not applicable for scalar tensor
      case _ => sys.error(s"Not implemented for: $t")

  private def scale(n: Double, stat: ColumnStat): Double =
    (n - stat.mean) / stat.stdDev
