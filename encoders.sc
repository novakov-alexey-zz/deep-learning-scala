// scala 2.13.3

import $file.tensor
import tensor._

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import Encoder._
import scala.collection.mutable.ArrayBuffer

object Encoder {
  def transformInt[T: TypeTag](data: Int): T =
    typeOf[T] match {
      case t if t =:= typeOf[Float]  => data.toFloat.asInstanceOf[T]
      case t if t =:= typeOf[String] => data.toString.asInstanceOf[T]
      case t if t =:= typeOf[Double] => data.toDouble.asInstanceOf[T]
      case t if t =:= typeOf[Int]    => data.asInstanceOf[T]
    }

  def transformAny[A: TypeTag, B: TypeTag](a: A): B =
    (typeOf[A], typeOf[B]) match {
      case (t1, t2) if t1 =:= typeOf[Float] && t2 =:= typeOf[String] =>
        a.toString.asInstanceOf[B]
      case (t1, t2) if t1 =:= typeOf[Double] && t2 =:= typeOf[String] =>
        a.toString.asInstanceOf[B]
      case (t1, t2) if t1 =:= typeOf[String] && t2 =:= typeOf[Float] =>
        a.toString.toFloat.asInstanceOf[B]
      case (t1, t2) if t1 =:= typeOf[String] && t2 =:= typeOf[Double] =>
        a.toString.toDouble.asInstanceOf[B]
      case (t1, t2) if t1 =:= typeOf[Float] && t2 =:= typeOf[Double] =>
        a.asInstanceOf[Float].toDouble.asInstanceOf[B]
      case (t1, t2) if t1 =:= typeOf[Double] && t2 =:= typeOf[Float] =>
        a.asInstanceOf[Double].toFloat.asInstanceOf[B]
    }

  def toClasses[T: TypeTag: Ordering, U: TypeTag](
      samples: Tensor1D[T]
  ): Map[T, U] =
    samples.data.distinct.sorted.zipWithIndex.toMap.view
      .mapValues(transformInt[U])
      .toMap[T, U]
}
case class LabelEncoder[T: ClassTag: Ordering: TypeTag](
    classes: Map[T, T] = Map.empty[T, T]
) {
  def fit(samples: Tensor1D[T]): LabelEncoder[T] =
    LabelEncoder(toClasses[T, T](samples))

  def transform(t: Tensor2D[T], col: Int): Tensor2D[T] = {
    val data = t.data.map {
      _.zipWithIndex.map { case (d, i) =>
        if (i == col) classes.getOrElse(d, d) else d
      }
    }
    Tensor2D(data)
  }
}

case class OneHotEncoder[
    T: Ordering: TypeTag: ClassTag,
    U: TypeTag: Numeric: Ordering
](
    classes: Map[T, U] = Map.empty[T, U],
    noFound: Int = -1
) {
  def fit(samples: Tensor1D[T]) =
    OneHotEncoder[T, U](toClasses[T, U](samples))

  def transform(t: Tensor2D[T], col: Int): Tensor2D[T] = {
    lazy val numeric = implicitly[Numeric[U]]
    val data = t.data.map { row =>
      row.zipWithIndex
        .foldLeft(List.empty[T]) { case (acc, (d, i)) =>
          if (i == col) {
            val pos = classes.get(d)
            val array = Array.fill[T](classes.size)(transformInt[T](0))
            pos match {
              case Some(p) =>
                array(numeric.toInt(p)) = transformAny[U, T](numeric.one)
              case None =>
                array(0) = transformAny[U, T](numeric.fromInt(noFound))
            }
            acc ++ array
          } else acc :+ d
        }
        .toArray[T]
    }
    Tensor2D(data)
  }
}

case class Stats(mean: Double, stdDev: Double)
case class StandardScaler[T: Numeric: TypeTag: ClassTag](
    stats: Array[Stats] = Array.empty
) {

  def fit(samples: Tensor[T]): StandardScaler[T] = {
    samples match {
      case Tensor1D(data) =>
        StandardScaler(Array(fitColumn(data)))
      case t @ Tensor2D(data) =>
        StandardScaler(t.T.asInstanceOf[Tensor2D[T]].data.map(fitColumn))
      case t @ Tensor1D(d) => StandardScaler()
    }
  }

  private def fitColumn(data: Array[T]) = {
    val nums = data.map(transformAny[T, Double])
    val mean = nums.sum / data.length
    val stdDev = math.sqrt(
      nums.map(n => math.pow(n - mean, 2)).sum / (data.length - 1)
    )
    Stats(mean, stdDev)
  }

  def transform(t: Tensor[T]): Tensor[T] = {
    t match {
      case Tensor1D(data) =>
        val stat = stats(0)
        val res = data.map(n =>
          transformAny[Double, T](scale(transformAny[T, Double](n), stat))
        )
        Tensor1D(res)
      case t2 @ Tensor2D(data) =>
        val (rows, cols) = t2.sizes2D
        val res = Array.ofDim[T](rows, cols)

        for (i <- (0 until rows)) {
          for (j <- (0 until cols)) {
            val stat = stats(j)
            val n = transformAny[T, Double](data(i)(j))
            res(i)(j) = transformAny[Double, T](scale(n, stat))
          }
        }

        Tensor2D(res)
      case Tensor1D(_) => t
    }
  }

  private def scale(n: Double, stat: Stats): Double =
    (n - stat.mean) / stat.stdDev
}
