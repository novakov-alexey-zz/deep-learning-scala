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
        a.toString().asInstanceOf[B]
      case (t1, t2) if t1 =:= typeOf[String] && t2 =:= typeOf[Float] =>
        a.toString.toFloat.asInstanceOf[B]
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
