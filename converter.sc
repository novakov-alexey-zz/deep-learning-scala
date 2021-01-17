// scala 2.13.3

import $file.tensor

import tensor._

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

def transform[T: ClassTag: TypeTag](data: Array[Array[String]]): Tensor2D[T] = {
  val transformed = data.map(a => transformArr[T](a))
  Tensor2D[T](transformed)
}

def transformArr[T: TypeTag: ClassTag](data: Array[String]): Array[T] =
  typeOf[T] match {
    case t if t =:= typeOf[Float]  => data.map(_.toFloat.asInstanceOf[T])
    case t if t =:= typeOf[String] => data.map(_.asInstanceOf[T])
    case t if t =:= typeOf[Double] => data.map(_.toDouble.asInstanceOf[T])
  }
