import scala.reflect.ClassTag
import scala.reflect.runtime.universe.{TypeTag, typeOf}

object converter {

  def transform[T: ClassTag: TypeTag](
      data: Array[Array[String]]
  ): Tensor2D[T] = {
    val transformed = data.map(a => transformArr[T](a))
    Tensor2D[T](transformed)
  }

  def transformArr[T: TypeTag: ClassTag](data: Array[String]): Array[T] =
    typeOf[T] match {
      case t if t =:= typeOf[Float]  => data.map(_.toFloat.asInstanceOf[T])
      case t if t =:= typeOf[String] => data.map(_.asInstanceOf[T])
      case t if t =:= typeOf[Double] => data.map(_.toDouble.asInstanceOf[T])
    }

  private def transformInt[T: TypeTag](data: Int): T =
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
      case (t1, t2) if t1 =:= typeOf[Float] && t2 =:= typeOf[Float] =>
        a.asInstanceOf[B]
      case (t1, _) if t1 =:= typeOf[Int] =>
        transformInt[B](a.asInstanceOf[Int])
    }
}
