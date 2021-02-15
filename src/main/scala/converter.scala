import ImplicitConfigHelper._

import scala.reflect.ClassTag

object ImplicitConfigHelper:
  val String_ = classOf[String]
  val Int_ = classOf[Int]
  val Long_ = classOf[Long]
  val Float_ = classOf[Float]
  val Double_ = classOf[Double]

object converter:

  def transform[T: ClassTag](
      data: Array[Array[String]]
  ): Tensor2D[T] =
    val transformed = data.map(a => transformArr[T](a))
    Tensor2D[T](transformed)

  def transformArr[T: ClassTag](data: Array[String]): Array[T] =
    summon[ClassTag[T]].runtimeClass match
      case Float_  => data.map(_.toFloat.asInstanceOf[T])
      case String_ => data.map(_.asInstanceOf[T])
      case Double_ => data.map(_.toDouble.asInstanceOf[T])

  private def transformInt[T: ClassTag](data: Int): T =
    summon[ClassTag[T]].runtimeClass match
      case Float_  => data.toFloat.asInstanceOf[T]
      case String_ => data.toString.asInstanceOf[T]
      case Double_ => data.toDouble.asInstanceOf[T]
      case Int_    => data.asInstanceOf[T]

  def transformAny[A, B](a: A)(implicit ev1: ClassTag[A], ev2: ClassTag[B]): B =
    (ev1.runtimeClass, ev2.runtimeClass) match
      case (Float_, String_)  => a.toString.asInstanceOf[B]
      case (Double_, String_) => a.toString.asInstanceOf[B]
      case (String_, Float_)  => a.toString.toFloat.asInstanceOf[B]
      case (String_, Double_) => a.toString.toDouble.asInstanceOf[B]
      case (Float_, Double_)  => a.asInstanceOf[Float].toDouble.asInstanceOf[B]
      case (Double_, Float_)  => a.asInstanceOf[Double].toFloat.asInstanceOf[B]
      case (Double_, Double_)  => a.asInstanceOf[B]
      case (Float_, Float_)   => a.asInstanceOf[B]
      case (Int_, _)          => transformInt[B](a.asInstanceOf[Int])