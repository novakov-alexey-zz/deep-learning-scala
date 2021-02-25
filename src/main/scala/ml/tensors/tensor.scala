package ml.tensors.api

import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag
import math.Ordering.Implicits.infixOrderingOps

sealed trait Tensor[T]:
  type A
  def data: A
  def length: Int
  def sizes: List[Int]
  def cols: Int

case class Tensor0D[T: ClassTag](data: T) extends Tensor[T]:
  type A = T
  
  override val length: Int = 1
  
  override val sizes: List[Int] = length :: Nil
  
  private val meta = s"sizes: $length, Tensor0D[${summon[ClassTag[T]]}]"

  override def toString: String =
    s"$meta:\n" + data + "\n"

  override val cols: Int = length

case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T]:
  type A = Array[T]

  override def sizes: List[Int] = List(data.length)

  override def toString: String =
    val meta = s"sizes: ${sizes.head}, Tensor1D[${summon[ClassTag[T]]}]"
    s"$meta:\n[" + data.mkString(",") + "]\n"

  override def length: Int = data.length

  override def cols: Int = length

object Tensor1D:
  def apply[T: ClassTag](data: T*): Tensor1D[T] = 
    Tensor1D[T](data.toArray)

case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T]:
  type A = Array[Array[T]]

  override def sizes: List[Int] =
    val (r, c) = sizes2D
    List(r, c)

  def sizes2D: (Int, Int) =
    (data.length, data.headOption.map(_.length).getOrElse(0))

  private val meta =
    s"sizes: ${sizes.mkString("x")}, Tensor2D[${summon[ClassTag[T]]}]"

  override def toString: String =
    s"$meta:\n[" + data
      .map(_.mkString("[", ",", "]"))
      .mkString("\n ") + "]\n"

  override def cols: Int = sizes2D._2

  override def length: Int = data.length

object Tensor2D:
  def apply[T: ClassTag](rows: Array[T]*): Tensor2D[T] =
    Tensor2D[T](rows.toArray)