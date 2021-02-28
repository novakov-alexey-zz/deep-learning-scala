package ml.tensors.api

import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag
import math.Ordering.Implicits.infixOrderingOps

sealed trait Tensor[T]:  
  def length: Int
  def shape: List[Int]
  def cols: Int  

object Tensor:    
  def of[T:ClassTag](size: Int, size2: Int): Tensor2D[T] = 
    Tensor2D[T](Array.fill(size)(of[T](size2).data))

  def of[T: ClassTag](size: Int): Tensor1D[T] = 
    Tensor1D[T](Array.ofDim[T](size))

case class Tensor0D[T: ClassTag](data: T) extends Tensor[T]:
  override val length: Int = 1
  
  override val shape: List[Int] = length :: Nil
  
  private val meta = s"shape: $length, Tensor0D[${summon[ClassTag[T]]}]"

  override def toString: String =
    s"$meta:\n" + data + "\n"

  override val cols: Int = length

case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T]:
  override def shape: List[Int] = List(data.length)

  override def toString: String =
    val meta = s"shape: ${shape.head}, Tensor1D[${summon[ClassTag[T]]}]"
    s"$meta:\n[" + data.mkString(",") + "]\n"

  override def length: Int = data.length

  override def cols: Int = length

object Tensor1D:
  def apply[T: ClassTag](data: T*): Tensor1D[T] = 
    Tensor1D[T](data.toArray)

case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T]:
  override def shape: List[Int] =
    val (r, c) = shape2D
    List(r, c)

  def shape2D: (Int, Int) =
    (data.length, data.headOption.map(_.length).getOrElse(0))

  private val meta =
    s"shape: ${shape.mkString("x")}, Tensor2D[${summon[ClassTag[T]]}]"

  override def toString: String =
    s"$meta:\n[" + data
      .map(_.mkString("[", ",", "]"))
      .mkString("\n ") + "]\n"

  override def cols: Int = shape2D._2

  override def length: Int = data.length

object Tensor2D:
  def apply[T: ClassTag](rows: Array[T]*): Tensor2D[T] =
    Tensor2D[T](rows.toArray)