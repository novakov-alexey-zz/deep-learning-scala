package ml.tensors.api

import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag
import math.Ordering.Implicits.infixOrderingOps

sealed trait Tensor[T]:  
  def shape: List[Int]
  def length: Int = 
    shape.headOption.getOrElse(0)
  def shape(axis: Int): List[Int] = 
    shape.drop(axis)

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

case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T]:
  override def shape: List[Int] = List(data.length)

  override def toString: String =
    val meta = s"shape: ${shape.head}, Tensor1D[${summon[ClassTag[T]]}]"
    s"$meta:\n[" + data.mkString(",") + "]\n"

  override def length: Int = data.length

object Tensor1D:
  def apply[T: ClassTag](data: T*): Tensor1D[T] = 
    Tensor1D[T](data.toArray)

case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T]:
  override def shape: List[Int] =
    shape2D.toList    

  def shape2D: (Int, Int) =
    (data.length, data.headOption.map(_.length).getOrElse(0))

  private val meta =
    s"shape: ${shape.mkString("x")}, Tensor2D[${summon[ClassTag[T]]}]"

  override def toString: String =
    s"$meta:\n[" + data
      .map(_.mkString("[", ",", "]"))
      .mkString("\n ") + "]\n"  

  override def length: Int = data.length

  override def shape(axis: Int) = 
    shape.drop(axis)

object Tensor2D:
  def apply[T: ClassTag](rows: Array[T]*): Tensor2D[T] =
    Tensor2D[T](rows.toArray)

case class Tensor3D[T: ClassTag](data: Array[Array[Array[T]]]) extends Tensor[T]:
  def shape3D: (Int, Int, Int) =
    val rows = data.headOption.map(_.length).getOrElse(0)
    val cols = data.headOption.flatMap(_.headOption.map(_.length)).getOrElse(0)
    (data.length, rows, cols)
  
  override def shape: List[Int] =
    shape3D.toList    

  override def length: Int = data.length

object Tensor3D:
  def apply[T: ClassTag](matrices: Tensor2D[T]*): Tensor3D[T] =
    Tensor3D(matrices.toArray.map(t => t.data))

case class Tensor4D[T: ClassTag](data: Array[Array[Array[Array[T]]]]) extends Tensor[T]:
  def shape4D: (Int, Int, Int, Int) =
    val cubes = data.headOption.map(_.length).getOrElse(0)
    val rows = data.headOption.flatMap(_.headOption.map(_.length)).getOrElse(0)
    val cols = for {
      cube <- data.headOption
      row <- cube.headOption
      col <- row.headOption
    } yield col.length

    (data.length, cubes, rows, cols.getOrElse(0))
  
  override def shape: List[Int] =
    shape4D.toList    

  override def length: Int = data.length

object Tensor4D:
  def apply[T: ClassTag](cubes: Array[Tensor2D[T]]*): Tensor4D[T] =
    Tensor4D(cubes.toArray.map(t => t.map(_.data)))