// scala 2.13.3

import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

sealed trait Tensor[T] {
  type A
  def data: A
  def length: Int
  def sizes: List[Int]
}

case class Tensor0D[T: ClassTag](data: T) extends Tensor[T] {
  type A = T
  override def length: Int = 0
  override def sizes: List[Int] = Nil
  override def toString: String = {
    val meta = s"sizes: ${sizes.head}, Tensor0D[${implicitly[ClassTag[T]]}]"
    s"$meta:\n" + data + "\n"
  }
}

case class Tensor1D[T: ClassTag](data: Array[T]) extends Tensor[T] {
  type A = Array[T]

  override def sizes: List[Int] = List(data.length)

  override def toString: String = {
    val meta = s"sizes: ${sizes.head}, Tensor1D[${implicitly[ClassTag[T]]}]"
    s"$meta:\n[" + data.mkString(",") + "]\n"
  }
  override def length: Int = data.length
}

object Tensor1D {
  def apply[T: ClassTag](data: T*): Tensor1D[T] = Tensor1D[T](data.toArray)
}

case class Tensor2D[T: ClassTag](data: Array[Array[T]]) extends Tensor[T] {
  type A = Array[Array[T]]

  override def sizes: List[Int] =
    List(_sizes._1, _sizes._2)

  private def _sizes: (Int, Int) =
    (data.length, data.headOption.map(_.length).getOrElse(0))

  override def toString: String = {
    val meta =
      s"sizes: ${sizes.mkString("x")}, Tensor2D[${implicitly[ClassTag[T]]}]"
    s"$meta:\n[" + data
      .map(a => a.mkString("[", ",", "]"))
      .mkString("\n ") + "]\n"
  }

  def cols: Int = _sizes._2

  override def length: Int = data.length
}

object Tensor2D {
  def apply[T: ClassTag](rows: Array[T]*): Tensor2D[T] =
    Tensor2D[T](rows.toArray)

  def col[T: ClassTag](data: Array[Array[T]], i: Int): Array[T] = {
    val to = i + 1
    slice(data, None, Some(i, to)).flatMap(_.headOption)
  }

  def slice[T: ClassTag](
      data: Array[Array[T]],
      rows: Option[(Int, Int)] = None,
      cols: Option[(Int, Int)] = None
  ): Array[Array[T]] =
    (rows, cols) match {
      case (Some((rowsFrom, rowsTo)), Some((colsFrom, colsTo))) =>
        sliceArr(data, (rowsFrom, rowsTo)).map(a =>
          sliceArr(a, (colsFrom, colsTo))
        )
      case (None, Some((colsFrom, colsTo))) =>
        data.map(a => sliceArr(a, (colsFrom, colsTo)))
      case (Some((rowsFrom, rowsTo)), None) =>
        sliceArr(data, (rowsFrom, rowsTo))
      case _ => data
    }

  def sliceArr[T](
      data: Array[T],
      range: (Int, Int)
  ): Array[T] = {
    // println(s"range: $range")
    val (l, r) = range
    val from = if (l < 0) data.length + l else l
    val to = if (r < 0) data.length + r else if (r == 0) data.length else r

    // println(s"from = $from, to = $to")    
    data.slice(from, to)
  }
}

implicit class TensorOps[T: ClassTag: Numeric](val t: Tensor[T]) {
  def *(that: Tensor[T]): Tensor[T] = Tensor.mul(t, that)
  def map(f: T => T) = Tensor.map(t, f)
  def -(that: T): Tensor[T] = Tensor.substract(t, Tensor0D(that))
}

implicit class Tensor0DOps[T: ClassTag: Numeric](val t: T) {
  def *(that: Tensor[T]): Tensor[T] = Tensor.mul(Tensor0D(t), that)
  def -(that: Tensor[T]): Tensor[T] = Tensor.substract(Tensor0D(t), that)
}

implicit class TensorListOps[T: ClassTag: Numeric](val t: Array[Tensor[T]]) {
  def combineAllAs1D: Tensor1D[T] = Tensor.combineAllAs1D(t)
}

implicit class Tensor2DOps[T: ClassTag](val t: Tensor2D[T]) {
  def col(i: Int): Tensor1D[T] = Tensor1D(Tensor2D.col(t.data, i))
}

object Tensor {
  def of[T: ClassTag](size: Int): Tensor1D[T] =
    Tensor1D[T](Array.ofDim[T](size))

  def of[T: ClassTag](size: Int, size2: Int): Tensor2D[T] =
    Tensor2D[T](Array.fill(size)(of[T](size2).data))

  def substract[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match {
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor1D(data.flatMap(d => data2.map(d2 => d - d2)))
      case (Tensor2D(data), Tensor0D(data2)) =>
        Tensor2D(data.map(_.map(_ - data2)))
      case (Tensor1D(data), Tensor0D(data2)) =>
        Tensor1D(data.map(_ - data2))
      case _ => sys.error("Not implemented!")
    }

  def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match {
      case (Tensor0D(data), t) =>
        scalarMul(t, data)
      case (t, Tensor0D(data)) =>
        scalarMul(t, data)
      case (Tensor1D(data), Tensor2D(data2)) =>
        Tensor2D[T](matMul[T](asColumn(data), data2))
      case (Tensor2D(data), Tensor1D(data2)) =>
        Tensor2D[T](matMul[T](data, asColumn(data2)))
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor1D[T](matMul[T](asColumn(data), Array(data2)).head)
      case (Tensor2D(data), Tensor2D(data2)) =>
        Tensor2D[T](matMul[T](data, data2))
    }

  private def asColumn[T: ClassTag](a: Array[T]) = a.map(Array(_))

  def map[T: ClassTag](t: Tensor[T], f: T => T): Tensor[T] =
    t match {
      case Tensor1D(data) => Tensor1D(data.map(f(_)))
      case Tensor2D(data) => Tensor2D(data.map(d => d.map(f(_))))
    }

  private def scalarMul[T: ClassTag: Numeric](
      t: Tensor[T],
      scalar: T
  ): Tensor[T] =
    t match {
      case Tensor0D(data) => Tensor0D(data * scalar)
      case Tensor1D(data) => Tensor1D(data.map(_ * scalar))
      case Tensor2D(data) => Tensor2D(data.map(_.map(_ * scalar)))
    }

  private def matMul[T: ClassTag: Numeric](
      a: Array[Array[T]],
      b: Array[Array[T]]
  ): Array[Array[T]] = {
    val rows = a.length
    val cols = b.headOption.map(_.length).getOrElse(0)
    val res = Array.ofDim[T](rows, cols)

    for (i <- (0 until rows).indices) {
      for (j <- (0 until cols).indices) {
        var sum = implicitly[Numeric[T]].zero
        for (k <- b.indices) {
          sum = sum + (a(i)(k) * b(k)(j))
        }
        res(i)(j) = sum
      }
    }
    res
  }

  def combineAll[T: ClassTag](ts: List[Tensor1D[T]]): Tensor1D[T] =
    ts.reduce[Tensor1D[T]] { case (a, b) => Tensor.combine(a, b) }

  def combine[T: ClassTag](a: Tensor1D[T], b: Tensor1D[T]): Tensor1D[T] =
    Tensor1D(a.data ++ b.data)

  def combineAllAs1D[T: ClassTag](ts: Iterable[Tensor[T]]): Tensor1D[T] =
    ts.foldLeft(Tensor1D[T]()) { case (a, b) => combineAs1D[T](a, b) }

  def combineAs1D[T: ClassTag](a: Tensor[T], b: Tensor[T]): Tensor1D[T] =
    (a, b) match {
      case (t1 @ Tensor1D(_), t2 @ Tensor1D(_)) => combine(t1, t2)
      case (t1 @ Tensor1D(_), Tensor2D(data2)) =>
        combine(t1, Tensor1D(data2.flatten))
      case (Tensor2D(data), t2 @ Tensor1D(_)) =>
        combine(Tensor1D(data.flatten), t2)
      case (Tensor2D(data), Tensor2D(data2)) =>
        combine(Tensor1D(data.flatten), Tensor1D(data2.flatten))
    }
}
