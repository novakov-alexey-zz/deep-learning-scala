package ml.tensors

import ml.tensors.api._
import scala.reflect.ClassTag
import math.Numeric.Implicits.infixNumericOps
import math.Ordering.Implicits.infixOrderingOps

private trait genOps:    
  extension [T: ClassTag: Numeric](t: Tensor[T])
    // dot product    
    def *(that: Tensor[T]): Tensor[T] = TensorOps.mul(t, that)    
    def -(that: T): Tensor[T] = TensorOps.subtract(t, Tensor0D(that))
    def -(that: Tensor[T]): Tensor[T] = TensorOps.subtract(t, that)
    def +(that: Tensor[T]): Tensor[T] = TensorOps.plus(t, that)    
    def sum: T = TensorOps.sum(t)        
    def split(fraction: Float): (Tensor[T], Tensor[T]) = TensorOps.split(fraction, t)
    // Hadamard product
    def multiply(that: Tensor[T]): Tensor[T] = TensorOps.multiply(t, that)
    def batches(
        batchSize: Int
    ): Iterator[Array[Array[T]]] = TensorOps.batches(t, batchSize)
    def equalRows(that: Tensor[T]): Int = TensorOps.equalRows(t, that)
    def clipInRange(min: T, max: T): Tensor[T] = TensorOps.clipInRange(t, min, max)

object ops extends genOps:
  extension [T: ClassTag](t: Tensor2D[T])
    def col(i: Int): Tensor1D[T] = Tensor1D(Tensor2D.col(t.data, i))

    def T: Tensor2D[T] = TensorOps.transpose(t).asInstanceOf[Tensor2D[T]]

  extension [T: ClassTag: Numeric, U: ClassTag](t: Tensor[T])  
    def map(f: T => U): Tensor[U] = TensorOps.map[T, U](t, f)
  
  extension [T: ClassTag](t: Tensor[T])
    def T: Tensor[T] = TensorOps.transpose(t)
    def as1D: Tensor1D[T] = TensorOps.as1D(t)
    def as2D: Tensor2D[T] = TensorOps.as2D(t)    

  extension [T: ClassTag](t: T)
    def as0D: Tensor0D[T] = Tensor0D(t)
    def as1D: Tensor1D[T] = Tensor1D(Array(t))
    def as2D: Tensor2D[T] = Tensor2D(Array(Array(t)))

  implicit class Tensor0DOps[T: ClassTag: Numeric](val t: T):
    // dot product
    def *(that: Tensor[T]): Tensor[T] = TensorOps.mul(Tensor0D(t), that)
    def -(that: Tensor[T]): Tensor[T] = TensorOps.subtract(Tensor0D(t), that)    
  
  extension [T: ClassTag: Numeric](t: Array[Tensor[T]]) 
    def combineAllAs1D: Tensor1D[T] = TensorOps.combineAllAs1D(t)  
  
  extension [T: ClassTag: Numeric](a: Array[T])
    def as1D: Tensor1D[T] = Tensor1D(a)
  
  extension [T: ClassTag: Numeric](a: Array[Array[T]])
    def as2D: Tensor2D[T] = Tensor2D(a)

  extension [T: ClassTag: Numeric](t: List[Tensor[T]])
    def combineAllAs1D: Tensor1D[T] = TensorOps.combineAllAs1D(t)

  extension [T: ClassTag: Numeric, U: ClassTag: Numeric](pair: (Tensor[T], Tensor[T]))
    def map2(f: (T, T) => U): Tensor[U] = 
      TensorOps.map2(pair._1, pair._2, f)

  extension [T: ClassTag: Numeric](pair: (Tensor[T], Tensor[T]))
    def split(
        fraction: Float
    ): ((Tensor[T], Tensor[T]), (Tensor[T], Tensor[T])) =
      TensorOps.split(fraction, pair)

  extension [T: ClassTag](t: Tensor1D[T])
    def batchColumn(batchSize: Int): Iterator[Array[T]] =
      t.data.grouped(batchSize)

object TensorOps:
  def of[T: ClassTag](size: Int): Tensor1D[T] =
    Tensor1D[T](Array.ofDim[T](size))

  def of[T: ClassTag](size: Int, size2: Int): Tensor2D[T] =
    Tensor2D[T](Array.fill(size)(of[T](size2).data))

  def subtract[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match
      case (Tensor1D(data), Tensor1D(data2)) =>
        assert(
          data.length == data2.length,
          s"Vectors must have the same length, ${data.length} ! = ${data2.length}"
        )
        val res = data.zip(data2).map { case (a, b) => a - b }
        Tensor1D(res)
      case (t1 @ Tensor2D(data), t2 @ Tensor2D(data2)) =>
        val (rows, cols) = t1.sizes2D
        val (rows2, cols2) = t2.sizes2D
        assert(
          rows == rows2 && cols == cols2,
          s"Matrices must have the same amount of rows and size, ${(rows, cols)} ! = ${(rows2, cols2)}"
        )
        val res = Array.ofDim[T](rows, cols)
        for i <- data.indices do
          for j <- 0 until cols do
            res(i)(j) = data(i)(j) - data2(i)(j)
        Tensor2D(res)
      case (Tensor2D(data), Tensor0D(data2)) =>
        Tensor2D(data.map(_.map(_ - data2)))
      case (Tensor1D(data), Tensor0D(data2)) =>
        Tensor1D(data.map(_ - data2))
      case (t1 @ Tensor2D(_), t2 @ Tensor1D(_)) =>
        matrixSubstractVector(t1, t2)
      case (t1, t2) => sys.error(s"Not implemented for\n$t1 and\n$t2")

  private def matrixSubstractVector[T: Numeric: ClassTag](
      matrix: Tensor2D[T],
      vector: Tensor1D[T]
  ) =
    val cols = matrix.cols
    assert(
      cols == vector.length,
      s"trailing axis must have the same size, $cols != ${vector.length}"
    )
    val res = matrix.data.map(_.zip(vector.data).map{(a, b) => a - b })
    Tensor2D(res)

  def plus[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor1D(data.zip(data2).map {(a, b) => a + b })
      case (Tensor2D(data), Tensor0D(data2)) =>
        Tensor2D(data.map(_.map(_ + data2)))
      case (t1 @ Tensor2D(_), t2 @ Tensor1D(_)) =>
        matrixPlusVector(t1, t2)
      case (t1 @ Tensor1D(_), t2 @ Tensor2D(_)) =>
        matrixPlusVector(t2, t1)
      case (Tensor1D(data), Tensor0D(data2)) =>
        Tensor1D(data.map(_ + data2))
      case (Tensor0D(data), Tensor1D(data2)) =>
        Tensor1D(data2.map(_ + data))
      case _ => sys.error(s"Not implemented for:\n$a\nand\n$b!")

  private def matrixPlusVector[T: ClassTag: Numeric](
      t1: Tensor2D[T],
      t2: Tensor1D[T]
  ) =
    val (rows, cols) = t1.sizes2D
    assert(
      cols == t2.length,
      s"tensors must have the same amount of cols to sum them up element-wise, but were: $cols != ${t2.length}"
    )
    val sum = Array.ofDim[T](rows, cols)
    for i <- 0 until rows do
      for j <- 0 until cols do
        sum(i)(j) = t1.data(i)(j) + t2.data(j)
    Tensor2D(sum)

  def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match
      case (Tensor0D(data), t) =>
        scalarMul(t, data)
      case (t, Tensor0D(data)) =>
        scalarMul(t, data)
      case (Tensor1D(data), Tensor2D(data2)) =>
        Tensor2D(matMul(asColumn(data), data2))
      case (Tensor2D(data), Tensor1D(data2)) =>
        Tensor2D(matMul(data, asColumn(data2)))
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor1D(matMul(asColumn(data), Array(data2)).head)
      case (Tensor2D(data), Tensor2D(data2)) =>
        Tensor2D(matMul(data, data2))

  private def asColumn[T: ClassTag](a: Array[T]) = a.map(Array(_))

  def map[T: ClassTag, U: ClassTag](t: Tensor[T], f: T => U): Tensor[U] =
    t match
      case Tensor0D(data) => Tensor0D(f(data))
      case Tensor1D(data) => Tensor1D(data.map(f))
      case Tensor2D(data) => Tensor2D(data.map(_.map(f)))
  
  private def map2[T: ClassTag, U: ClassTag](a: Array[T], b: Array[T], f: (T, T) => U): Array[U] = 
    val res = Array.ofDim[U](a.length)
    for i <- (0 until a.length).indices do
      res(i) = f(a(i), b(i))
    res  

  def map2[T: ClassTag: Numeric, U: ClassTag: Numeric](a: Tensor[T], b: Tensor[T], f: (T, T) => U): Tensor[U] =
    (a, b) match
      case (Tensor0D(data), Tensor0D(data2)) => 
        Tensor0D(f(data, data2))
      case (Tensor1D(data), Tensor1D(data2)) =>                 
        Tensor1D(map2(data, data2, f))
      case (Tensor2D(data), Tensor2D(data2)) =>
        val res = Array.ofDim[U](data.length, colsCount(data2))
        for i <- (0 until data.length).indices do
          res(i) = map2(data(i), data2(i), f)
        Tensor2D(res)
      case _ => sys.error(s"Both tensors must be the same dimenion: ${a.sizes} != ${b.sizes}")

  private def colsCount[T](a: Array[Array[T]]): Int =
    a.headOption.map(_.length).getOrElse(0)

  private def scalarMul[T: ClassTag: Numeric](
      t: Tensor[T],
      scalar: T
  ): Tensor[T] =
    t match
      case Tensor0D(data) => Tensor0D(data * scalar)
      case Tensor1D(data) => Tensor1D(data.map(_ * scalar))
      case Tensor2D(data) => Tensor2D(data.map(_.map(_ * scalar)))

  private def matMul[T: ClassTag: Numeric](
      a: Array[Array[T]],
      b: Array[Array[T]]
  ): Array[Array[T]] =
    assert(
      a.head.length == b.length,
      s"The number of columns in the first matrix should be equal to the number of rows in the second, ${a.head.length} != ${b.length}"
    )
    val rows = a.length
    val cols = colsCount(b)
    val res = Array.ofDim[T](rows, cols)

    for i <- (0 until rows).indices do
      for j <- (0 until cols).indices do
        var sum = summon[Numeric[T]].zero
        for k <- b.indices do
          sum = sum + (a(i)(k) * b(k)(j))
        res(i)(j) = sum
    res

  def combineAll[T: ClassTag](ts: List[Tensor1D[T]]): Tensor1D[T] =
    ts.reduce[Tensor1D[T]] { case (a, b) => TensorOps.combine(a, b) }

  def combine[T: ClassTag](a: Tensor1D[T], b: Tensor1D[T]): Tensor1D[T] =
    Tensor1D(a.data ++ b.data)

  def combineAllAs1D[T: ClassTag](ts: Iterable[Tensor[T]]): Tensor1D[T] =
    ts.foldLeft(Tensor1D[T]()) { case (a, b) => combineAs1D[T](a, b) }

  def combineAs1D[T: ClassTag](a: Tensor[T], b: Tensor[T]): Tensor1D[T] =
    (a, b) match
      case (Tensor1D(data), Tensor0D(data2))    => Tensor1D(data :+ data2)
      case (Tensor0D(data), Tensor0D(data2))    => Tensor1D(Array(data, data2))
      case (Tensor0D(data), Tensor1D(data2))    => Tensor1D(data +: data2)
      case (Tensor0D(data), Tensor2D(data2))    => Tensor1D(data +: data2.flatten)
      case (t1 @ Tensor1D(_), t2 @ Tensor1D(_)) => combine(t1, t2)
      case (t1 @ Tensor1D(_), Tensor2D(data2)) =>
        combine(t1, Tensor1D(data2.flatten))
      case (Tensor2D(data), Tensor0D(data2)) =>
        Tensor1D(data.flatten :+ data2)
      case (Tensor2D(data), t2 @ Tensor1D(_)) =>
        combine(Tensor1D(data.flatten), t2)
      case (Tensor2D(data), Tensor2D(data2)) =>
        combine(Tensor1D(data.flatten), Tensor1D(data2.flatten))

  def as1D[T: ClassTag](t: Tensor[T]): Tensor1D[T] =
    t match
      case Tensor0D(data)   => Tensor1D(data)
      case t1 @ Tensor1D(_) => t1
      case Tensor2D(data)   => Tensor1D(data.flatten)

  def as2D[T: ClassTag](t: Tensor[T]): Tensor2D[T] =
    t match
      case Tensor0D(data)   => Tensor2D(Array(Array(data)))
      case Tensor1D(data)   => Tensor2D(data.map(Array(_)))
      case t2 @ Tensor2D(_) => t2

  def sum[T: Numeric: ClassTag](t: Tensor[T]): T =
    t match
      case Tensor0D(data) => data
      case Tensor1D(data) => data.sum
      case Tensor2D(data) => data.map(_.sum).sum

  def transpose[T: ClassTag](t: Tensor[T]): Tensor[T] =
    t match
      case t2 @ Tensor2D(data) =>
        val (rows, cols) = t2.sizes2D
        val transposed = Array.ofDim[T](cols, rows)

        for i <- (0 until rows).indices do
          for j <- (0 until cols).indices do
            transposed(j)(i) = data(i)(j)
        Tensor2D(transposed)
      case Tensor1D(data) => Tensor2D(asColumn(data))
      case _              => t

  def split[T: ClassTag](
      fraction: Float,
      t: Tensor[T]
  ): (Tensor[T], Tensor[T]) =
    t match
      case Tensor0D(_) => (t, t)
      case Tensor1D(data) =>
        val (l, r) = splitArray(fraction, data)
        (Tensor1D(l), Tensor1D(r))
      case Tensor2D(data) =>
        val (l, r) = splitArray(fraction, data)
        (Tensor2D(l), Tensor2D(r))

  private def splitArray[T](
      fraction: Float,
      data: Array[T]
  ): (Array[T], Array[T]) =
    val count = data.length * fraction
    val countOrZero = if count < 1 then 0 else count
    data.splitAt(data.length - countOrZero.toInt)

  def split[T: ClassTag](
      fraction: Float,
      t: (Tensor[T], Tensor[T])
  ): ((Tensor[T], Tensor[T]), (Tensor[T], Tensor[T])) =
    val (l, r) = t
    assert(l.length == r.length, s"Both tensors must have the same length, ${l.length} != ${r.length}")
    split(fraction, l) -> split(fraction, r)

  def multiply[T: Numeric: ClassTag](
      t1: Tensor[T],
      t2: Tensor[T]
  ): Tensor[T] =    
    assert(
      t1.length == t2.length,
      s"Both vectors must have the same length, ${t1.length} != ${t2.length}"
    )
    (t1, t2) match
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor1D(data.zip(data2).map((a, b) => a * b))
      case (a @ Tensor2D(data), Tensor2D(data2)) =>
        val (rows, cols) = a.sizes2D
        val sum = Array.ofDim[T](rows, cols)
        for i <- 0 until rows do
          for j <- 0 until cols do
            sum(i)(j) = data(i)(j) * data2(i)(j)
        Tensor2D(sum)
      case (a, b) => sys.error(s"Not implemented for:\n$a\nand\n$b")

  def batches[T: ClassTag: Numeric](
      t: Tensor[T],
      batchSize: Int
  ): Iterator[Array[Array[T]]] =
    t match
      case Tensor1D(_)    => as2D(t).data.grouped(batchSize)
      case Tensor2D(data) => data.grouped(batchSize)
      case Tensor0D(data) => Iterator(Array(Array(data)))

  def equalRows[T: ClassTag](t1: Tensor[T], t2: Tensor[T]): Int = 
    assert(t1.sizes == t2.sizes, sys.error(s"Tensors must be the same dimension: ${t1.sizes} != ${t2.sizes}"))
    (t1, t2) match
      case (Tensor0D(data), Tensor0D(data2)) => 
        if data == data2 then 1 else 0
      case (Tensor1D(data), Tensor1D(data2)) => 
        data.zip(data2).foldLeft(0) { case (acc, (a, b)) => if a == b then acc + 1 else acc }
      case (Tensor2D(data), Tensor2D(data2)) => 
        data.zip(data2).foldLeft(0) { case (acc, (a, b)) => if a.sameElements(b) then acc + 1 else acc }
      case _ => 
        sys.error(s"Tensors must be the same dimension: ${t1.sizes} != ${t2.sizes}")
  
  def clipInRange[T: ClassTag](t: Tensor[T], min: T, max: T)(using n: Numeric[T]): Tensor[T] =
    def clipValue(v: T) =
      val vAbs = v.abs          
          if vAbs > max then max
          else if vAbs < min then min
          else v

    def clipArray(data: Array[T]) =
      data.map(clipValue)

    t match 
      case Tensor2D(data) => Tensor2D(data.map(clipArray))
      case Tensor1D(data) => Tensor1D(clipArray(data))
      case Tensor0D(data) => Tensor0D(clipValue(data))          