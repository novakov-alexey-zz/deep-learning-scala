package ml.tensors

import ml.tensors.api._
import ml.transformation.castFromTo

import scala.reflect.ClassTag
import math.Numeric.Implicits.infixNumericOps
import math.Ordering.Implicits.infixOrderingOps
import math.Fractional.Implicits.infixFractionalOps
import math.Integral.Implicits.infixIntegralOps

private trait genOps:    
  extension [T: ClassTag: Numeric](t: Tensor[T])
    // dot product    
    def *(that: Tensor[T]): Tensor[T] = TensorOps.mul(t, that)    
    def -(that: T): Tensor[T] = TensorOps.subtract(t, Tensor0D(that))
    def -(that: Tensor[T]): Tensor[T] = TensorOps.subtract(t, that)
    def +(that: Tensor[T]): Tensor[T] = TensorOps.plus(t, that)    
    def +(that: T): Tensor[T] = TensorOps.plus(t, Tensor0D(that))    
    def sum: T = TensorOps.sum(t)        
    def split(fraction: Float): (Tensor[T], Tensor[T]) = TensorOps.split(fraction, t)
    // Hadamard product
    def multiply(that: Tensor[T]): Tensor[T] = TensorOps.multiply(t, that)
    def batches(batchSize: Int): Iterator[Array[Array[T]]] = TensorOps.batches(t, batchSize)
    def equalRows(that: Tensor[T]): Int = TensorOps.equalRows(t, that)
    def clipInRange(min: T, max: T): Tensor[T] = TensorOps.clipInRange(t, min, max)
    def :**(to: Int): Tensor[T] = TensorOps.pow(t, to)
    def sqr: Tensor[T] = TensorOps.pow(t, 2)    
    def sqrt: Tensor[T] = TensorOps.sqrt(t)
    def zero: Tensor[T] = TensorOps.zero(t)

  extension [T: ClassTag: Fractional](t: Tensor[T])
    def /(that: Tensor[T]): Tensor[T] = TensorOps.div(t, that)
    def :/(that: T): Tensor[T] = TensorOps.div(t, Tensor0D(that))

  extension [T: ClassTag](t: Tensor[T])
    def T: Tensor[T] = TensorOps.transpose(t)    
    def map[U: ClassTag](f: T => U): Tensor[U] = TensorOps.map[T, U](t, f)          

object ops extends genOps:
  extension [T: ClassTag](t: Tensor2D[T])
    def col(i: Int): Tensor1D[T] = Tensor1D(TensorOps.col(t.data, i))
    def T: Tensor2D[T] = TensorOps.transpose(t).asInstanceOf[Tensor2D[T]]
    
  extension [T: ClassTag](t: Tensor[T])
    def as1D: Tensor1D[T] = TensorOps.as1D(t)    
    def as2D: Tensor2D[T] = TensorOps.as2D(t)

  extension [T: ClassTag](t: T)
    def asT: Tensor[T] = Tensor0D(t)
    def as0D: Tensor0D[T] = Tensor0D(t)
    def as1D: Tensor1D[T] = Tensor1D(Array(t))
    def as2D: Tensor2D[T] = Tensor2D(Array(Array(t)))       
    
  extension [T: ClassTag](t: T)(using n: Numeric[T])    
    def **(to: Int): T = castFromTo[Double, T](math.pow(n.toDouble(t), to))
  
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

  extension [T: ClassTag](a: Array[Array[T]])
    def col(i: Int): Array[T] = TensorOps.col(a, i)
    def slice(          
          rows: Option[(Int, Int)] = None,
          cols: Option[(Int, Int)] = None
      ): Array[Array[T]] = TensorOps.slice(a, rows, cols)

  extension [T: ClassTag: Numeric](t: List[Tensor[T]])
    def combineAllAs1D: Tensor1D[T] = TensorOps.combineAllAs1D(t)

  extension [T: ClassTag: Numeric](pair: (Tensor[T], Tensor[T]))
    def map2[U: ClassTag: Numeric](f: (T, T) => U): Tensor[U] = 
      TensorOps.map2(pair._1, pair._2, f)

    def split(
        fraction: Float
    ): ((Tensor[T], Tensor[T]), (Tensor[T], Tensor[T])) =
      TensorOps.split(fraction, pair)

  extension [T: ClassTag](t: Tensor1D[T])
    def batchColumn(batchSize: Int): Iterator[Array[T]] =
      t.data.grouped(batchSize)

object TensorOps:

  def subtract[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match
      case (Tensor1D(data), Tensor1D(data2)) =>
        assert(
          data.length == data2.length,
          s"Vectors must have the same length, ${data.length} ! = ${data2.length}"
        )        
        Tensor1D(data.zip(data2).map(_ - _))
      case (t1 @ Tensor2D(data), t2 @ Tensor2D(data2)) =>
        val (rows, cols) = t1.shape2D
        val (rows2, cols2) = t2.shape2D
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
        Tensor1D(data.zip(data2).map {(a, b) => a + b }) // TODO: check both shapes are equal
      case (Tensor2D(data), Tensor0D(data2)) =>
        Tensor2D(data.map(_.map(_ + data2)))
      case (Tensor2D(data), Tensor2D(data2)) =>
        Tensor2D(data.zip(data2).map((a, b) => a.zip(b).map(_ + _))) // TODO: check both shapes are equal
      case (Tensor0D(data), Tensor2D(data2)) =>
        Tensor2D(data2.map(_.map(_ + data)))
      case (t1 @ Tensor2D(_), t2 @ Tensor1D(_)) =>
        matrixPlusVector(t1, t2)
      case (t1 @ Tensor1D(_), t2 @ Tensor2D(_)) =>
        matrixPlusVector(t2, t1)
      case (Tensor1D(data), Tensor0D(data2)) =>
        Tensor1D(data.map(_ + data2))
      case (Tensor0D(data), Tensor1D(data2)) =>
        Tensor1D(data2.map(_ + data))
      case (Tensor0D(data), Tensor0D(data2)) =>
        Tensor0D(data + data2)      

  private def matrixPlusVector[T: ClassTag: Numeric](
      t1: Tensor2D[T],
      t2: Tensor1D[T]
  ) =
    val (rows, cols) = t1.shape2D
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
      case _ => 
        sys.error(s"Both tensors must have the same dimension: ${a.shape} != ${b.shape}")

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

  private def matMul[T: ClassTag](
      a: Array[Array[T]],
      b: Array[Array[T]]
  )(using n: Numeric[T]): Array[Array[T]] =
    assert(
      a.head.length == b.length,
      s"The number of columns in the first matrix should be equal to the number of rows in the second, ${a.head.length} != ${b.length}"
    )
    val rows = a.length
    val cols = colsCount(b)
    val res = Array.ofDim[T](rows, cols)

    for i <- (0 until rows).indices do
      for j <- (0 until cols).indices do
        var sum = n.zero
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
      case (t1 @ Tensor1D(_), Tensor2D(data2))  => combine(t1, Tensor1D(data2.flatten))
      case (Tensor2D(data), Tensor0D(data2))    => Tensor1D(data.flatten :+ data2)
      case (Tensor2D(data), t2 @ Tensor1D(_))   => combine(Tensor1D(data.flatten), t2)
      case (Tensor2D(data), Tensor2D(data2))    =>  combine(Tensor1D(data.flatten), Tensor1D(data2.flatten))

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
        val (rows, cols) = t2.shape2D
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
        val (rows, cols) = a.shape2D
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
    assert(t1.shape == t2.shape, sys.error(s"Tensors must be the same dimension: ${t1.shape} != ${t2.shape}"))
    (t1, t2) match
      case (Tensor0D(data), Tensor0D(data2)) => 
        if data == data2 then 1 else 0
      case (Tensor1D(data), Tensor1D(data2)) => 
        data.zip(data2).foldLeft(0) { case (acc, (a, b)) => if a == b then acc + 1 else acc }
      case (Tensor2D(data), Tensor2D(data2)) => 
        data.zip(data2).foldLeft(0) { case (acc, (a, b)) => if a.sameElements(b) then acc + 1 else acc }
      case _ => 
        sys.error(s"Tensors must be the same dimension: ${t1.shape} != ${t2.shape}")
  
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
  
  def div[T: ClassTag: Fractional](t1: Tensor[T], t2: Tensor[T]): Tensor[T] =    
    (t1, t2) match
      case (Tensor2D(data), Tensor0D(data2)) => Tensor2D(data.map(_.map(_ / data2)))
      case (Tensor0D(data), Tensor0D(data2)) => Tensor0D(data / data2)
      case (Tensor1D(data), Tensor1D(data2)) => Tensor1D(data.zip(data2).map(_ /_))
      case (Tensor1D(data), Tensor0D(data2)) => Tensor1D(data.map(_ / data2))
      case (Tensor2D(data), Tensor2D(data2)) => 
        val res = data.zip(data2).map((a, b) => a.zip(b).map(_ / _))
        Tensor2D(res)
      case _ => sys.error(s"Not implemented for $t1 \n and $t2")
  
  def sqrt[T: ClassTag: Numeric](t: Tensor[T]): Tensor[T] = 
    map(t, v => castFromTo[Double, T](math.sqrt(castFromTo[T, Double](v))))

  def pow[T: ClassTag](t: Tensor[T], to: Int)(using n: Numeric[T]): Tensor[T] =
    def powValue(v: T) =
      val res = math.pow(n.toDouble(v), to)
      castFromTo[Double, T](res)

    t match
      case Tensor0D(data) => Tensor0D(powValue(data))
      case Tensor1D(data) => Tensor1D(data.map(powValue))
      case Tensor2D(data) => Tensor2D(data.map(_.map(powValue)))
  
  def zero[T: ClassTag](t: Tensor[T])(using n: Numeric[T]): Tensor[T] =
    map(t, _ => n.zero)
  
  def col[T: ClassTag](data: Array[Array[T]], i: Int): Array[T] =
    val to = i + 1
    slice(data, None, Some(i, to)).flatMap(_.headOption)

  def slice[T: ClassTag](
      data: Array[Array[T]],
      rows: Option[(Int, Int)] = None,
      cols: Option[(Int, Int)] = None
  ): Array[Array[T]] =
    (rows, cols) match
      case (Some((rowsFrom, rowsTo)), Some((colsFrom, colsTo))) =>
        sliceArr(data, (rowsFrom, rowsTo)).map(a =>
          sliceArr(a, (colsFrom, colsTo))
        )
      case (None, Some((colsFrom, colsTo))) =>
        data.map(a => sliceArr(a, (colsFrom, colsTo)))
      case (Some((rowsFrom, rowsTo)), None) =>
        sliceArr(data, (rowsFrom, rowsTo))
      case _ => data

  def sliceArr[T](
      data: Array[T],
      range: (Int, Int)
  ): Array[T] =
    val (l, r) = range
    val from = if l < 0 then data.length + l else l
    val to = if r < 0 then data.length + r else if r == 0 then data.length else r
    data.slice(from, to)