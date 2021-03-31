package ml.tensors

import ml.tensors.api._
import ml.transformation.castFromTo

import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer
import math.Numeric.Implicits.infixNumericOps
import math.Ordering.Implicits.infixOrderingOps
import math.Fractional.Implicits.infixFractionalOps
import math.Integral.Implicits.infixIntegralOps

private trait genOps:    
  extension [T: ClassTag: Numeric](t: Tensor[T])
    // dot product    
    def *(that: Tensor[T]): Tensor[T] = TensorOps.mul(t, that)    
    def *(that: Option[Tensor[T]]): Tensor[T] = TensorOps.optMul(t, that)    
    def *(that: T): Tensor[T] = TensorOps.mul(t, Tensor0D(that))
    def -(that: T): Tensor[T] = TensorOps.subtract(t, Tensor0D(that))
    def -(that: Tensor[T]): Tensor[T] = TensorOps.subtract(t, that)
    def +(that: Tensor[T]): Tensor[T] = TensorOps.plus(t, that)    
    def +(that: Option[Tensor[T]]): Tensor[T] = TensorOps.optPlus(t, that)    
    def +(that: T): Tensor[T] = TensorOps.plus(t, Tensor0D(that))    
    def sum: T = TensorOps.sum(t)        
    def split(fraction: Float): (Tensor[T], Tensor[T]) = TensorOps.split(fraction, t)
    
    // Hadamard product
    def multiply(that: Tensor[T]): Tensor[T] = TensorOps.multiply(t, that)
    def multiply(that: Option[Tensor[T]]): Tensor[T] = TensorOps.optMultiply(t, that)
    def |*|(that: Tensor[T]): Tensor[T] = TensorOps.multiply(t, that)
    def |*|(that: Option[Tensor[T]]): Tensor[T] = TensorOps.optMultiply(t, that)

    def batches(batchSize: Int): Iterator[Tensor[T]] = TensorOps.batches(t, batchSize)
    def equalRows(that: Tensor[T]): Int = TensorOps.equalRows(t, that)
    def clipInRange(min: T, max: T): Tensor[T] = TensorOps.clipInRange(t, min, max)    
    def :**(to: Int): Tensor[T] = TensorOps.pow(t, to)
    def sqr: Tensor[T] = TensorOps.pow(t, 2)    
    def sqrt: Tensor[T] = TensorOps.sqrt(t)
    def zero: Tensor[T] = TensorOps.zero(t)
    def argMax: Tensor[T] = TensorOps.argMax(t)
    def outer(that: Tensor[T]) = TensorOps.outer(t, that)
    def flatten: Tensor[T] = TensorOps.flatten(t)
    def diag: Tensor[T] = TensorOps.diag(t)
    def sumRows: Tensor[T] = TensorOps.sumRows(t)    
    def sumCols: Tensor[T] = TensorOps.sumCols(t)
    def max: T = TensorOps.max(t)
    def reshape(shape: List[Int]): Tensor[T] = TensorOps.reshape(t, shape)
 
  extension [T: ClassTag: Fractional](t: Tensor[T])
    def clipByNorm(norm: T): Tensor[T] = TensorOps.clipByNorm(t, norm)
    def /(that: Tensor[T]): Tensor[T] = TensorOps.div(t, that)
    def :/(that: T): Tensor[T] = TensorOps.div(t, Tensor0D(that))

  extension [T: ClassTag](t: Tensor[T])
    def T: Tensor[T] = TensorOps.transpose(t)    
    def map[U: ClassTag](f: T => U): Tensor[U] = TensorOps.map[T, U](t, f)          
    def mapRow[U: ClassTag](f: Array[T] => Array[U]): Tensor[U] = TensorOps.mapRow[T, U](t, f)          

object ops extends genOps:
  extension [T: ClassTag](t: Tensor2D[T])
    def col(i: Int): Tensor1D[T] = Tensor1D(TensorOps.col(t.data, i))
    def T: Tensor2D[T] = TensorOps.transpose(t).asInstanceOf[Tensor2D[T]]
    def slice(
      rows: Option[(Int, Int)],
      cols: Option[(Int, Int)]
    ): Tensor2D[T] =
      Tensor2D(t.data.slice(rows, cols))
    def slice(
      rows: (Int, Int),
      cols: (Int, Int)
    ): Tensor2D[T] =
      Tensor2D(TensorOps.sliceArr(t.data, rows, cols))      
    
  extension [T: ClassTag: Numeric](t: Tensor2D[T])
    def |*|(that: Tensor2D[T]): Tensor2D[T] = TensorOps.multiply(t, that).asInstanceOf[Tensor2D[T]]
    def +(that: Tensor[T]): Tensor2D[T] = TensorOps.plus(t, that).asInstanceOf[Tensor2D[T]]    

  extension [T: ClassTag](t: Tensor[T])
    def as0D: Tensor0D[T] = TensorOps.as0D(t)    
    def as1D: Tensor1D[T] = TensorOps.as1D(t)    
    def as2D: Tensor2D[T] = TensorOps.as2D(t)    
    def as3D: Tensor3D[T] = TensorOps.as3D(t)    
    def as4D: Tensor4D[T] = TensorOps.as4D(t)    

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
    def +(that: Tensor[T]): Tensor[T] = TensorOps.plus(Tensor0D(t), that)    
  
  extension [T: ClassTag: Numeric](a: Array[T])
    def as1D: Tensor1D[T] = Tensor1D(a)
    def as2D: Tensor2D[T] = Tensor2D(a)

  extension [T: ClassTag](a: Array[T])(using n: Numeric[T])
    def +(b: Array[T]): Array[T] = a.zip(b).map(n.plus)
  
  extension [T: ClassTag: Numeric](a: Array[Array[T]])
    def as2D: Tensor2D[T] = Tensor2D(a)
    def sum: T = a.map(_.sum).sum
  
  extension [T: ClassTag: Numeric](a: IndexedSeq[IndexedSeq[T]])
    def as2D: Tensor2D[T] = Tensor2D(a.map(_.toArray).toArray)
  
  extension [T: ClassTag: Numeric](a: Array[Tensor2D[T]])
    def as3D: Tensor3D[T] = Tensor3D(a:_*)
  
  extension [T: ClassTag: Numeric](a: Array[Array[Array[T]]])
    def as3D: Tensor3D[T] = Tensor3D(a)
  
  extension [T: ClassTag: Numeric](a: Array[Array[Array[Array[T]]]])
    def as4D: Tensor4D[T] = Tensor4D(a)
  
  extension [T: ClassTag: Numeric](a: Array[Array[Tensor2D[T]]])
    def as4D: Tensor4D[T] = Tensor4D(a.map(_.map(_.data)))

  extension [T: ClassTag](a: Array[Array[T]])
    def col(i: Int): Array[T] = TensorOps.col(a, i)
    def slice(          
          rows: Option[(Int, Int)] = None,
          cols: Option[(Int, Int)] = None
      ): Array[Array[T]] = TensorOps.slice(a, rows, cols)

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
        checkShapeEquality(a, b)        
        Tensor1D(data.zip(data2).map(_ - _))
      case (Tensor2D(data), Tensor2D(data2)) =>        
        checkShapeEquality(a, b)    
        Tensor2D(matrixMinusMatrix(data, data2))
      case (Tensor2D(data), Tensor0D(data2)) => // broadcasting
        Tensor2D(data.map(_.map(_ - data2)))
      case (Tensor0D(data), Tensor2D(data2)) => // broadcasting
        Tensor2D(data2.map(_.map(v => data - v)))
      case (Tensor1D(data), Tensor0D(data2)) => // broadcasting
        Tensor1D(data.map(_ - data2))
      case (t1 @ Tensor2D(_), t2 @ Tensor1D(_)) => // broadcasting
        matrixMinusVector(t1, t2)
      case (Tensor4D(data), Tensor4D(data2)) =>
        checkShapeEquality(a, b)
        val res = data.zip(data2).map { (cubes, cubes2) =>
          cubes.zip(cubes2).map { (mat1, mat2) =>
            matrixMinusMatrix(mat1, mat2)
          }
        }
        Tensor4D(res)
      case (t1, t2) => 
        sys.error(s"Not implemented for\n$t1 and\n$t2")

  private def matrixMinusMatrix[T: ClassTag: Numeric](a: Array[Array[T]], b: Array[Array[T]]): Array[Array[T]] =
    val rows = a.length
    val cols = a.headOption.map(_.length).getOrElse(0)
    val res = Array.ofDim[T](rows, cols)

    for i <- a.indices do
      for j <- 0 until cols do
        res(i)(j) = a(i)(j) - b(i)(j)
    res
        
  private def matrixMinusVector[T: Numeric: ClassTag](
      matrix: Tensor2D[T],
      vector: Tensor1D[T]
  ) =
    val cols = matrix.shape2D._2
    assert(
      cols == vector.length,
      s"trailing axis must have the same size, $cols != ${vector.length}"
    )
    val res = matrix.data.map(_.zip(vector.data).map{(a, b) => a - b })
    Tensor2D(res)

  private def checkShapeEquality[T](a: Tensor[T],  b: Tensor[T]) = 
    assert(a.shape == b.shape, s"Tensors must have the same shape: ${a.shape} != ${b.shape}")

  def optPlus[T: ClassTag: Numeric](a: Tensor[T], b: Option[Tensor[T]]): Tensor[T] =
    b.fold(a)(t => plus(a, t))

  def plus[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match
      // broadcasting
      case (Tensor2D(data), Tensor0D(data2)) =>
        Tensor2D(data.map(_.map(_ + data2)))
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
      case (Tensor4D(data), Tensor0D(data2)) =>
        Tensor4D(data.map(_.map(_.map(_.map(_ + data2)))))

      case (Tensor1D(data), Tensor1D(data2)) =>
        checkShapeEquality(a, b)        
        val res = Array.ofDim(data.length)
        for i <- 0 until data.length do 
          res(i) = data(i) + data2(i) 
        Tensor1D(res)
      case (t1 @ Tensor2D(data), Tensor2D(data2)) =>
        checkShapeEquality(a, b)        
        val res = matrixPlusMatrix(data, data2)
        Tensor2D(res)
      case (Tensor4D(data), Tensor4D(data2)) =>
        checkShapeEquality(a, b)
        val res = data.zip(data2).map { (cubes1, cubes2) =>
          cubes1.zip(cubes2).map { (mat1, mat2) =>
            matrixPlusMatrix(mat1, mat2)
          }
        }
        Tensor4D(res)      
      case (Tensor0D(data), Tensor0D(data2)) =>
        Tensor0D(data + data2)      
      case _ => notImplementedError(a :: b:: Nil)

  private def matrixPlusMatrix[T: ClassTag: Numeric](a: Array[Array[T]], b: Array[Array[T]]): Array[Array[T]] = 
    val (rows, cols) = (a.length, a.head.length)
    val res = Array.ofDim(rows, cols)
    for i <- 0 until rows do
      for j <- 0 until cols do
        res(i)(j) = a(i)(j) + b(i)(j)
    res
    
  private def notImplementedError[T](ts: List[Tensor[T]]) =
    sys.error(s"Not implemented for: ${ts.mkString("\n")}")

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

  def optMul[T: ClassTag: Numeric](a: Tensor[T], b: Option[Tensor[T]]): Tensor[T] =
    b.fold(a)(t => mul(a, t))     

  def mul[T: ClassTag: Numeric](a: Tensor[T], b: Tensor[T]): Tensor[T] =
    (a, b) match
      case (Tensor0D(data), t) =>
        scalarMul(t, data)
      case (t, Tensor0D(data)) =>
        scalarMul(t, data)
      case (Tensor1D(data), Tensor2D(data2)) =>
        Tensor2D(matMul(Array(data), data2))
      case (Tensor2D(data), Tensor1D(data2)) =>
        Tensor2D(matMul(data, asColumn(data2)))
      case (Tensor1D(data), Tensor1D(data2)) =>
        Tensor0D(matMul(Array(data), asColumn(data2)).head.head)
      case (Tensor2D(data), Tensor2D(data2)) =>
        Tensor2D(matMul(data, data2))
      case _ => notImplementedError(a :: b :: Nil)

  private def asColumn[T: ClassTag](a: Array[T]) = a.map(Array(_))

  def map[T: ClassTag, U: ClassTag](t: Tensor[T], f: T => U): Tensor[U] =
    t match
      case Tensor0D(data) => Tensor0D(f(data))
      case Tensor1D(data) => Tensor1D(data.map(f))
      case Tensor2D(data) => Tensor2D(data.map(_.map(f)))
      case Tensor3D(data) => Tensor3D(data.map(_.map(_.map(f))))
      case Tensor4D(data) => Tensor4D(data.map(_.map(_.map(_.map(f)))))      
  
  def mapRow[T: ClassTag, U: ClassTag](t: Tensor[T], f: Array[T] => Array[U]): Tensor[U] =
    t match
      case Tensor0D(data) => Tensor0D(f(Array(data)).head)
      case Tensor1D(data) => Tensor1D(f(data))
      case Tensor2D(data) => Tensor2D(data.map(f))
      case _ => notImplementedError(t :: Nil)
  
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
      case Tensor4D(data) => Tensor4D(data.map(_.map(_.map(_.map(_ * scalar)))))
      case _ => notImplementedError(t :: Nil)

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

  def as0D[T: ClassTag](t: Tensor[T]): Tensor0D[T] =
    t match
      case Tensor0D(data)   => Tensor0D(data)
      case t1 @ Tensor1D(data) => Tensor0D(data.head)
      case Tensor2D(data)   => Tensor0D(data.head.head)
      case _ => notImplementedError(t :: Nil)
  
  def as1D[T: ClassTag](t: Tensor[T]): Tensor1D[T] =
    t match
      case Tensor0D(data)   => Tensor1D(data)
      case t1 @ Tensor1D(_) => t1
      case Tensor2D(data)   => Tensor1D(data.flatten)
      case _ => notImplementedError(t :: Nil)

  def as2D[T: ClassTag](t: Tensor[T]): Tensor2D[T] =
    t match
      case Tensor0D(data)   => Tensor2D(Array(Array(data)))
      case Tensor1D(data)   => Tensor2D(data.map(Array(_)))
      case t1 @ Tensor2D(_) => t1
      case t1 @ Tensor4D(data) => Tensor2D(data.map(_.map(_.flatten).flatten))
      case _ => notImplementedError(t :: Nil)
  
  def as3D[T: ClassTag](t: Tensor[T]): Tensor3D[T] =
    t match
      case Tensor0D(data)   => Tensor3D(Array(Array(Array(data))))
      case Tensor2D(data)   => Tensor3D(Array(data))
      case t1 @ Tensor3D(_) => t1
      case _ => notImplementedError(t :: Nil)

  def as4D[T: ClassTag](t: Tensor[T]): Tensor4D[T] =
    t match
      case Tensor0D(data)   => Tensor4D(Array(Array(Array(Array(data)))))
      case Tensor1D(data)   => Tensor4D(Array(Array(data.map(Array(_)))))
      case t2 @ Tensor2D(_) => Tensor4D(Array(Array(t2.data)))
      case t1 @ Tensor4D(_) => t1
      case _ => notImplementedError(t :: Nil)

  def sum[T: Numeric: ClassTag](t: Tensor[T]): T =
    t match
      case Tensor0D(data) => data
      case Tensor1D(data) => data.sum
      case Tensor2D(data) => data.map(_.sum).sum
      case _ => notImplementedError(t :: Nil)
  
  def sumRows[T: Numeric: ClassTag](t: Tensor[T]): Tensor[T] =
    t match
      case Tensor0D(_) => t
      case Tensor1D(_) => t
      case Tensor2D(data) => 
        Tensor1D(data.reduce((a, b) => a.lazyZip(b).map(_ + _).toArray))
      case _ => notImplementedError(t :: Nil)
  
  def sumCols[T: Numeric: ClassTag](t: Tensor[T]): Tensor[T] =
    t match
      case Tensor0D(_) => t
      case Tensor1D(data) => Tensor0D(data.sum)
      case Tensor2D(data) => Tensor2D(data.map(a => Array(a.sum)))
      case _ => notImplementedError(t :: Nil)

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
      case _ => notImplementedError(t :: Nil)

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

  def optMultiply[T: Numeric: ClassTag](
    t1: Tensor[T], t2: Option[Tensor[T]]
  ): Tensor[T] = 
    t2.fold(t1)(a => multiply(t1, a))

  def batches[T: ClassTag: Numeric](
      t: Tensor[T],
      batchSize: Int
  ): Iterator[Tensor[T]] =
    t match
      case Tensor0D(data) => Iterator(t)
      case Tensor1D(data) => data.grouped(batchSize).map(Tensor1D(_))
      case Tensor2D(data) => data.grouped(batchSize).map(Tensor2D(_))
      case Tensor3D(data) => data.grouped(batchSize).map(Tensor3D(_))
      case Tensor4D(data) => data.grouped(batchSize).map(Tensor4D(_))

  def equalRows[T: ClassTag](t1: Tensor[T], t2: Tensor[T]): Int = 
    assert(t1.shape == t2.shape, sys.error(s"Tensors must have the same shape: ${t1.shape} != ${t2.shape}"))
    (t1, t2) match
      case (Tensor0D(data), Tensor0D(data2)) =>         
        if data == data2 then 1 else 0
      case (Tensor1D(data), Tensor1D(data2)) => 
        data.zip(data2).count(_ == _)        
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

    map(t, clipValue)    
  
  def clipByNorm[T: ClassTag](t: Tensor[T], norm: T)(using n: Fractional[T]): Tensor[T] = 
    val l2norm = castFromTo[Double, T](math.sqrt(castFromTo[T, Double](sum(pow(t, 2)))))
    if l2norm > norm then
      map(t, v => n.times(v, norm) / l2norm)  
    else t
  
  def div[T: ClassTag: Fractional](t1: Tensor[T], t2: Tensor[T]): Tensor[T] =    
    (t1, t2) match
      // broadcasting
      case (Tensor2D(data), Tensor0D(data2)) => Tensor2D(data.map(_.map(_ / data2)))
      case (Tensor1D(data), Tensor0D(data2)) => Tensor1D(data.map(_ / data2))
      case (Tensor4D(data), Tensor0D(data2)) => Tensor4D(data.map(_.map(_.map(_.map(_ / data2)))))
      
      case (Tensor0D(data), Tensor0D(data2)) => Tensor0D(data / data2)
      case (Tensor1D(data), Tensor1D(data2)) => Tensor1D(data.zip(data2).map(_ /_))
      case (Tensor2D(data), Tensor2D(data2)) =>        
        Tensor2D(matrixDivMatrix(data, data2))
      case (Tensor4D(data), Tensor4D(data2)) =>
        val res = data.zip(data2).map { (cubes1, cubes2) =>
          cubes1.zip(cubes2).map { (mat1, mat2) => 
            matrixDivMatrix(mat1, mat2)
          }
        }
        Tensor4D(res)
      case _ => notImplementedError(t1 :: t2 :: Nil)
  
  private def matrixDivMatrix[T: ClassTag: Fractional](a: Array[Array[T]], b: Array[Array[T]]): Array[Array[T]] =
    a.zip(b).map((a, b) => a.zip(b).map(_ / _))

  def sqrt[T: ClassTag: Numeric](t: Tensor[T]): Tensor[T] = 
    map(t, v => castFromTo[Double, T](math.sqrt(castFromTo[T, Double](v))))

  def pow[T: ClassTag](t: Tensor[T], to: Int)(using n: Numeric[T]): Tensor[T] =
    def powValue(v: T) =
      val res = math.pow(n.toDouble(v), to)
      castFromTo[Double, T](res)
    def powArray(a: Array[T]) =
      a.map(powValue)
    def powMatrix(a: Array[Array[T]]) =
      a.map(_.map(powValue))

    t match
      case Tensor0D(data) => Tensor0D(powValue(data))
      case Tensor1D(data) => Tensor1D(powArray(data))
      case Tensor2D(data) => Tensor2D(powMatrix(data))
      case Tensor4D(data) => Tensor4D(data.map(_.map(powMatrix)))
      case _ => notImplementedError(t :: Nil)
  
  def zero[T: ClassTag](t: Tensor[T])(using n: Numeric[T]): Tensor[T] =
    t match 
      case Tensor0D(_) => Tensor0D(n.zero)
      case Tensor1D(data) => Tensor1D(Array.fill(data.length)(n.zero))
      case t1 @ Tensor2D(_) => 
        val (rows, cols) = t1.shape2D
        Tensor2D(Array.fill(rows, cols)(n.zero))
      case t1 @ Tensor3D(_) =>
        val (cubes, rows, cols) = t1.shape3D
        Tensor3D(Array.fill(cubes, rows, cols)(n.zero))
      case t1 @ Tensor4D(_) =>
        val (tensors, cubes, rows, cols) = t1.shape4D
        Tensor4D(Array.fill(tensors, cubes, rows, cols)(n.zero))

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

  def sliceArr[T: ClassTag](
      data: Array[Array[T]],
      rows: (Int, Int),
      cols: (Int, Int)
  ): Array[Array[T]] = 
    sliceArr(data, rows).map(a =>
      sliceArr(a, cols)
    )

  def sliceArr[T](
      data: Array[T],
      range: (Int, Int)
  ): Array[T] =
    val (l, r) = range
    val from = if l < 0 then data.length + l else l
    val to = if r < 0 then data.length + r else if r == 0 then data.length else r
    data.slice(from, to)

  // returns max index per array
  // for 2D Tensor: returns an array of indices where every element is a max index for a specific row  
  def argMax[T: ClassTag](t: Tensor[T])(using n: Numeric[T]) =
    def maxIndex(a: Array[T]) = 
      n.fromInt(a.indices.maxBy(a))

    t match
      case Tensor2D(data) => Tensor1D(data.map(maxIndex))
      case Tensor1D(data) => Tensor0D(maxIndex(data))
      case Tensor0D(_) => t
      case _ => notImplementedError(t :: Nil)

  def outer[T: ClassTag: Numeric](t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    def product(a: Array[T], b: Array[T]) =
      val res = Array.ofDim(a.length, b.length)
      for i <- 0 until a.length do
        for j <- 0 until b.length do
          res(i)(j) = a(i) * b(j)
      res

    (t1, t2) match
      case (Tensor0D(d), Tensor0D(d2)) => Tensor0D(d * d2)
      case (Tensor0D(d), _) => scalarMul(t2, d)  
      case (Tensor1D(d), Tensor0D(d2)) => scalarMul(t1, d2)
      case (Tensor1D(d), Tensor1D(d2)) => Tensor2D(product(d, d2))
      case (Tensor1D(d), Tensor2D(d2)) => Tensor2D(product(d, d2.flatten))
      case (Tensor2D(d), Tensor0D(d2)) => scalarMul(t1, d2)
      case (Tensor2D(d), Tensor1D(d2)) => Tensor2D(product(d.flatten, d2))
      case (Tensor2D(d), Tensor2D(d2)) => Tensor2D(product(d.flatten, d2.flatten))
      case _ => notImplementedError(t1 :: t2 :: Nil)

  def flatten[T: ClassTag](t: Tensor[T]): Tensor[T] = 
    t match
      case Tensor0D(_) => t
      case Tensor1D(_) => t
      case Tensor2D(data) => Tensor1D(data.flatten)
      case _ => notImplementedError(t :: Nil)
  
  def diag[T: ClassTag](t: Tensor[T])(using n: Numeric[T]): Tensor[T] =
    t match
      case Tensor0D(_) => t
      case Tensor1D(d) => 
        val res = Array.ofDim(d.length, d.length)
        for i <- 0 until d.length do
          for j <- 0 until d.length do
            res(i)(j) = if i == j then d(i) else n.zero
        Tensor2D(res)
      case t2 @ Tensor2D(d) =>
        val size = t2.shape.min
        val res = Array.ofDim(size)
        for i <- 0 until size do
          for j <- 0 until size if i == j do
            res(i) = d(i)(j)
        Tensor1D(res)
      case _ => notImplementedError(t :: Nil)

  def max[T: ClassTag: Numeric](t: Tensor[T]): T =
    t match
      case Tensor0D(d) => d
      case Tensor1D(d) => d.max
      case Tensor2D(d) => d.map(_.max).max
      case Tensor3D(d) => d.map(_.map(_.max).max).max
      case Tensor4D(d) => d.map(_.map(_.map(_.max).max).max).max

  def reshape[T: ClassTag: Numeric](t: Tensor[T], shape: List[Int]): Tensor[T] =
    shape match
      case cubes :: rows :: cols :: _ => t match
        case Tensor2D(data) =>
          Tensor4D(data.flatMap(_.grouped(cols).toArray.grouped(rows).toArray.grouped(cubes).toArray))
        case _ => t    
      case _ => t

      
