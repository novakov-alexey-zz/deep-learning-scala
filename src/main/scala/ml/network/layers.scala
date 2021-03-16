package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

trait Layer[T]:
  val f: ActivationFunc[T]
  val w: Option[Tensor[T]]
  val b: Option[Tensor[T]]
  val shape: List[Int]
  val optimizerParams: Option[OptimizerParams[T]]

  def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T]

  def apply(x: Tensor[T]): Activation[T]

  def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T]

  override def toString() = 
    s"\n(\nweight = $w,\nbias = $b,\nf = ${f.name},\nshape = $shape)"

case class Dense[T: ClassTag: Numeric](
    f: ActivationFunc[T] = ActivationFuncApi.linear[T],
    units: Int = 1,
    shape: List[Int] = Nil,
    w: Option[Tensor[T]] = None,
    b: Option[Tensor[T]] = None,
    optimizerParams: Option[OptimizerParams[T]] = None
) extends Layer[T]:

  override def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T] =
    val inputs = prevShape.drop(1).reduce(_ * _)
    val w = initializer.weights(inputs,  units)
    val b = initializer.biases(units)
    val optimizerParams = optimizer.init(w, b)
    copy(w = Some(w), b = Some(b), shape = List(inputs, units), optimizerParams = optimizerParams)

  override def apply(x: Tensor[T]): Activation[T] =    
    val z = x * w + b
    val a = f(z)
    Activation(x, z, a)

  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    val updatedW = w.map(_ - wGradient)  
    val updatedB = b.map(_ - bGradient)  
    copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)

case class Conv2D[T: ClassTag: Numeric](
    f: ActivationFunc[T],
    filterCount: Int = 1,   
    kernel: (Int, Int) = (2, 2), 
    strides: (Int, Int) = (1, 1), 
    shape: List[Int] = Nil, 
    w: Option[Tensor[T]] = None,
    b: Option[Tensor[T]] = None,
    optimizerParams: Option[OptimizerParams[T]] = None
) extends Layer[T]:

  override def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T] =    
    val (width, height) = kernel
    val channels = prevShape.drop(1).headOption.getOrElse(0) // take second axis as inputs    
    val w = (0 until channels)
        .map(_ => (0 until filterCount)
        .map(_ => initializer.weights(width,  height)).toArray)
        .toArray.as4D

    val b = initializer.biases(filterCount)
    val optimizerParams = optimizer.init(w, b)    
    val shape = List(channels, filterCount, width, height)
    copy(w = Some(w), b = Some(b), shape = shape, optimizerParams = optimizerParams)

  override def apply(x: Tensor[T]): Activation[T] =
    val z = (w, b) match
      case (Some(weights), Some(biases)) =>
        val f = conv(kernel._1, strides._1, biases)          
        (x, weights).product(axis = 2)(f)
      case _ => x
    
    val a = f(z)
    Activation(x, z, a)

  private def conv(kernel: Int, stride: Int, bias: Tensor[T])(t1: Tensor[T], t2: Tensor[T], position: (List[Int], List[Int])): Tensor[T] =
    val image = t1.as2D
    val filter = t2.as2D
    val (rows, cols) = image.shape2D    
    val ((filterId :: _), _) = position
    val res = ListBuffer.empty[ListBuffer[T]]
    val filterBias = bias.as1D.data(filterId)

    for i <- 0 to rows - kernel by stride do
      val row = ListBuffer.empty[T]
      for j <- 0 to cols - kernel by stride do
        val area = image.slice(Some(i, i + kernel), Some(j, j + kernel))
        row += ((area |*| filter).sum + filterBias).as0D.data
      res += row  

    Tensor2D(res.map(_.toArray).toArray)        

  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    // val updatedW = w.map(_ - wGradient)  
    // val updatedB = b.map(_ - bGradient)  
    // copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)
    ???