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

  override def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Conv2D[T] =    
    val (width, height) = kernel
    val channels = prevShape.drop(1).headOption.getOrElse(0) // take axis '1' as inputs    
    val w = 
      (0 until filterCount)
        .map(_ =>  (0 until channels).toArray
        .map(_ => initializer.weights(width,  height)))
        .toArray.as4D

    val b = initializer.biases(filterCount)
    val optimizerParams = optimizer.init(w, b)    
    val shape = List(filterCount, channels, width, height)
    copy(w = Some(w), b = Some(b), shape = shape, optimizerParams = optimizerParams)

  // forward 
  override def apply(x: Tensor[T]): Activation[T] =
    val z = (w, b) match
      case (Some(weights), Some(biases)) =>
        conv(kernel._1, strides._1, biases)(x, weights)
      case _ => x
    
    val a = f(z)
    Activation(x, z, a)

  private def conv(kernel: Int, stride: Int, bias: Tensor[T])(t1: Tensor[T], t2: Tensor[T]): Tensor[T] =
    val image = t1.as3D
    val filter = t2.as4D
    val (channels, rows, cols) = image.shape3D    

    val res = filter.data.zip(bias.as1D.data).map { (f, bias) =>      
      val channels = f.zip(image.data).map { (fc, ic) =>
        filterChannel(fc.as2D, ic.as2D, kernel, stride, rows, cols)
      }.reduce(_ + _)
      channels.map(_ + bias).as2D
    }

    Tensor3D(res: _*)        
  
  private def filterChannel(filterChannel: Tensor2D[T], imageChannel: Tensor2D[T], kernel: Int, stride: Int, rows: Int, cols: Int) = 
    val filtered = ListBuffer[Array[T]]()

    for i <- 0 to rows - kernel by stride do
      val row = ListBuffer.empty[T]
      for j <- 0 to cols - kernel by stride do
        val area = imageChannel.slice(Some(i, i + kernel), Some(j, j + kernel)).as2D
        row += (filterChannel |*| area).sum
      filtered += row.toArray  

    filtered.toArray.as2D

  // backward  
  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    // val updatedW = w.map(_ - wGradient)  
    // val updatedB = b.map(_ - bGradient)  
    // copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)
    ???