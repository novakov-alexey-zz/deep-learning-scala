package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

trait Layer[T]:
  val f: ActivationFunc[T] = ActivationFuncApi.linear
  val w: Option[Tensor[T]]
  val b: Option[Tensor[T]]
  val shape: List[Int]
  val optimizerParams: Option[OptimizerParams[T]]

  def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T]

  def apply(x: Tensor[T]): Activation[T]

  //TODO: bGradient model as Tensor[T] | T
  def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T]

  def backward(a: Activation[T], prevDelta: Tensor[T], preWeight: Option[Tensor[T]]): (Option[Tensor[T]], Option[Tensor[T]], Tensor[T])

  override def toString() = 
    s"\n(\nweight = $w,\nbias = $b,\nf = ${f.name},\nshape = $shape)"

case class Dense[T: ClassTag: Numeric](
    override val f: ActivationFunc[T] = ActivationFuncApi.linear[T],
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

  override def backward(a: Activation[T], prevDelta: Tensor[T], prevWeight: Option[Tensor[T]]): 
    (Option[Tensor[T]], Option[Tensor[T]], Tensor[T]) = 
    val delta = (prevWeight match 
      case Some(pw) => prevDelta * pw.T
      case None     => prevDelta
    ) |*| f.derivative(a.z)

    val wGradient = Some(a.x.T * delta)
    val bGradient = Some(delta)
    (wGradient, bGradient, delta)

case class Conv2D[T: ClassTag: Numeric](
    override val f: ActivationFunc[T],
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
    val w = initializer.weights4D(filterCount, channels, width, height)
    val b = initializer.biases(filterCount)
    val optimizerParams = optimizer.init(w, b)    
    val shape = List(filterCount, channels, width, height)
    copy(w = Some(w), b = Some(b), shape = shape, optimizerParams = optimizerParams)
  
  override def apply(x: Tensor[T]): Activation[T] =
    val z = (w, b) match
      case (Some(weights), Some(biases)) => forward(kernel._1, strides._1, biases)(x, weights)
      case _ => x // does nothing when one the params is empty
    val a = f(z)
    Activation(x, z, a)

  private def forward(kernel: Int, stride: Int, bias: Tensor[T])(x: Tensor[T], w: Tensor[T]): Tensor[T] =
    val (images, filters) = (x.as4D, w.as4D)    
    
    def filterImage(image: Array[Array[Array[T]]]) =
      filters.data.zip(bias.as1D.data).map { (f, bias) =>
        val channels = f.zip(image).map { (fc, ic) =>
          conv(fc.as2D, ic.as2D, kernel, stride)
        }.reduce(_ + _)
        channels.map(_ + bias).as2D
      }
    
    images.data.map(filterImage).as4D    
  
  private def conv(filterChannel: Tensor2D[T], imageChannel: Tensor2D[T], kernel: Int, stride: Int) = 
    val filtered = ListBuffer[Array[T]]()
    val (rows, cols) = imageChannel.shape2D

    for i <- 0 to rows - kernel by stride do
      val row = ListBuffer.empty[T]
      for j <- 0 to cols - kernel by stride do
        val area = imageChannel.slice((i, i + kernel), (j, j + kernel)).as2D
        row += (area |*| filterChannel).sum
      filtered += row.toArray  

    filtered.toArray.as2D

  private def fullConv(filter: Tensor2D[T], loss: Tensor2D[T], kernel: Int, stride: Int, rows: Int, cols: Int) = 
    val out = Array.ofDim(rows, cols)
    
    for i <- 0 until kernel do      
      for j <- 0 until kernel do        
        val lossCell = loss.data(i)(j)
        val delta = filter.map(_ * lossCell).as2D
        val (x, y) = (i * stride, j * stride)
        
        val iter = delta.data.flatten.iterator
        for k <- x until x + kernel do
          for l <- y until y + kernel do            
            out(k)(l) += iter.next

    out.as2D

  private def calcGradient(loss: Tensor2D[T], image: Tensor2D[T], kernel: Int, stride: Int) = 
    val grad = ListBuffer[Tensor2D[T]]()
    
    for i <- 0 until kernel do      
      for j <- 0 until kernel do
        val lossCell = loss.data(i)(j)        
        val (x, y) = (i * stride, j * stride)        
        val area = image.slice((x, x + kernel), (y, y + kernel)).as2D                
        grad += area.map(_ * lossCell).as2D

    grad.reduce(_ + _).as2D

  override def backward(a: Activation[T], prevDelta: Tensor[T], preWeight: Option[Tensor[T]]): 
    (Option[Tensor[T]], Option[Tensor[T]], Tensor[T]) =
    (w, b) match 
      case (Some(w), Some(b)) =>
        val prevLoss = prevDelta.as4D
        val x = a.x.as4D        
        val (_, _, rows, cols) = x.shape4D
      
        val wGradient = x.data.map { image =>
          prevLoss.data.map { lossChannels =>
            val channels = lossChannels.zip(image).map { (lc, ic) =>
              calcGradient(lc.as2D, ic.as2D, kernel._1, strides._1)
            }.reduce(_ + _)
            channels
          }
        }.as4D
            
        val delta = w.as4D.data.map { channels =>          
          prevLoss.data.map { lossChannels =>
            val r = lossChannels.zip(channels).map { (lc, fc) =>
              fullConv(fc.as2D, lc.as2D, kernel._1, strides._1, rows, cols)
            }
            r.reduce(_ + _)
          }
        }.as4D

        (Some(wGradient), Some(wGradient), delta) //TODO: calc b grad
      case _ =>    
        (None, None, prevDelta)
  
  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    val updatedW = w.map(_ - wGradient)  
    val updatedB = b.map(_ - bGradient)  
    copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)    

case class MaxPool[T: ClassTag: Numeric](
    pool: (Int, Int) = (2, 2), 
    strides: (Int, Int) = (1, 1),     
    shape: List[Int] = Nil, 
    w: Option[Tensor[T]] = None,
    b: Option[Tensor[T]] = None,
    optimizerParams: Option[OptimizerParams[T]] = None
) extends Layer[T]: 

  def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T] =
    copy(shape = prevShape)

  def apply(x: Tensor[T]): Activation[T] =    
    val pooled = x.as4D.data.map(_.map(c => poolMax(c.as2D))).as4D
    Activation(x, pooled, pooled)
  
  private def poolMax(image: Tensor2D[T]): Tensor2D[T] =
    val (rows, cols) = image.shape2D    
    (0 to rows - pool._1 by strides._1).map { i =>
      (0 to cols - pool._2 by strides._2).map { j =>
        image.slice((i, i + pool._1), (j, j + pool._2)).max
      }
    }.map(_.toArray).toArray.as2D    

  def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] = ???

  private def maxIndex(matrix: Tensor2D[T]): (Int, Int) =    
    val maxPerRow = matrix.data.zipWithIndex.map((row, i) => (row.max, i, row.indices.maxBy(row)))
    val max = maxPerRow.maxBy(_._1)
    (max._2, max._3)

  def backward(a: Activation[T], prevDelta: Tensor[T], preWeight: Option[Tensor[T]]): (Option[Tensor[T]], Option[Tensor[T]], Tensor[T]) = 
    val delta = a.x.as4D.data.zip(prevDelta.as4D.data).map { (imageChannels, deltaChannels) =>
      imageChannels.zip(deltaChannels).map { (ic, dc) =>
        val image = ic.as2D        
        val (rows, cols) = dc.as2D.shape2D
        val out = image.zero.as2D.data

        for i <- 0 until rows do
          for j <- 0 until cols do
            val (x, y) = (i * strides._1, j * strides._2)        
            val area = image.slice((x, x + pool._1), (y, y + pool._2))
            val (a, b) = maxIndex(area)
            out(x + a)(y + b) = dc(i)(j)
        
        out
      }
    }
    (None, None, delta.as4D)    