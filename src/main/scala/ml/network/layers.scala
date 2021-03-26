package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

final case class Gradient[T](
  delta: Tensor[T], 
  w: Option[Tensor[T]] = None, 
  b: Option[Tensor[T]] = None
)

trait Layer[T]:
  val f: ActivationFunc[T] = ActivationFuncApi.linear  
  val shape: List[Int]  

  def init[U, V](prevShape: List[Int]): Layer[T] = this
  def apply(x: Tensor[T]): Activation[T]
  def backward(a: Activation[T], prevDelta: Tensor[T], preWeight: Option[Tensor[T]]): Gradient[T]

  override def toString() = 
    s"\nf = ${f.name},\nshape = $shape"

trait Optimizable[T] extends Layer[T]:
  val w: Option[Tensor[T]]
  val b: Option[Tensor[T]]
  val optimizerParams: Option[OptimizerParams[T]]
  
  def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T]

  //TODO: bGradient model as Tensor[T] | T
  def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T]  
  
  override def toString() = 
    s"(${super.toString},\nweight = $w,\nbias = $b)"

case class Dense[T: ClassTag: Numeric](
    override val f: ActivationFunc[T] = ActivationFuncApi.linear[T],
    units: Int = 1,
    shape: List[Int] = Nil,
    w: Option[Tensor[T]] = None,
    b: Option[Tensor[T]] = None,
    optimizerParams: Option[OptimizerParams[T]] = None
) extends Optimizable[T]:

  override def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Layer[T] =
    val inputs = prevShape.drop(1).reduce(_ * _)
    val w = initializer.weights(inputs, units)
    val b = initializer.biases(units)
    val optimizerParams = optimizer.init(w, b)
    // println(s"Dense shape: ${List(inputs, units)}")
    copy(w = Some(w), b = Some(b), shape = List(inputs, units), optimizerParams = optimizerParams)

  override def apply(x: Tensor[T]): Activation[T] =    
    val z = x * w + b 
    val a = f(z)
    // println(s"x: $x")
    // println(s"w: $w")
    // println(s"a: $a")
    // println(s"Dense output shape: ${a.shape}")
    Activation(x, z, a)

  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    val updatedW = w.map(_ - wGradient)  
    val updatedB = b.map(_ - bGradient)  
    copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)

  override def backward(a: Activation[T], prevDelta: Tensor[T], prevWeight: Option[Tensor[T]]): Gradient[T] = 
    // println(s"Dense prevDelta: $prevDelta")
    val delta = (prevWeight match 
      case Some(pw) => prevDelta * pw.T
      case None     => prevDelta
    ) |*| f.derivative(a.z)

    val wGradient = Some(a.x.T * delta)
    val bGradient = Some(delta)
    Gradient(delta, wGradient, bGradient)

case class Conv2D[T: ClassTag: Numeric](
    override val f: ActivationFunc[T],
    filterCount: Int = 1,   
    kernel: (Int, Int) = (2, 2), 
    strides: (Int, Int) = (1, 1), 
    shape: List[Int] = Nil, 
    w: Option[Tensor[T]] = None,
    b: Option[Tensor[T]] = None,
    optimizerParams: Option[OptimizerParams[T]] = None
) extends Optimizable[T]:

  override def init[U, V](prevShape: List[Int], initializer: ParamsInitializer[T, V], optimizer: Optimizer[U]): Conv2D[T] =        
    val images :: channels :: height :: width :: _ = prevShape    
    val w = initializer.weights4D(filterCount, channels, kernel._1, kernel._2)
    val b = initializer.biases(filterCount)
    val optimizerParams = optimizer.init(w, b)        
    val rows = (height - kernel._1) / strides._1 + 1
    val cols = (width - kernel._2) / strides._2 + 1
    val shape = List(images, filterCount, rows, cols)    
    copy(w = Some(w), b = Some(b), shape = shape, optimizerParams = optimizerParams)
  
  override def apply(x: Tensor[T]): Activation[T] =
    val z = (w, b) match
      case (Some(w), Some(b)) => forward(kernel._1, strides._1, b, x, w)
      case _ => x // does nothing when one the params is empty    
    val a = f(z)    
    Activation(x, z, a)

  private def forward(kernel: Int, stride: Int, bias: Tensor[T], x: Tensor[T], w: Tensor[T]): Tensor[T] =
    val (images, filters) = (x.as4D, w.as4D)    
    
    def filterImage(image: Array[Array[Array[T]]]) =
      filters.data.zip(bias.as1D.data).map { (f, bias) =>
        val filtered = f.zip(image).map { (fc, ic) =>
          conv(fc.as2D, ic.as2D, kernel, stride)
        }.reduce(_ + _)
        filtered + bias.asT
      }
    
    images.data.map(filterImage).as4D    
  
  private def conv(filterChannel: Tensor2D[T], imageChannel: Tensor2D[T], kernel: Int, stride: Int) =    
    val filtered = 
      for row <- imageRegions(imageChannel, kernel, stride) yield
        for (region, _, _) <- row yield
          (region |*| filterChannel).sum

    filtered.as2D

  // TODO: rewrite via imageRegions
  private def fullConv(filter: Tensor2D[T], loss: Tensor2D[T], kernel: Int, stride: Int, rows: Int, cols: Int) = 
    val out = Array.ofDim(rows, cols)
    
    for i <- 0 until kernel do
      for j <- 0 until kernel do                
        val delta = filter * loss.data(i)(j)
        val (x, y) = (i * stride, j * stride)
        
        val iter = delta.as2D.data.flatten.iterator
        for k <- x until x + kernel do
          for l <- y until y + kernel do            
            out(k)(l) += iter.next

    out.as2D

  private def calcGradient(loss: Tensor2D[T], image: Tensor2D[T], kernel: Int, stride: Int) =        
    val grad = 
      for (region, i, j) <- imageRegions(image, kernel, stride).flatten
      yield region * loss.data(i)(j)    

    grad.reduce(_ + _).as2D

  private def imageRegions(image: Tensor2D[T], kernel: Int, stride: Int) =
    val (rows, cols) = image.shape2D    
    for i <- 0 to rows - kernel by stride yield   
      for j <- 0 to cols - kernel by stride yield          
        (image.slice((i, i + kernel), (j, j + kernel)).as2D, i, j)    

  override def backward(a: Activation[T], prevDelta: Tensor[T], preWeight: Option[Tensor[T]]): Gradient[T] =
    println(s"Conv2D prevDelta: $prevDelta")
    (w, b) match 
      case (Some(w), Some(b)) =>
        val prevLoss = prevDelta.as4D
        val x = a.x.as4D                
      
        //val oneLoss = prevLoss.data.head
        val oneImage = x.data.head

        val wGradient = prevLoss.data.map { lossChannels =>
          lossChannels.zip(oneImage).map { (lc, ic) =>          
            calcGradient(lc.as2D, ic.as2D,  kernel._1, strides._1)                    
          }
        }.as4D
        println(s"Conv2D wGradient:$wGradient")
        
        val bGradient = prevLoss.data.map { channels =>
          channels.map { image =>
            image.as2D.sum
          }
        }.as2D

        val (_, _, rows, cols) = x.shape4D
        val delta = w.as4D.data.map { channels =>          
          prevLoss.data.map { lossChannels =>
            val r = lossChannels.zip(channels).map { (lc, fc) =>
              fullConv(fc.as2D, lc.as2D, kernel._1, strides._1, rows, cols)
            }
            r.reduce(_ + _)
          }
        }.as4D

        Gradient(delta, Some(wGradient), Some(bGradient))
      case _ =>    
        Gradient(prevDelta)
  
  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    val updatedW = w.map(_ - wGradient)  
    val updatedB = b.map(_ - bGradient)  
    copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)    

case class MaxPool[T: ClassTag: Numeric](
    pool: (Int, Int) = (2, 2), 
    strides: (Int, Int) = (1, 1),     
    shape: List[Int] = Nil,
    shape2D: (Int, Int) = (0, 0),
    padding: Boolean = true
) extends Layer[T]: 

  override def init[U, V](prevShape: List[Int]): Layer[T] =
    val (a :: b :: rows :: cols :: _) = prevShape
    val pad = if padding then 1 else 0
    val height = (rows - pool._1 + pad) / strides._1 + 1
    val width = (cols - pool._2 + pad) / strides._2 + 1
    val shape = List(a, b, height, width)
    println(s"MaxPool shape: $shape")
    copy(shape = shape, shape2D = (height, width))

  def apply(x: Tensor[T]): Activation[T] =    
    val pooled = x.as4D.data.map(_.map(c => poolMax(c.as2D))).as4D
    println(s"MaxPool output shape: ${pooled.shape}")
    println(s"pooled: ${pooled}")
    Activation(x, pooled, pooled)
  
  private def imageRegions(image: Tensor2D[T], kernel: Int, stride: Int) =
    val (rows, cols) = shape2D
    for i <- 0 until rows by stride yield   
      for j <- 0 until cols by stride yield          
        (image.slice((i, i + kernel), (j, j + kernel)).as2D, i, j)
        
  private def poolMax(image: Tensor2D[T]): Tensor2D[T] =
    val (rows, cols) = shape2D
    val out = Array.ofDim(rows, cols)
    val pooled = 
      for (region, i, j) <- imageRegions(image, pool._1, strides._1).flatten yield            
        out(i)(j) = region.max
    out.as2D

  private def maxIndex(matrix: Tensor2D[T]): (Int, Int) =    
    val maxPerRow = matrix.data.zipWithIndex.map((row, i) => (row.max, i, row.indices.maxBy(row)))
    val (i, j) = maxPerRow.maxBy(_._1).tail    
    (i, j)

  def backward(a: Activation[T], prevDelta: Tensor[T], preWeight: Option[Tensor[T]]): Gradient[T] = 
    println(s"MaxPool prevDelta: $prevDelta")
    val image = a.x.as4D.data
    println(s"MaxPool image: ${a.x}")
    val delta = image.zip(prevDelta.as4D.data).map { (imageChannels, deltaChannels) =>
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
    Gradient(delta.as4D)    

case class Flatten2D[T: ClassTag: Numeric](  
  shape: List[Int] = Nil,
  prevShape: List[Int] = Nil
) extends Layer[T]:

  override def init[U, V](prevShape: List[Int]): Layer[T] =
    val (head :: tail ) = prevShape
    val shape = List(head, tail.reduce(_ * _))
    copy(shape = shape, prevShape = prevShape)

  def apply(x: Tensor[T]): Activation[T] =
    val flat = x.as2D
    println(s"flattened: $flat")
    Activation(x, flat, flat)

  def backward(a: Activation[T], prevDelta: Tensor[T], prevWeight: Option[Tensor[T]]): Gradient[T] =    
    val delta = (prevWeight match 
      case Some(pw) => prevDelta * pw.T
      case None     => prevDelta
    ) //|*| f.derivative(a.z) TODO: is any z multiply required here?

    println(s"Flatten prevDelta: $prevDelta")    
    val (filters :: rows :: cols :: _) = prevShape.drop(1)
    val unflatten = delta.as2D.data
      .flatMap(_.grouped(cols).toArray.grouped(rows).toArray.grouped(filters).toArray) //TODO: replace with some reshape method
      .as4D    
    Gradient(unflatten)