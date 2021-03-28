package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._
import optimizers.given_Optimizer_Adam as adam

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._
import scala.collection.mutable.ListBuffer

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Conv2DTest extends AnyFlatSpec with Matchers {
  def testActivation[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:
      override def apply(x: Tensor[T]): Tensor[T] = x.map(_ + n.one)
      override def derivative(x: Tensor[T]): Tensor[T] = apply(x)
      override val name = "test"

  given testInit[T: ClassTag](using n: Numeric[T]): ParamsInitializer[T, RandomUniform] with    
    def gen: T = n.one

    override def weights(rows: Int, cols: Int): Tensor2D[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen)))

    override def biases(length: Int): Tensor1D[T] = 
      inits.zeros(length)
      
  val image1 = Tensor3D(Array(
      Array(
        Array(1d, 2, 3, 3), 
        Array(2d, 3, 4, 3), 
        Array(5d, 6, 7, 3)
      ),
      Array(
        Array(1d, 2, 3, 1), 
        Array(2d, 3, 4, 1), 
        Array(5d, 6, 7, 1)
      ),
      Array(
        Array(1d, 2, 3, 2), 
        Array(2d, 3, 4, 2), 
        Array(5d, 6, 7, 2)
      )
    ))

  val image2 = Tensor3D(Array(
      Array(
        Array(1d, 2, 3, 1), 
        Array(2d, 3, 4, 1), 
        Array(5d, 6, 7, 1)
      ),
      Array(
        Array(1d, 2, 3, 2), 
        Array(2d, 3, 4, 2), 
        Array(5d, 6, 7, 2)
      ),
      Array(
        Array(1d, 2, 3, 3), 
        Array(2d, 3, 4, 3), 
        Array(5d, 6, 7, 3)
      )
    ))    
  
  val images = Tensor4D(image1, image2)

  it should "do forward propagation" in {
    // given            
    val inputShape = images.shape4D

    val layer = Conv2D[Double](
      f = testActivation,
      filterCount = 3,
      kernel = (2, 2),
      strides = (1, 1)
    ).init(inputShape.toList, testInit, adam)    
    
    // when
    val activation = layer(images)    
    val (imageCount, inputChannels, width, height) = inputShape

    // then
    activation.z.shape should ===(List(imageCount, layer.filterCount, 2, 3))

    val w = layer.w.getOrElse(fail("Weight must not be empty"))
    val b = layer.b.getOrElse(fail("Bias must not be empty"))  

    def applyFilter[T: ClassTag: Numeric](filter: Array[Array[T]], window: Array[Array[T]]): T =
      filter.zip(window).map((a, b) => a.zip(b).map(_ * _).sum).sum

    def filterChannel[T: Numeric: ClassTag](channel: Array[Array[T]], filter: Array[Array[T]]) = 
      val rows = ListBuffer[Array[T]]()

      for i <- 0 to width - layer.kernel._1 by layer.strides._1 do  
        val img = channel.drop(i).take(layer.kernel._1)
        val row = ListBuffer.empty[T]

        for j <- 0 to height - layer.kernel._1 by layer.strides._2 do           
          val window = img.map(_.drop(j).take(layer.kernel._2))                                
          row += applyFilter(filter, window)            

        rows += row.toArray  
      rows.toArray

    def filterChannels[T: ClassTag : Numeric](filters: Tensor4D[T], images: Tensor4D[T]) =
      images.data.map { image => 
        filters.data.map { channels => 
          channels.zip(image).map { (fc, ic) =>
            filterChannel(ic, fc).as2D
          }.reduce(_ + _)       
        }
      }
      
    val expectedActivities = filterChannels(w.as4D, images).as4D
            
    val layerActivity = activation.z.as4D.data    
    layerActivity.zip(expectedActivities.data).foreach { (actual, expected) =>      
      actual should ===(expected)      
    }
    
    val expectedActivation = layer.f(expectedActivities)
    activation.a.as4D.data sameElements expectedActivation.as4D.data    
  }

  it should "do backward propagation from max pooling layer" in {     
    // given              
    val inputShape = images.shape4D    
    val convLayer = Conv2D[Double](
      f = testActivation,
      filterCount = 3,
      kernel = (2, 2),
      strides = (1, 1)
    ).init(inputShape.toList, testInit, adam)

    // when
    val a = convLayer(images)        
    val poolingLayer = MaxPool[Double](padding = false).init(convLayer.shape)
    val pooled = poolingLayer(a.a)

    val maxPoolDelta = Array.fill(images.length)(
      Array(
        Array(
          Array(1d, 2)          
        ),
        Array(
          Array(7d, 1)          
        ),
        Array(
          Array(4d, 8)
        )
      )
    )    
    val Gradient(convDelta, _, _) = poolingLayer.backward(pooled, maxPoolDelta.as4D, None)
    val Gradient(delta, Some(wGrad), Some(bGrad)) = convLayer.backward(a, convDelta, None)    
    val Some(weightsShape) = convLayer.w.map(_.shape)
    
    // then
    weightsShape should ===(wGrad.shape)
    val expectedConvGrad = Tensor4D(Array(      
      Array.fill(3)(Array(
          Array(6d, 8),
          Array(12d,14)
        )), 
      Array.fill(3)(Array(
          Array(3.0,4.0),
          Array(6.0,7.0)
        )), 
      Array.fill(3)(Array(
          Array(24.0,32.0),
          Array(48.0,56.0)
      ))
    )).map(_ * images.length).as4D.data    

    wGrad.as4D.data should ===(expectedConvGrad)        
    val expectedDelta = Array.fill(2)(         
        Array.fill(3)(
          Array(
            Array(0.0, 0.0, 0.0, 0.0), 
            Array(0.0, 11.0, 11.0, 0.0), 
            Array(0.0, 11.0, 11.0, 0.0)
          )
        ))    
    delta.as4D.data should===(expectedDelta)    

    bGrad.shape should ===(List(3))
    val expectedBias = Array(2d,1,8).map(_ * images.length)
    bGrad.as1D.data should ===(expectedBias)
  }
}
