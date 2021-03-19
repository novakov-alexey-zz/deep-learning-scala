package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._
import optimizers.given_Optimizer_Adam as adam
import inits.given_ParamsInitializer_T_HeNormal as normal

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

  given testInit[T: Numeric: ClassTag]: ParamsInitializer[T, RandomUniform] with    
  
    def gen: T = 
      summon[Numeric[T]].one

    override def weights(rows: Int, cols: Int): Tensor2D[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen)))

    override def biases(length: Int): Tensor1D[T] = 
      inits.zeros(length)
      
  it should "apply convolution on input data" in {
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
    val inputShape = images.shape4D

    val layer = Conv2D[Double](
      f = testActivation,
      filterCount = 3,
      kernel = (2, 2),
      strides = (1, 1)
    ).init(inputShape.toList, normal, adam)    

    val a = layer(images)    

    val (imageCount, inputChannels, width, height) = inputShape
    a.z.shape should ===(List(imageCount, layer.filterCount, 2, 3))
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
            
    val layerActivity = a.z.as4D.data    
    layerActivity.zip(expectedActivities.data).foreach { (actual, expected) =>      
      actual should ===(expected)      
    }
    
    val expectedActivation = layer.f(expectedActivities)
    a.a.as4D.data sameElements expectedActivation.as4D.data    
  }
}
