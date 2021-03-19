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
  it should "convolution on input data" in {
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
    val images = image1//Tensor4D(image1)//, image2)
        
    def testActivation[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:
      override def apply(x: Tensor[T]): Tensor[T] = x.map(t => t + n.one)
      override def derivative(x: Tensor[T]): Tensor[T] = apply(x)
      override val name = "test"

    given testInit[T: Numeric: ClassTag]: ParamsInitializer[T, RandomUniform] with    
    
      def gen: T = 
        summon[Numeric[T]].one

      override def weights(rows: Int, cols: Int): Tensor2D[T] =
        Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen)))

      override def biases(length: Int): Tensor1D[T] = 
        inits.zeros(length)

    val kernel = (2,2)
    val strides = (1, 1)
    val imageCount = images.length    
    val filterCount = 3
    val inputShape = images.shape3D

    val layer = Conv2D[Double](
      f = testActivation,
      filterCount = filterCount,
      kernel = kernel,
      strides = strides
    ).init(inputShape.toList, testInit, adam)
    println("weights:\n" + layer.w)
    println("biases:\n" + layer.b)

    val a = layer(images)
    println(s"z:\n${a.z}")
    println(s"a:\n${a.a}")

    val (inputChannels, width, height) = inputShape
    a.z.shape should be (List(filterCount, 2, 3))
    val w = layer.w.getOrElse(fail("Weight must not be empty"))
    val b = layer.b.getOrElse(fail("Bias must not be empty"))  

    def applyFilter[T: ClassTag: Numeric](filter: Array[Array[T]], window: Array[Array[T]]): T =
      filter.zip(window).map((a, b) => a.zip(b).map(_ * _).sum).sum

    def filterChannel[T: Numeric: ClassTag](channel: Array[Array[T]])(filter: Array[Array[T]]) = 
      val rows = ListBuffer[Array[T]]()

      for i <- 0 to width - kernel._1 by strides._1 do  
        val img = channel.drop(i).take(kernel._1)
        val row = ListBuffer.empty[T]

        for j <- 0 to height - kernel._1 by strides._2 do           
          val window = img.map(_.drop(j).take(kernel._2))                                
          row += applyFilter(filter, window)            

        rows += row.toArray  
      rows.toArray

    val expectedActivities = w.as4D.data.map { channels => 
        channels.zip(image1.data).map { (fc, ic) =>
          filterChannel(ic)(fc).as2D
        }.reduce(_ + _)       
    }
            
    // println(s"expectedActivity:\n${Tensor4D(expectedActivity)}")
    // println(s"actualActivity:\n${a.z}")

    val layerActivity = a.z.as3D.data    
    layerActivity.zip(expectedActivities).foreach { (actual, expected) =>      
      actual should ===(expected.data)      
    }
    
    val expectedActivation = layer.f(expectedActivities.as3D)
    a.a.as3D.data sameElements expectedActivation.as3D.data    
  }
}
