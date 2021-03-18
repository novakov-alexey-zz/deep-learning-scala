package ml.network

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
    val image = Tensor3D(Array(
      Array(
        Array(1d, 2, 3, 1), 
        Array(2d, 3, 4, 0), 
        Array(5d, 6, 7, 8)
      ),
      Array(
        Array(1d, 2, 3, 1), 
        Array(2d, 3, 4, 0), 
        Array(5d, 6, 7, 8)
      ),
      Array(
        Array(1d, 2, 3, 1), 
        Array(2d, 3, 4, 0), 
        Array(5d, 6, 7, 8)
      )
    ))
        
    def testActivation[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:
      override def apply(x: Tensor[T]): Tensor[T] = x.map(t => t + n.one)
      override def derivative(x: Tensor[T]): Tensor[T] = apply(x)
      override val name = "test"

    val kernel = (2,2)
    val strides = (1, 1)
    val images = 1    
    val filterCount = 3
    val inputShape = List(images) ++ image.shape3D.toList

    val layer = Conv2D[Double](
      f = testActivation,
      filterCount = filterCount,
      kernel = kernel,
      strides = strides
    ).init(inputShape, normal, adam)    
    println("weights:\n" + layer.w)
    println("biases:\n" + layer.b)

    val a = layer(Tensor4D(image))
    println(s"z:\n${a.z}")
    println(s"a:\n${a.a}")

    a.z.shape should be (List(image.shape3D.head, filterCount, 2, 3))
    val w = layer.w.getOrElse(fail("Weight must not be empty"))
    val b = layer.b.getOrElse(fail("Bias must not be empty"))  

    val (_, width, height) = image.shape3D

    def applyFilter[T: ClassTag: Numeric](filter: Array[Array[T]], window: Array[Array[T]]) =
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

    val expectedActivity = image.data.zip(w.as4D.data).map { (channel, fs) =>
      fs.map(filterChannel(channel))
    }

    // println(s"expectedActivity:\n${Tensor4D(expectedActivity)}")
    // println(s"actualActivity:\n${a.z}")

    val layerActivity = a.z.as4D.data
    layerActivity.zip(expectedActivity).foreach { (actual, expected) =>
      actual should be(expected)  
    }
    
    val expectedActivation = layer.f(Tensor4D(expectedActivity))
    
    a.a.as4D.data sameElements expectedActivation.as4D.data
  }
}
