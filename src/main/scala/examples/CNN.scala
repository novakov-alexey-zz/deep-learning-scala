package examples

import ml.transformation.{castTo, castFromTo}
import ml.tensors.api._
import ml.tensors.ops._
import ml.network.api._
import ml.network.api.given
import ml.network.inits.given
import ml.preprocessing._

import java.nio.file.Path
import scala.reflect.ClassTag

@main def CNN() =
  def accuracyMnist[T: ClassTag: Ordering](using n: Numeric[T]) = new Metric[T]:
    val name = "accuracy"
    
    def matches(actual: Tensor[T], predicted: Tensor[T]): Int =      
      val predictedArgMax = predicted.argMax      
      actual.argMax.equalRows(predictedArgMax)
      
  val accuracy = accuracyMnist[Double]    

  type TestInit
  given testInit[T: Numeric: ClassTag]: ParamsInitializer[T, TestInit] with    
  
    def gen: T = summon[Numeric[T]].one

    override def weights(rows: Int, cols: Int): Tensor2D[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen)))

    override def biases(length: Int): Tensor1D[T] = 
      inits.zeros(length)

  val cnn = Sequential[Double, StandardGD, TestInit](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByValue(5.0d)
  )
    .add(Conv2D(relu, 3))      
    .add(MaxPool())       
    .add(Flatten2D())
    .add(Dense(relu, 6))      
    .add(Dense(softmax, 10))

  val debugX = Tensor4D(
    Tensor3D(
      Array(
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
      )
    ))

  val debugY = Array(Array.fill(10)(0d)).as2D  
  val model = cnn.train(debugX, debugY, epochs = 1, shuffle = false)