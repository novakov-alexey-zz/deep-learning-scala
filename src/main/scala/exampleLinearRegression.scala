import ActivationFunc._
import Loss._
import ops._
import optimizers.given
import RandomGen.uniform

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

@main def linearRegression() =       
  val random = new Random()
  val weight = random.nextFloat()
    
  def batch(batchSize: Int): (ArrayBuffer[Float], ArrayBuffer[Float]) = {
    val inputs = ArrayBuffer.empty[Float]
    val outputs = ArrayBuffer.empty[Float]
    var i = 0
    while (i < batchSize) {
      val input = random.nextFloat()
      inputs += input
      outputs += weight * input
      i += 1
    }

    (inputs, outputs)    
  }

  val ann = Sequential[Float, SimpleGD](
    meanSquareError,
    learningRate = 0.015f,    
    batchSize = 64
  ).add(Dense())    

  val (xBatch, yBatch) = batch(10000)
  val x = Tensor1D(xBatch.toArray)
  val y = Tensor1D(yBatch.toArray)
  val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)

  val model = ann.train(xTrain.T, yTrain.T, epochs = 100)

  println(s"current weight: ${model.currentWeights}")
  println(s"true weight: $weight")

  // Test Dataset
  val testPredicted = model.predict(xTest)  
  val value = meanSquareError[Float].apply(yTest.T, testPredicted)
  println(s"test meanSquareError = $value")  