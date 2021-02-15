import ActivationFunc._
import Loss._
import ops._
import optimizers.given
import RandomGen.uniform

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._
import scala.collection.mutable.ArrayBuffer
import scala.util.{Random, Using}
import java.io.File
import java.io.PrintWriter

@main def linearRegression() =       
  val random = new Random(100)
  val weight = random.nextFloat()
    
  def batch(batchSize: Int): (ArrayBuffer[Double], ArrayBuffer[Double]) = {
    val inputs = ArrayBuffer.empty[Double]
    val outputs = ArrayBuffer.empty[Double]
    (0 until batchSize).foldLeft(inputs, outputs) { case ((i, o), _) =>
        val input = random.nextDouble()
        i += input
        o += weight * input
        (i, o)
    }
  }

  val ann = Sequential[Double, SimpleGD](
    meanSquareError,
    learningRate = 0.0015f,    
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
  val value = meanSquareError[Double].apply(yTest.T, testPredicted)
  println(s"test meanSquareError = $value")

  Using.resource(new PrintWriter(new File("metrics/lr.csv"))) { w =>
    w.write("epoch,loss")
    model.losses.foldLeft(1) { case (epoch, l) =>      
      w.write(s"\n$epoch,$l")
      epoch + 1
    }
  }