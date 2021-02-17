import ActivationFunc._
import Loss._
import ops._
import optimizers.given
import RandomGen.uniform

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._
import scala.collection.mutable.ArrayBuffer
import scala.util.{Random, Using}
import scala.concurrent.{Future, Await}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import java.io.{File,PrintWriter}

@main def linearRegression() =       
  val random = new Random(100)
  val weight = random.nextFloat()
  val bias = random.nextFloat()  

  def batch(batchSize: Int): (ArrayBuffer[Double], ArrayBuffer[Double]) =
    val inputs = ArrayBuffer.empty[Double]
    val outputs = ArrayBuffer.empty[Double]
    (0 until batchSize).foldLeft(inputs, outputs) { case ((i, o), _) =>
        val input = random.nextDouble()
        i += input
        o += bias + weight * input 
        (i, o)
    }

  val ann = Sequential[Double, SimpleGD](
    meanSquareError,
    learningRate = 0.0002f,    
    batchSize = 100
  ).add(Dense())    

  val (xBatch, yBatch) = batch(10000)
  val x = Tensor1D(xBatch.toArray)
  val y = Tensor1D(yBatch.toArray)
  val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)

  val model = ann.train(xTrain.T, yTrain.T, epochs = 300)

  println(s"current weight: ${model.currentWeights}")
  println(s"true weight: $weight")
  println(s"true bias: $bias")

  // Test Dataset
  val testPredicted = model.predict(xTest)  
  val value = meanSquareError[Double].apply(yTest.T, testPredicted)
  println(s"test meanSquareError = $value")

  //Store loss metric into CSV file
  val lossData = model.losses.zipWithIndex.map((l,i) => List(i.toString, l.toString))
  store("metrics/lr.csv", "epoch,loss", lossData)

  // Loss Surface calculation
  val weightsF = Future {
    (0 until 200).foldLeft((-2.0d, ArrayBuffer.empty[Double])) { case ((n, acc), _) =>
      (n + random.between(0.01d, 0.03d), acc :+ n)
    }._2.toList
  }
  val biasesF = Future { 
    (0 until 200).foldLeft((-2.6d, ArrayBuffer.empty[Double])) { case ((n, acc), _) =>
      (n + random.between(0.01d, 0.03d), acc :+ n)
    }._2.toList
  }

  val result = Await.result(
    (for {
      weight <- weightsF
      biases <- biasesF
    } yield (weight, biases)), 
    60.seconds)
  val (weights, biases) = result
  
  val losses = weights.map { w =>
    val wT = w.as0D
    biases.foldLeft(ArrayBuffer.empty[Double]) { (acc, b) =>
      val loss = ann.loss(x.T, y.T, List(Weight(wT, b.as0D)))  
      acc :+ loss
    }
  }
  
  val metricsData = weights.zip(biases).zip(losses)
    .map{ case ((w, b), l) => List(w.toString, b.toString, l.mkString("\"", ",", "\"")) }
  
  store("metrics/lr-surface.csv", "w,b,l", metricsData)