import ml.network.api._
import ml.network.api.given
import ml.tensors.api._
import ml.tensors.ops._
import ml.network.inits.given

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._
import scala.collection.mutable.ArrayBuffer
import scala.util.{Random, Using}
import scala.collection.parallel.CollectionConverters._

import java.io.{File,PrintWriter}

@main def linearRegression() =
  val random = new Random(100)
  val weight = random.nextFloat()
  val bias = random.nextFloat()

  def batch(batchSize: Int): (ArrayBuffer[Double], ArrayBuffer[Double]) =
    val inputs = ArrayBuffer.empty[Double]
    val outputs = ArrayBuffer.empty[Double]
    def noise = random.nextDouble / 5
    (0 until batchSize).foldLeft(inputs, outputs) { case ((x, y), _) =>        
        val rnd = random.nextDouble
        x += rnd + noise
        y += bias + weight * rnd + noise
        (x, y)
    }
  
  val alg = "adam"

  val ann = Sequential[Double, Adam, RandomUniform](
    meanSquareError,
    learningRate = 0.0012f,    
    batchSize = 16,
    gradientClipping = clipByValue(5.0d)
  ).add(Dense())    

  val (xBatch, yBatch) = batch(10000)
  val x = Tensor1D(xBatch.toArray)
  val y = Tensor1D(yBatch.toArray)
  val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)

  val model = ann.train(xTrain.T, yTrain.T, epochs = 100)

  println(s"current weight: ${model.layers}")
  println(s"true weight: $weight")
  println(s"true bias: $bias")

  // Test Dataset
  val testPredicted = model(xTest)  
  val value = meanSquareError[Double].apply(yTest.T, testPredicted)
  println(s"test meanSquareError = $value")

  //////////////////////////////////////////
  // Store all posible data for plotting ///
  //////////////////////////////////////////

  // datapoints
  val dataPoints = xBatch.zip(yBatch).map((x, y) => List(x.toString, y.toString))
  store("metrics/datapoints.csv", "x,y", dataPoints.toList)

  //Store loss metric into CSV file
  val lossData = model.history.losses.zipWithIndex.map((l,i) => List(i.toString, l.toString))
  store("metrics/lr.csv", "epoch,loss", lossData)

  //gradient
  val gradientData = model.history.layers.zip(model.history.losses)
      .map { (layers, loss) => 
        layers.headOption.map(l => 
          List(l.w.as1D.data.head.toString, l.b.as1D.data.head.toString)
        ).toList.flatten :+ loss.toString
      }

  store(s"metrics/$alg-gradient.csv", "w,b,loss", gradientData)

  // loss surface
  val weights = for (i <- 0 until 100) yield i/100d 
  val biases = weights
  
  println("Calculating loss surface")
  val losses = weights.par.map { w =>
    val wT = w.as2D
    biases.foldLeft(ArrayBuffer.empty[Double]) { (acc, b) =>
      val loss = ann.loss(x.T, y.T, List(Layer(wT, b.as1D)))  
      acc :+ loss
    }
  }
  println("Done calculating loss surface.")

  val metricsData = weights.zip(biases).zip(losses)
    .map { case ((w, b), l) => List(w.toString, b.toString, l.mkString("\"", ",", "\"")) }
  
  store(s"metrics/$alg-lr-surface.csv", "w,b,l", metricsData.toList)