package examples

import mnistCommon._
import ml.transformation.{castTo, castFromTo}
import ml.tensors.api._
import ml.tensors.ops._
import ml.network.api._
import ml.network.api.given
import ml.network.inits.given
import ml.preprocessing._

import java.nio.file.Path
import scala.reflect.ClassTag

@main def MNIST() =
  val dataset = MnistLoader.loadData[Double](imageDir, flat = true)

  val ann = Sequential[Double, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByValue(5.0d)
  )
    .add(Dense(relu, 50))      
    .add(Dense(softmax, 10))

  val (xTrain, yTrain) = prepareData(dataset.trainImage, dataset.trainLabels)
  val start = System.currentTimeMillis()
  val model = ann.train(xTrain, yTrain, epochs = 15, shuffle = true)
  println(s"training time: ${(System.currentTimeMillis() - start) / 1000f} in sec")
  
  val (xTest, yTest) = prepareData(dataset.testImages, dataset.testLabels)
  val testPredicted = model(xTest)
  val value = accuracy(yTest, testPredicted)
  println(s"test accuracy = $value")  

  // Single Test
  val singleTestImage = dataset.testImages.as2D.data.head
  val imageMap = singleTestImage.grouped(28)
    .map(_.map(s => f"${s.toInt}%4s").mkString).mkString("\n")
  println(imageMap)
  val label = dataset.testLabels.as1D.data.head  
  val predicted = model(singleTestImage.as2D).argMax.as0D.data  
  println(s"predicted = $predicted")
  
  assert(label == predicted, 
    s"Predicted label is not equal to expected '$label' label, but was '$predicted'")
  
  storeMetrics(model, Path.of("metrics/mnist.csv"))    