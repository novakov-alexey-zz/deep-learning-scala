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

@main def CNN() =
  val cnn = Sequential[Double, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 64,
    gradientClipping = clipByValue(5.0d),
    printStepTps = true
  )
    .add(Conv2D(relu, 4, kernel = (5, 5)))    
    .add(MaxPool(strides = (2, 2), pool = (2, 2)))       
    .add(Flatten2D())
    .add(Dense(relu, 6))      
    .add(Dense(softmax, 10))
  
  val dataset = MnistLoader.loadData[Double](imageDir, flat = false)
  val (xTrain, yTrain) = prepareData(dataset.trainImage, dataset.trainLabels)
  
  val start = System.currentTimeMillis()  
  val model = cnn.train(xTrain, yTrain, epochs = 10, shuffle = true)
  println(s"training time: ${(System.currentTimeMillis() - start) / 1000f} in sec")

  val (xTest, yTest) = prepareData(dataset.testImages, dataset.testLabels)
  val testPredicted = model(xTest)
  val value = accuracy(yTest, testPredicted)
  println(s"test accuracy = $value")  