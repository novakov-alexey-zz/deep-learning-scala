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
  type Precision = Float
  val accuracy = accuracyMnist[Precision]

  val cnn = Sequential[Precision, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByValue(5.0),
    printStepTps = true
  )
    .add(Conv2D(relu, 8, kernel = (5, 5)))    
    .add(MaxPool(strides = (2, 2), pool = (2, 2), padding = false))
    // .add(Conv2D(relu, 8, kernel = (5, 5)))    
    // .add(MaxPool(strides = (2, 2), pool = (2, 2), padding = false))       
    .add(Flatten2D())
    .add(Dense(relu, 128))      
    .add(Dense(softmax, 10))
  
  val dataset = MnistLoader.loadData[Precision](imageDir, flat = false)
  val (xTrain, yTrain) = prepareData(dataset.trainImage, dataset.trainLabels)
  
  val start = System.currentTimeMillis()  
  val model = cnn.train(xTrain, yTrain, epochs = 5, shuffle = true)
  println(s"training time: ${(System.currentTimeMillis() - start) / 1000f} in sec")

  val (xTest, yTest) = prepareData(dataset.testImages, dataset.testLabels)
  val testPredicted = model(xTest)
  val value = accuracy(yTest, testPredicted)
  println(s"test accuracy = $value")  