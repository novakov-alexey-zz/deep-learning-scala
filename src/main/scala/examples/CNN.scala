package examples

import mnistCommon._
import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._
import ml.network.api._
import ml.network.api.given
import ml.network.api.inits.given
import ml.preprocessing._

import java.nio.file.Path
import java.util.Random
import scala.reflect.ClassTag

@main 
def CNN() =
  type Precision = Float
  val accuracy = accuracyMnist[Precision]

  def clipByNorm[T: Fractional: ClassTag](norm: T) = new GradientClipping[T]:     
    def apply(t: Tensor[T]) =
      t match
        case (Tensor4D(data)) => 
          data.map(_.map(_.as2D.clipByNorm(norm).as2D)).as4D // clipping within matrix 
        case _ => 
          t.clipByNorm(norm)

  val cnn = Sequential[Precision, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.0015,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByNorm(10.0),
    printStepTps = true
  )    
    .add(Conv2D(relu, 8, kernel = (5, 5)))    
    .add(MaxPool(strides = (2, 2), window = (4, 4), padding = false))    
    .add(Flatten2D())
    .add(Dense(relu, 64))      
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