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
  val dataset = MnistLoader.loadData[Double]("images")  

  def accuracyMnist[T: ClassTag: Ordering](using n: Numeric[T]) = new Metric[T]:
    val name = "accuracy"
    
    def matches(actual: Tensor[T], predicted: Tensor[T]): Int =      
      val predictedArgMax = predicted.argMax      
      actual.argMax.equalRows(predictedArgMax)
      
  val accuracy = accuracyMnist[Double]    
  val ann = Sequential[Double, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByValue(5.0d)
  )
    .add(Dense(relu, 24))    
    .add(Dense(relu, 16))    
    .add(Dense(softmax, 10))
  
  val encoder = OneHotEncoder(classes = 
    (0 to 9).zipWithIndex.toMap.map((k, v) => (k.toDouble, v.toDouble))
  )  

  def prepareData(x: Tensor[Double], y: Tensor[Double]) =
    val xData = x.map(_ / 255d) // normalize to [0,1] range
    val yData = encoder.transform(y.as1D)
    (xData, yData) 

  val (xTrain, yTrain) = prepareData(dataset.trainImage, dataset.trainLabels)
  val model = ann.train(xTrain, yTrain, epochs = 10, shuffle = true)

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