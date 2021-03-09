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
//   val image1 = dataset.trainImage.as2D.data.tail.head

//   println(image1.grouped(28).map(_.foldLeft(""){ (acc, s) =>
//     acc + f"${s.toInt}%4s"
//   }).mkString("\n"))
  //println(dataset.trainLabels.as1D.data.take(10).mkString(","))

  def accuracyMnist[T: ClassTag: Ordering](using n: Fractional[T]) = new Metric[T]:
    val name = "accuracy"
    
    def matches(actual: Tensor[T], predicted: Tensor[T]): Int =      
      val predictedArgMax = predicted.argMax      
      actual.argMax.equalRows(predictedArgMax)
      
  val ann = Sequential[Double, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracyMnist),
    batchSize = 64,
    gradientClipping = clipByValue(5.0d)
  )
    .add(Dense(relu, 24))    
    .add(Dense(softmax, 10))
  
  val encoder = OneHotEncoder(classes = 
    (0 to 9).zipWithIndex.toMap.map((k, v) => (k.toDouble, v.toDouble))
  )  

  val yTrain = encoder.transform(dataset.trainLabels.as1D)
  val xTrain = dataset.trainImage.map(_ / 255d) // normalize to [0,1] range
  
  val model = ann.train(xTrain, yTrain, epochs = 20, shuffle = true)