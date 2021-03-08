import ml.transformation.castTo
import ml.tensors.api._
import ml.tensors.ops._
import ml.network.api._
import ml.network.api.given
import ml.preprocessing._

import java.nio.file.Path
import scala.reflect.ClassTag

@main def mnist() =
  val dataset = MnistLoader.loadData[Double]("images")
//   val image1 = dataset.trainImage.as2D.data.tail.head

//   println(image1.grouped(28).map(_.foldLeft(""){ (acc, s) =>
//     acc + f"${s.toInt}%4s"
//   }).mkString("\n"))
  //println(dataset.trainLabels.as1D.data.take(10).mkString(","))

  def accuracyMnist[T: ClassTag](using n: Numeric[T]) = new Metric[T]:
    val name = "accuracy"

    def matches(actual: Tensor[T], predicted: Tensor[T]): Int =      
      val predictedNormalized = predicted.argMax
      // println(s"predictedArgMax = $predictedNormalized")
      // println(s"actualArgmax = ${actual.argMax}")
      actual.argMax.equalRows(predictedNormalized)

  val accuracy = accuracyMnist[Double]
  
  val ann = Sequential[Double, Adam](
    crossEntropy,
    learningRate = 0.01,
    metrics = List(accuracy),
    batchSize = 256,
    gradientClipping = clipByValue(5.0f)
  )
    .add(Dense(relu, 32))    
    .add(Dense(softmax, 10))
  
  val encoder = OneHotEncoder(classes = 
    (0 to 9).zipWithIndex.toMap.map((k, v) => (k.toDouble, v.toDouble))
  )
  val yTrain = encoder.transform(dataset.trainLabels.as1D)
  val xTrain = dataset.trainImage.map(_ / 255) // normalize to [0,1] range
  //println("x: " + xTrain.as2D.data.take(2).map(_.mkString(",")).mkString("\n\n"))
  //println(yTrain.data.take(10).map(a => a.mkString(",")).mkString("\n"))
  val model = ann.train(xTrain, yTrain, epochs = 1, shuffle = false)