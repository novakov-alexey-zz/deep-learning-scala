import ActivationFunc._
import Loss._
import Metric.{accuracyMetric, predictedToBinary}
import converter.transform
import ops._
import optimizers.given
import RandomGen._

import java.nio.file.Paths
import scala.reflect.ClassTag

@main def starter() = 

  def createEncoders[T: Numeric: ClassTag](
    data: Tensor2D[String]
  ): Tensor2D[String] => Tensor2D[T] =
    val encoder = LabelEncoder[String]().fit(data.col(2))
    val hotEncoder = OneHotEncoder[String, T]().fit(data.col(1))
    
    val label = x => encoder.transform(x, 2)
    val hot = x => hotEncoder.transform(x, 1)
    val typeTransform = (x: Tensor2D[String]) => transform[T](x.data)
    
    label andThen hot andThen typeTransform
  
  val accuracy = accuracyMetric[Float]
  
  val ann = Sequential[Float, MiniBatchGD](
    binaryCrossEntropy,
    learningRate = 0.05f,
    metric = accuracy,
    batchSize = 32
  )
    .add(Dense(relu, 6))
    .add(Dense(relu, 6))    
    .add(Dense(sigmoid))
  
  val dataLoader = TextLoader(Paths.get("data", "Churn_Modelling.csv")).load()
  val data = dataLoader.cols[String](3, -1)
  
  val encoders = createEncoders[Float](data)
  val numericData = encoders(data)
  val scaler = StandardScaler[Float]().fit(numericData)
  
  val prepareData = (d: Tensor2D[String]) => {
    val numericData = encoders(d)
    scaler.transform(numericData)
  }
  
  val x = prepareData(data)
  val y = dataLoader.cols[Float](-1)
  
  val ((xTrain, xTest), (yTrain, yTest)) = (x, y).split(0.2f)
  
  val start = System.currentTimeMillis()
  val model = ann.train(xTrain, yTrain, epochs = 100)
  println(s"training time: ${(System.currentTimeMillis() - start) / 1000f} in sec")

  // Single test
  val example = TextLoader(
    "n/a,n/a,n/a,600,France,Male,40,3,60000,2,1,1,50000,n/a"
  ).cols[String](3, -1)
  val testExample = prepareData(example)
  val exited = predictedToBinary(model.predict(testExample).as1D.data.head) == 1
  println(s"Exited customer? $exited")
  
  // Test Dataset
  val testPredicted = model.predict(xTest)
  val value = accuracy(yTest.as1D, testPredicted.as1D)
  println(s"test accuracy = $value")  