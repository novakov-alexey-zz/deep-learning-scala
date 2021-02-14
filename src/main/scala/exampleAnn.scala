import ActivationFunc._
import Loss._
import Metric.{accuracyMetric, predictedToBinary}
import converter.transform
import ops._
import optimizers.given
import RandomGen._

import java.nio.file.Path
import scala.reflect.ClassTag

@main def ann() = 

  def createEncoders[T: Numeric: ClassTag](
    data: Tensor2D[String]
  ): Tensor2D[String] => Tensor2D[T] =
    val encoder = LabelEncoder[String]().fit(data.col(2))
    val hotEncoder = OneHotEncoder[String, T]().fit(data.col(1))
    
    val label = t => encoder.transform(t, 2)
    val hot = t => hotEncoder.transform(t, 1)
    val typeTransform = (t: Tensor2D[String]) => transform[T](t.data)
    
    label andThen hot andThen typeTransform
  
  val accuracy = accuracyMetric[Float]
  
  val ann = Sequential[Float, SimpleGD](
    binaryCrossEntropy,
    learningRate = 0.019f,
    metrics = List(accuracy),
    batchSize = 64
  )
    .add(Dense(relu, 6))
    .add(Dense(relu, 6))    
    .add(Dense(sigmoid))
  
  val dataLoader = TextLoader(Path.of("data", "Churn_Modelling.csv")).load()
  val data = dataLoader.cols[String](3, -1)
  
  val encoders = createEncoders[Float](data)
  val numericData = encoders(data)
  val scaler = StandardScaler[Float]().fit(numericData)
  
  val prepareData = (t: Tensor2D[String]) => {
    val numericData = encoders(t)
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
  val value = accuracy(yTest, testPredicted)
  println(s"test accuracy = $value")  