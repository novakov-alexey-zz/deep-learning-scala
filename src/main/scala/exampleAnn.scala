import ml.preprocessing._
import ml.transformation.castTo
import ml.tensors.api._
import ml.tensors.ops._
import ml.network.api._
import ml.network.api.given

import java.nio.file.Path
import java.io.{File, PrintWriter}
import scala.reflect.ClassTag
import scala.util.Using

@main def ann() =

  def createEncoders[T: Numeric: ClassTag](
    data: Tensor2D[String]
  ): Tensor2D[String] => Tensor2D[T] =
    val encoder = LabelEncoder[String]().fit(data.col(2))
    val hotEncoder = OneHotEncoder[String, T]().fit(data.col(1))
    
    val label = t => encoder.transform(t, 2)
    val hot = t => hotEncoder.transform(t, 1)
    val typeTransform = (t: Tensor2D[String]) => castTo[T](t.data)
    
    label andThen hot andThen typeTransform
  
  val accuracy = accuracyBinaryClassification[Double]
  
  val ann = Sequential[Double, Adam](
    binaryCrossEntropy,
    learningRate = 0.002d,
    metrics = List(accuracy),
    batchSize = 64,
    gradientClipping = clipByValue(5.0d)
  )
    .add(Dense(relu, 6))
    .add(Dense(relu, 6))    
    .add(Dense(sigmoid))
  
  val dataLoader = TextLoader(Path.of("data", "Churn_Modelling.csv")).load()
  val data = dataLoader.cols[String](3, -1)
  
  val encoders = createEncoders[Double](data)
  val numericData = encoders(data)
  val scaler = StandardScaler[Double]().fit(numericData)
  
  val prepareData = (t: Tensor2D[String]) => {
    val numericData = encoders(t)
    scaler.transform(numericData)
  }
  
  val x = prepareData(data)
  val y = dataLoader.cols[Double](-1)
  
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

  val header = s"epoch,loss,${model.metricValues.map(_._1.name).mkString(",")}"
  val acc = model.metricValues.headOption.map(_._2).getOrElse(Nil)
  val lrData = model.history.losses.zip(acc).zipWithIndex.map { case ((loss, acc), epoch) =>      
    List(epoch.toString, loss.toString, acc.toString)      
  } 
  store("metrics/ann.csv", header, lrData)