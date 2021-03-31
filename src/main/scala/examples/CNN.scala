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

@main def CNN() =
  type Precision = Float
  val accuracy = accuracyMnist[Precision]

  // type CNNNormal
  // given [T: ClassTag: Numeric]: ParamsInitializer[T, CNNNormal] with
  //   val rnd = new Random() 

  //   def gen(scale: Double): T = 
  //     castFromTo[Double, T] {
  //       val v = rnd.nextGaussian + 0.01d
  //       v * scale
  //     }

  //   override def weights4D(shape: List[Int])(using c: ClassTag[T], n: Numeric[T]): Tensor4D[T] = 
  //     val tensors :: cubes :: rows :: cols :: _ = shape
  //     val size = shape.reduce(_ * _)
  //     val scale = math.sqrt(2d / size)
  //     def w2d = Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen(scale))))
  //     (0 until tensors).map(_ =>  (0 until cubes).toArray.map(_ => w2d)).toArray.as4D

  //   override def weights(rows: Int, cols: Int): Tensor2D[T] =
  //     Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen(rows))))

  //   override def biases(length: Int): Tensor1D[T] = 
  //     inits.zeros(length)

  def clipByNorm[T: Fractional: ClassTag](norm: T) = new GradientClipping[T]:     
    def apply(t: Tensor[T]) =
      t match
        case (Tensor4D(data)) => 
          data.map(_.map(_.as2D.clipByNorm(norm).as2D)).as4D // clipping within matrix 
        case _ => 
          t.clipByNorm(norm)

  val cnn = Sequential[Precision, Adam, HeNormal](
    crossEntropy,
    learningRate = 0.001,
    metrics = List(accuracy),
    batchSize = 128,
    gradientClipping = clipByNorm(1.0),
    printStepTps = true
  )    
    .add(Conv2D(relu, 16, kernel = (5, 5)))    
    .add(MaxPool(strides = (1, 1), pool = (2, 2), padding = false))       
    .add(Flatten2D())
    .add(Dense(relu, 32))      
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