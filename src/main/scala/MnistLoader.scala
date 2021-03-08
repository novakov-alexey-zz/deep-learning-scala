import ml.tensors.api._
import ml.tensors.ops._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import java.io.{DataInputStream, BufferedInputStream, FileInputStream}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream

object MnistLoader:
  val trainImagesFilename = "train-images-idx3-ubyte.gz"
  val trainLabelsFilename = "train-labels-idx1-ubyte.gz"
  val testImagesFilename = "t10k-images-idx3-ubyte.gz"
  val testLabelsFilename = "t10k-labels-idx1-ubyte.gz"

  case class MnistDataset[T: Numeric](
    trainImage: Tensor[T], 
    trainLabels: Tensor[T], 
    testImages: Tensor[T], 
    testLabels: Tensor[T]
  )

  def loadData[T: Numeric: ClassTag](mnistDir: String): MnistDataset[T] =     
    val (trainImages, trainLabels) = loadDataset(
        Path.of(mnistDir, trainImagesFilename), Path.of(mnistDir, trainLabelsFilename))
    val (testImages, testLabels) = loadDataset(
        Path.of(mnistDir, testImagesFilename), Path.of(mnistDir, testLabelsFilename))
    MnistDataset(trainImages, trainLabels, testImages, testLabels)
  
  private def loadDataset[T: ClassTag](images: Path, labels: Path)(using n: Numeric[T]): (Tensor[T], Tensor[T]) =
    val imageStream = new GZIPInputStream(Files.newInputStream(images))
    val imageInputStream = new DataInputStream(new BufferedInputStream(imageStream))
    val magicNumber = imageInputStream.readInt()
    val numberOfItems = imageInputStream.readInt()
    val nRows = imageInputStream.readInt()
    val nCols = imageInputStream.readInt()

    // println("magic number: " + magicNumber)
    // println("number of items: " + numberOfItems)
    // println("number of rows: " + nRows)
    // println("number of cols: " + nCols)

    val labelStream = new GZIPInputStream(Files.newInputStream(labels))
    val labelInputStream = new DataInputStream(new BufferedInputStream(labelStream))
    val labelMagicNumber = labelInputStream.readInt()
    val numberOfLabels = labelInputStream.readInt()

    // println(s"labels magic number: $labelMagicNumber")
    // println(s"number of labels: $numberOfLabels")

    assert(numberOfItems == numberOfLabels)

    val labelsTensor = labelInputStream.readAllBytes.map(l => n.fromInt(l)).take(2).as1D
    val singeImageSize = nRows * nCols
    val imageArray = ArrayBuffer.empty[Array[T]]

    for i <- (0 until numberOfItems) do
      val image = (0 until singeImageSize)
        .map(_ => n.fromInt(imageInputStream.readUnsignedByte())).toArray      
      imageArray += image

    (imageArray.toArray.take(2).as2D, labelsTensor)
