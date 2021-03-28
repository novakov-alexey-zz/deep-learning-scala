package examples

import ml.tensors.api._
import ml.tensors.ops._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Using

import java.io.{DataInputStream, BufferedInputStream, FileInputStream}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

// data to be taken from http://yann.lecun.com/exdb/mnist/ or at GitHub somewhere
object MnistLoader:
  val trainImagesFilename = "train-images-idx3-ubyte.gz"
  val trainLabelsFilename = "train-labels-idx1-ubyte.gz"
  val testImagesFilename = "t10k-images-idx3-ubyte.gz"
  val testLabelsFilename = "t10k-labels-idx1-ubyte.gz"

  val LabelFileMagicNumber = 2049
  val ImageFileMagicNumber = 2051

  case class MnistDataset[T: Numeric](
    trainImage: Tensor[T],
    trainLabels: Tensor[T],
    testImages: Tensor[T],
    testLabels: Tensor[T]
  )

  case class LoaderCfg(samples: Int, numberOfImages: Int, nRows: Int, nCols: Int)

  def loadData[T: Numeric: ClassTag](
      mnistDir: String,
      samples: Int = 60_000,
      flat: Boolean =  true
  ): MnistDataset[T] =
    val (trainImages, trainLabels) = loadDataset(
      Path.of(mnistDir, trainImagesFilename),
      Path.of(mnistDir, trainLabelsFilename),
      samples,
      flat
    )
    val (testImages, testLabels) = loadDataset(
      Path.of(mnistDir, testImagesFilename),
      Path.of(mnistDir, testLabelsFilename),
      samples,
      flat
    )
    MnistDataset(trainImages, trainLabels, testImages, testLabels)

  private def loadDataset[T: ClassTag](
      images: Path,
      labels: Path,
      samples: Int,
      flat: Boolean
  )(using n: Numeric[T]): (Tensor[T], Tensor[T]) =
    Using.resource(
      new DataInputStream(        
        new GZIPInputStream(Files.newInputStream(images))        
      )
    ) { imageInputStream =>
      val magicNumber = imageInputStream.readInt()
      assert(
        magicNumber == ImageFileMagicNumber,
        s"Image file magic number is incorrect, expected $ImageFileMagicNumber, but was $magicNumber"
      )

      val numberOfImages = imageInputStream.readInt()
      val (nRows, nCols) =
        (imageInputStream.readInt(), imageInputStream.readInt())

      Using.resource(
        new DataInputStream(        
          new GZIPInputStream(Files.newInputStream(labels))          
        )
      ) { labelInputStream =>
        val labelMagicNumber = labelInputStream.readInt()
        assert(
          labelMagicNumber == LabelFileMagicNumber,
          s"Image file magic number is incorrect, expected $LabelFileMagicNumber, but was $labelMagicNumber"
        )

        val numberOfLabels = labelInputStream.readInt()

        assert(
          numberOfImages == numberOfLabels,
          s"Number of images is not equal to number of labels, $numberOfImages != $numberOfLabels"
        )

        val labelsTensor = labelInputStream.readAllBytes
          .map(l => n.fromInt(l))
          .take(samples)
          .as1D
    
        val cfg = LoaderCfg(samples, numberOfImages, nRows, nCols)
        val images = 
          if flat then readAsVector(cfg, imageInputStream) 
          else readAsMatrix(cfg, imageInputStream)

        (images, labelsTensor)
      }
    }

  private def readAsVector[T: ClassTag](cfg: LoaderCfg, imageInputStream: DataInputStream)(using n: Numeric[T]) = 
    val images = ArrayBuffer.empty[Array[T]]
    val singeImageSize = cfg.nRows * cfg.nCols
    
    for i <- (0 until cfg.numberOfImages) do      
      val image = (0 until singeImageSize).map(_ => n.fromInt(imageInputStream.readUnsignedByte())).toArray      
      images += image    
    
    images.toArray.take(cfg.samples).as2D

  private def readAsMatrix[T: ClassTag](cfg: LoaderCfg, imageInputStream: DataInputStream)(using n: Numeric[T]) = 
    val images = ArrayBuffer.empty[Array[Array[Array[T]]]]

    for i <- (0 until cfg.numberOfImages) do
      val image = ArrayBuffer.empty[Array[T]]
      for _ <- (0 until cfg.nRows) do            
        val row = (0 until cfg.nCols).map(_ => n.fromInt(imageInputStream.readUnsignedByte())).toArray
        image += row
      images += Array(image.toArray)    
    
    images.toArray.take(cfg.samples).as4D
