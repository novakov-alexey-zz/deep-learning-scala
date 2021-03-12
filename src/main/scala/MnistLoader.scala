import ml.tensors.api._
import ml.tensors.ops._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Using

import java.io.{DataInputStream, BufferedInputStream, FileInputStream}
import java.nio.file.{Files, Path}
import java.util.zip.GZIPInputStream

// data to be taken from http://yann.lecun.com/exdb/mnist/
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

  def loadData[T: Numeric: ClassTag](
      mnistDir: String,
      samples: Int = 60_000
  ): MnistDataset[T] =
    val (trainImages, trainLabels) = loadDataset(
      Path.of(mnistDir, trainImagesFilename),
      Path.of(mnistDir, trainLabelsFilename),
      samples
    )
    val (testImages, testLabels) = loadDataset(
      Path.of(mnistDir, testImagesFilename),
      Path.of(mnistDir, testLabelsFilename),
      samples
    )
    MnistDataset(trainImages, trainLabels, testImages, testLabels)

  private def loadDataset[T: ClassTag](
      images: Path,
      labels: Path,
      samples: Int
  )(using n: Numeric[T]): (Tensor[T], Tensor[T]) =
    Using.resource(
      new DataInputStream(
        new BufferedInputStream(
          new GZIPInputStream(Files.newInputStream(images))
        )
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

      // println("magic number: " + magicNumber)
      // println("number of items: " + numberOfItems)
      // println("number of rows: " + nRows)
      // println("number of cols: " + nCols)

      Using.resource(
        new DataInputStream(
          new BufferedInputStream(
            new GZIPInputStream(Files.newInputStream(labels))
          )
        )
      ) { labelInputStream =>
        val labelMagicNumber = labelInputStream.readInt()
        assert(
          labelMagicNumber == LabelFileMagicNumber,
          s"Image file magic number is incorrect, expected $LabelFileMagicNumber, but was $labelMagicNumber"
        )

        val numberOfLabels = labelInputStream.readInt()

        // println(s"labels magic number: $labelMagicNumber")
        // println(s"number of labels: $numberOfLabels")

        assert(
          numberOfImages == numberOfLabels,
          s"Number of images is not equal to number of labels, $numberOfImages != $numberOfLabels"
        )

        val labelsTensor = labelInputStream.readAllBytes
          .map(l => n.fromInt(l))
          .take(samples)
          .as1D

        val singeImageSize = nRows * nCols
        val imageArray = ArrayBuffer.empty[Array[T]]

        for i <- (0 until numberOfImages) do
          val image = (0 until singeImageSize)
            .map(_ => n.fromInt(imageInputStream.readUnsignedByte())).toArray
          imageArray += image

        (imageArray.toArray.take(samples).as2D, labelsTensor)
      }
    }
