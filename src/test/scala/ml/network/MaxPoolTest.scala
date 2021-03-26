package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MaxPoolTest extends AnyFlatSpec with Matchers {
  it should "do forward and backward propagation" in {
    val image = Tensor4D(
      Array(
        Array(
          Array(
            Array(1d, 2, 3, 3),
            Array(2d, 3, 4, 3),
            Array(5d, 6, 7, 3)
          )
        )
      )
    )
    val padded = Array(
      Array(
        Array(
          Array(3d, 4, 4, 3),
          Array(6d, 7, 7, 3),
          Array(6d, 7, 7, 3)
        )
      )
    )

    val unpadded = Array(
      Array(
        Array(
          Array(3d, 4, 4),
          Array(6d, 7, 7)
        )
      )
    )

    // FORWARD
    val prevShape = List(1, 1, 3, 4)
    // given
    val unpaddedLayer = MaxPool[Double](padding = false).init(prevShape)
    // when
    val unpaddedPooling = unpaddedLayer(image).z.as4D
    // then
    unpaddedPooling.shape should ===(unpaddedLayer.shape)

    // given
    val paddedLayer = MaxPool[Double](padding = true).init(prevShape)
    // when
    val a = paddedLayer(image)

    // then
    a.z.shape should ===(paddedLayer.shape)
    a.z.as4D.data should ===(padded)

    // BACKWARD
    // given
    val delta = Array(
      Array(
        Array(
          Array(1d, 2, 3, 1),
          Array(7d, 1, 2, 1),
          Array(1d, 1, 2, 1)
        )
      )
    )
    // when
    val Gradient(nextDelta, w, b) = paddedLayer.backward(a, delta.as4D, None)

    //then
    nextDelta.as4D.shape4D should ===(a.x.as4D.shape4D)
    w should ===(None)
    b should ===(None)

    withClue(s"$nextDelta") {
      nextDelta.as4D.data should ===(
        Array(
          Array(
            Array(
              Array(0d, 0, 0, 1),
              Array(0d, 1, 3, 1),
              Array(0d, 1, 2, 1)
            )
          )
        )
      )
    }
  }
}
