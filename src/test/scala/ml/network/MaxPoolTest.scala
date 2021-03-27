package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MaxPoolTest extends AnyFlatSpec with Matchers {
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

  it should "do forward and backward propagation without padding" in {
    val unpadded = Array(
      Array(
        Array(
          Array(3d, 4, 4),
          Array(6d, 7, 7)
        )
      )
    )

    // FORWARD
    // given
    val noPaddingLayer = MaxPool[Double](padding = false).init(image.shape)
    // when
    val noPaddingAct = noPaddingLayer(image)
    // then
    val z = noPaddingAct.z.as4D
    z.shape should ===(noPaddingLayer.shape)
    z.data should ===(unpadded)

    val unpaddedDelta = Array(
      Array(
        Array(
          Array(1d, 2, 3),
          Array(7d, 1, 2)
        )
      )
    )

    val Gradient(unpaddedNextDelta, _, _) =
      noPaddingLayer.backward(noPaddingAct, unpaddedDelta.as4D, None)

    unpaddedNextDelta.as4D.data should ===(
      Array(
        Array(
          Array(
            Array(0d, 0, 0, 0),
            Array(0d, 1, 3, 0),
            Array(0d, 7, 2, 0)
          )
        )
      )
    )
  }

  it should "do forward propagation with padding" in {
    // given
    val padded = Array(
      Array(
        Array(
          Array(3d, 4, 4, 3),
          Array(6d, 7, 7, 3),
          Array(6d, 7, 7, 3)
        )
      )
    )
    val paddedLayer = MaxPool[Double](padding = true).init(image.shape)
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
