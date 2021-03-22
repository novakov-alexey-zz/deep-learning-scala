package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

import optimizers.given_Optimizer_Adam as adam
import inits.given_ParamsInitializer_T_HeNormal as normal

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MaxPoolTest extends AnyFlatSpec with Matchers {
  it should "do forward propagation" in {
    val image = Tensor4D(Array(
        Array(
            Array(
                Array(1d, 2, 3, 3), 
                Array(2d, 3, 4, 3), 
                Array(5d, 6, 7, 3) 
            ),
    )))    
    val expected = Array(
        Array (
            Array(
                Array(3d, 4, 4),
                Array(6d, 7, 7),
            )
        )
    )
    val l = MaxPool[Double]().init(List(1, 3, 2, 3), normal, adam)
    val a =l(image)

    a.z.as4D.data should ===(expected)
  }
}
