package examples

import org.scalatest.flatspec.AnyFlatSpec

class LinearRegSmokeTest extends AnyFlatSpec {
  it should "run linear regression example without a fail" in {
    lrTest()
  }
}
