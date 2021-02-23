package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag

trait Metric[T]:
  val name: String

  def calculate(
      actual: Tensor[T],
      predicted: Tensor[T]
  ): Int

  def average(count: Int, correct: Int): Double =
    correct.toDouble / count

  def apply(actual: Tensor[T], predicted: Tensor[T]): Double =
    val correct = calculate(actual, predicted)
    average(actual.length, correct)

trait MetricApi:
  def predictedToBinary[T](v: T)(using n: Fractional[T]): T =
    if n.toDouble(v) > 0.5 then n.one else n.zero

  def accuracyBinaryClassification[T: ClassTag: Fractional] = new Metric[T]:
    val name = "accuracy"

    def calculate(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): Int =      
        val predictedNormalized = predicted.map(predictedToBinary)
        actual.equalRows(predictedNormalized)      

object MetricApi extends MetricApi