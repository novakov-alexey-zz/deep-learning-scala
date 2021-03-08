package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag

trait Metric[T]:
  val name: String

  // number of matched predictions versus actual labels
  def matches(
      actual: Tensor[T],
      predicted: Tensor[T]
  ): Int

  def average(count: Int, correct: Int): Double =
    correct.toDouble / count

  def apply(actual: Tensor[T], predicted: Tensor[T]): Double =
    val correct = matches(actual, predicted)
    average(actual.length, correct)

object MetricApi:
  // def argMax[T: ClassTag](t: Tensor2D[T])(using n: Numeric[T]) =
  //   Tensor1D(t.data.map { row => n.fromInt(row.indices.maxBy(row))} )

  def predictedToBinary[T](v: T)(using n: Numeric[T]): T =
    if n.toDouble(v) > 0.5 then n.one else n.zero

  def accuracyBinaryClassification[T: ClassTag: Fractional] = new Metric[T]:
    val name = "accuracy"

    def matches(
        actual: Tensor[T],
        predicted: Tensor[T]
    ): Int =      
        val predictedNormalized = predicted.map(predictedToBinary)
        actual.equalRows(predictedNormalized)