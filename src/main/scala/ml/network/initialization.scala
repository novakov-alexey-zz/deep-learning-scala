package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops.as4D

import scala.reflect.ClassTag
import java.util.Random

trait ParamsInitializer[A, B]:

  def weights(rows: Int, cols: Int): Tensor2D[A]

  def biases(length: Int): Tensor1D[A]

  def weights4D(
      shape: List[Int]
  )(using c: ClassTag[A], n: Numeric[A]): Tensor4D[A] =
    val tensors :: cubes :: rows :: cols :: Nil = shape
    (0 until tensors)
      .map(_ => (0 until cubes).toArray.map(_ => weights(rows, cols)))
      .toArray
      .as4D

// support Initializers
type RandomUniform
type HeNormal

object inits:
  def zeros[T: ClassTag](length: Int)(using n: Numeric[T]): Tensor1D[T] =
    Tensor1D(Array.fill(length)(n.zero))

  given [T: Numeric: ClassTag]: ParamsInitializer[T, RandomUniform] with

    def gen: T =
      castFromTo[Double, T](math.random().toDouble + 0.001d)

    override def weights(rows: Int, cols: Int): Tensor2D[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen)))

    override def biases(length: Int): Tensor1D[T] =
      zeros(length)

  given [T: ClassTag: Numeric]: ParamsInitializer[T, HeNormal] with
    val rnd = new Random()

    def gen(lenght: Int): T =
      castFromTo[Double, T] {
        val v = rnd.nextGaussian + 0.001d
        v * math.sqrt(2d / lenght.toDouble)
      }

    override def weights(rows: Int, cols: Int): Tensor2D[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen(rows))))

    override def biases(length: Int): Tensor1D[T] =
      zeros(length)
