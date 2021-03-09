package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag

trait ParamsInitializer[A, B]:  

  def weights(rows: Int, cols: Int): Tensor[A]

  def biases(length: Int): Tensor[A]

// support Initializers
type RandomUniform
type HeNormal

object inits:
  def zeros[T: ClassTag](length: Int)(using n: Numeric[T]): Tensor[T] =    
    Tensor1D(Array.fill(length)(n.zero))

  given [T: Numeric: ClassTag]: ParamsInitializer[T, RandomUniform] with    
    
    def gen: T = 
      castFromTo[Double, T](math.random().toDouble + 0.001d)

    override def weights(rows: Int, cols: Int): Tensor[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen)))

    override def biases(length: Int): Tensor[T] = 
      zeros(length)

  given [T: ClassTag: Numeric]: ParamsInitializer[T, HeNormal] with    
    val rnd = new java.util.Random() 

    def gen(lenght: Int): T = 
      castFromTo[Double, T]{
        val v = rnd.nextGaussian + 0.001d
        v * math.sqrt(2d / lenght.toDouble)
      }

    override def weights(rows: Int, cols: Int): Tensor[T] =
      Tensor2D(Array.fill(rows)(Array.fill[T](cols)(gen(rows))))

    override def biases(length: Int): Tensor[T] = 
      zeros(length)