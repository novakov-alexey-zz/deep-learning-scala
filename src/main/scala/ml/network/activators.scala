package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._

import math.Ordering.Implicits.infixOrderingOps
import scala.reflect.ClassTag

trait ActivationFunc[T]:
  val name: String
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

object ActivationFuncApi:
  def relu[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double, T](math.max(0, n.toDouble(t))))      

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => if t < n.zero then n.zero else n.one)

    override val name = "relu"
  
  def sigmoid[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double,T](1 / (1 + math.exp(-n.toDouble(t)))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double, T](
        math.exp(-n.toDouble(t)) / math.pow(1 + math.exp(-n.toDouble(t)), 2)
      ))
    
    override val name = "sigmoid"  

  def linear[T] = new ActivationFunc[T]:
    override def apply(x: Tensor[T]): Tensor[T] = x
    override def derivative(x: Tensor[T]): Tensor[T] = x
    override val name = "linear"  