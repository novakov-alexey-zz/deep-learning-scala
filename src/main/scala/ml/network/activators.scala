package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag

trait ActivationFunc[T] extends (Tensor[T] => Tensor[T]):
  val name: String
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

object ActivationFuncApi:
  def relu[T: ClassTag](using n: Fractional[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double, T](math.max(0, n.toDouble(t))))      

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double, T](if n.toDouble(t) < 0 then 0 else 1))

    override val name = "relu"
  
  def sigmoid[T: ClassTag](using n: Fractional[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double,T](1 / (1 + math.exp(-n.toDouble(t)))))

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => castFromTo[Double, T](
          math.exp(-n.toDouble(t)) / math.pow(1 + math.exp(-n.toDouble(t)), 2)
        ))
    
    override val name = "sigmoid"  

  def noActivation[T] = new ActivationFunc[T]:
    override def apply(x: Tensor[T]): Tensor[T] = x
    override def derivative(x: Tensor[T]): Tensor[T] = x
    override val name = "no-activation"  