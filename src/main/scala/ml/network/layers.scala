package ml.network

import ml.tensors.api._
import ml.tensors.ops._

import scala.reflect.ClassTag

trait Layer[T]:
  val units: Int
  val f: ActivationFunc[T]
  val w: Option[Tensor[T]]
  val b: Option[Tensor[T]]
  val optimizerParams: Option[OptimizerParams[T]]
  
  def withParams(w: Tensor[T], b: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T]

  def apply(x: Tensor[T]): Activation[T]

  def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T]

  override def toString() = 
    s"\n(\nweight = $w,\nbias = $b,\nf = ${f.name},\nunits = $units)"

case class Dense[T: ClassTag: Numeric](
    f: ActivationFunc[T] = ActivationFuncApi.linear[T],
    units: Int = 1,
    w: Option[Tensor[T]] = None,
    b: Option[Tensor[T]] = None,
    optimizerParams: Option[OptimizerParams[T]] = None
) extends Layer[T]:

  override def withParams(w: Tensor[T], b: Tensor[T], optimizerParams: Option[OptimizerParams[T]]): Layer[T] =
    copy(w = Some(w), b = Some(b), optimizerParams = optimizerParams)

  override def apply(x: Tensor[T]): Activation[T] =    
    val z = x * w + b
    val a = f(z)
    Activation(x, z, a)

  override def update(wGradient: Tensor[T], bGradient: Tensor[T], optimizerParams: Option[OptimizerParams[T]] = None): Layer[T] =
    val updatedW = w.map(_ - wGradient)  
    val updatedB = b.map(_ - bGradient)  
    copy(w = updatedW, b = updatedB, optimizerParams = optimizerParams)
