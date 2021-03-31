package ml.network

import ml.transformation.castFromTo
import ml.tensors.api._
import ml.tensors.ops._
import ml.math.generic._

import math.Ordering.Implicits.infixOrderingOps
import math.Fractional.Implicits.infixFractionalOps
import scala.reflect.ClassTag

trait ActivationFunc[T]:
  val name: String
  def apply(x: Tensor[T]): Tensor[T]
  def derivative(x: Tensor[T]): Tensor[T]

object ActivationFuncApi:
  def relu[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => if t < n.zero then n.zero else t)

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => if t < n.zero then n.zero else n.one)

    override val name = "relu"
  
  def leakyRelu[T: ClassTag](using n: Numeric[T]) = new ActivationFunc[T]:
    val scaler = castFromTo[Double, T](0.01)
    
    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t =>  if t < n.zero then n.times(scaler, t) else t)

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => if t < n.zero then scaler else n.one)

    override val name = "leakyRelu"
  
  def sigmoid[T: ClassTag](using n: Fractional[T]) = new ActivationFunc[T]:

    override def apply(x: Tensor[T]): Tensor[T] =
      x.map(t => n.one / (n.one + exp(-t)))

    override def derivative(x: Tensor[T]): Tensor[T] =
      x.map(t => exp(-t) / pow(n.one + exp(-t), n.fromInt(2)))
    
    override val name = "sigmoid"  

  def softmax[T: ClassTag: Ordering](using n: Fractional[T]) = new ActivationFunc[T]:
    val toleration = castFromTo[Double, T](0.9E-6d)

    override def apply(x: Tensor[T]): Tensor[T] =      
      val applied = x.mapRow { row =>
        val max = row.max        
        val expNorm = row.map(v => exp(v - max))         
        val sum = expNorm.sum        
        expNorm.map(_ / sum)
      }

      val appliedSum = applied.sumCols.map(
        v => 
          if v.abs - toleration > n.one 
          then v 
          else n.one
      )
      val totalSum = appliedSum.sumRows.as1D.data.head
      assert(totalSum == x.length, 
        s"Softmax distribution sum is not equal to 1 at some activation, but\n${appliedSum}")
      applied
          
    override def derivative(x: Tensor[T]): Tensor[T] =       
      val sm = apply(x)      
      sm.multiply(n.one - sm)

    // override def derivative(x: Tensor[T]): Tensor[T] = 
      // println(s"derivative x:\n$x")
    //   val sm = apply(x)
    //   sm.mapRow { row =>
    //     val t = Tensor1D(row)        
    //     val dxDs = t.diag - (t * t)
    //     dxDs.sumRows.as1D.data                
    //   }      
      
    override val name = "softmax"  
  
  def linear[T] = new ActivationFunc[T]:
    override def apply(x: Tensor[T]): Tensor[T] = x
    override def derivative(x: Tensor[T]): Tensor[T] = x
    override val name = "linear"  