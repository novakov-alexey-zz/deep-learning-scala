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

  def softmax[T: ClassTag: Ordering](using n: Fractional[T]) = new ActivationFunc[T]:
    // stable, using "- max"
    override def apply(x: Tensor[T]): Tensor[T] = 
      // println(x)
      val applied = x.mapRow { row =>
        val max = row.max
        // println(s"max = $max")
        val norm = row.map(v => castFromTo[Double, T](math.exp(n.toDouble(n.minus(v, max))))) 
        // println(s"norm = ${norm.mkString(",")}")
        val sum = norm.sum
        // println(s"sum = $sum")
        norm.map(v => n.div(v, sum))
      }
      //println(applied)
      applied
      
    override def derivative(x: Tensor[T]): Tensor[T] = 
      val sm = apply(x)
      println("sm:\n" + sm)
      println("diag:\n" + sm.flatten.diag)
      println(s"outer:\n" + sm.outer(sm))
      val d = sm.flatten.diag - sm.outer(sm)
      println("d:\n" + d)
      x
    override val name = "softmax"  
  
  def linear[T] = new ActivationFunc[T]:
    override def apply(x: Tensor[T]): Tensor[T] = x
    override def derivative(x: Tensor[T]): Tensor[T] = x
    override val name = "linear"  