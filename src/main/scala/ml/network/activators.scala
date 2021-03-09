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
    
    override def apply(x: Tensor[T]): Tensor[T] =       
      val applied = x.mapRow { row =>
        val max = row.max        
        val expNorm = row.map(v => castFromTo[Double, T](math.exp(n.toDouble(n.minus(v, max)))))         
        val sum = expNorm.sum        
        expNorm.map(v => n.div(v, sum))
      }
      
      val appliedSum = applied.sumCols
        .map(v => castFromTo[Double, T](if n.toDouble(v).abs - 0.4E-15d > 1d then n.toDouble(v) else 1d))      
      val totalSum = appliedSum.sumRows.as0D.data      
      assert(totalSum == x.length, 
        s"Softmax distribution sum is not equal to 1 at some activation, but\n${appliedSum}")
      applied
          
    override def derivative(x: Tensor[T]): Tensor[T] =       
      val sm = apply(x)      
      sm multiply (n.one - sm)

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