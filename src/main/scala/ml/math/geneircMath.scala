package ml.math

import scala.reflect.ClassTag
import ml.transformation.castFromTo

object generic:
  def exp[T: ClassTag](v: T)(using n: Numeric[T]): T =
    castFromTo[Double, T](math.exp(n.toDouble(v)))
  
  def pow[T: ClassTag](x: T, y: T)(using n: Numeric[T]): T =    
    castFromTo[Double, T](math.pow(n.toDouble(x), n.toDouble(y)))

  def max[T: ClassTag](x: T, y: T)(using n: Numeric[T]): T =
    castFromTo[Double, T](math.max(n.toDouble(x), n.toDouble(y)))
    
  def log[T: ClassTag](x: T)(using n: Numeric[T]): T =
    castFromTo[Double, T](math.log(n.toDouble(x)))