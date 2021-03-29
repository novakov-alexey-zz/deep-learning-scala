package ml.network

import ml.tensors.api._
import ml.tensors.ops._
import ml.transformation.castFromTo

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

// suported Optimizers
type Adam
type StandardGD
type Stub

trait Optimizer[U]:

  def updateWeights[T: ClassTag](
      layers: List[Layer[T]],
      activations: List[Activation[T]],
      error: Tensor[T],
      cfg: OptimizerCfg[T],
      timestep: Int
  )(using n: Fractional[T]): List[Layer[T]]

  def init[T: ClassTag: Numeric](w: Tensor[T], b: Tensor[T]): Option[OptimizerParams[T]] = None

object optimizers:
  given Optimizer[Stub] with
    override def updateWeights[T: ClassTag](
        layers: List[Layer[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        c: OptimizerCfg[T],
        timestep: Int
    )(using n: Fractional[T]): List[Layer[T]] = layers

  given Optimizer[Adam] with        

    override def init[T: ClassTag: Numeric](w: Tensor[T], b: Tensor[T]): Option[OptimizerParams[T]] =
      Some(AdamState[T](w.zero, w.zero, b.zero, b.zero))

    override def updateWeights[T: ClassTag](
        layers: List[Layer[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        c: OptimizerCfg[T],
        timestep: Int
    )(using n: Fractional[T]): List[Layer[T]] =
      val AdamCfg(b1, b2, eps) = c.adam        

      def correction(gradient: Tensor[T], m: Tensor[T], v: Tensor[T]) =
        val mt = (b1 * m) + ((n.one - b1) * gradient)
        val vt = (b2 * v) + ((n.one - b2) * gradient.sqr)        
        val mHat = mt :/ (n.one - (b1 ** timestep))
        val vHat = vt :/ (n.one - (b2 ** timestep))            

        val corr = c.learningRate * (mHat / (vHat.sqrt + eps))
        (corr, mt, vt)
      
      layers
        .zip(activations)
        .foldRight(
          ListBuffer.empty[Layer[T]],
          error,
          None: Option[Tensor[T]]          
        ) {             
            case (
                  (layer, a),
                  (ls, prevDelta, prevWeight)
                ) =>                                        
              val Gradient(delta, wOpt, bOpt) = layer.backward(a, prevDelta, prevWeight)
              val (updated, weight) = (layer, wOpt, bOpt) match
                case (o: Optimizable[T], Some(w), Some(b)) =>
                  // Adam                        
                  o.optimizerParams match
                    case Some(AdamState(mw, vw, mb, vb)) =>
                      val wGradient = c.clip(w)
                      val bGradient = c.clip(b).sumRows
                      val batchSize = n.fromInt(a.x.length)                                   
                      val (corrW, weightM, weightV) = correction(wGradient :/ batchSize, mw, vw)                  
                      val (corrB, biasM, biasV) = correction(bGradient :/ batchSize, mb, vb)                  
                      val adamState = Some(AdamState(weightM, weightV, biasM, biasV))
                      (o.update(corrW, corrB, adamState), o.w)                    
                    case _ => 
                      (layer, None) // does nothing if Adam state is not set
                case _ => 
                  (layer, None) // does nothing if one of the params is empty 
              (updated +: ls, delta, weight)                        
        }
        ._1.toList    

  given Optimizer[StandardGD] with

    override def updateWeights[T: ClassTag](
        layers: List[Layer[T]],
        activations: List[Activation[T]],
        error: Tensor[T],
        cfg: OptimizerCfg[T],
        timestep: Int
    )(using n: Fractional[T]): List[Layer[T]] =      
      layers
        .zip(activations)
        .foldRight(
          ListBuffer.empty[Layer[T]],
          error,
          None: Option[Tensor[T]]
        ) {
          case (
                (layer, a),
                (ls, prevDelta, prevWeight)
              ) =>            
            val Gradient(delta, w, b) = layer.backward(a, prevDelta, prevWeight)
            val (updated, weight) = (layer, w, b) match
              case (o: Optimizable[T], Some(w), Some(b)) =>
                val batchSize = n.fromInt(a.x.length)
                val wGradient = cfg.clip(w)
                val bGradient = cfg.clip(b).sumRows :/ batchSize
                val corrW = cfg.learningRate * wGradient
                val corrB = cfg.learningRate * bGradient
                (o.update(corrW, corrB), o.w)
              case _ => 
                (layer, None)
            (updated +: ls, delta, weight)
        }
        ._1.toList    

case class OptimizerCfg[T: ClassTag: Fractional](
  learningRate: T,
  clip: GradientClipping[T] = GradientClippingApi.noClipping[T],
  adam: AdamCfg[T]
)

sealed trait OptimizerParams[T]

case class AdamState[T](mw: Tensor[T], vw: Tensor[T], mb: Tensor[T], vb: Tensor[T]) extends OptimizerParams[T]

case class AdamCfg[T: ClassTag](b1: T, b2: T, eps: T)

object AdamCfg:

  def default[T: ClassTag]: AdamCfg[T] =
    AdamCfg[T](
      castFromTo[Double, T](0.9),
      castFromTo[Double, T](0.999),
      castFromTo[Double, T](10E-8)
    )

trait GradientClipping[T] extends (Tensor[T] => Tensor[T]) 

object GradientClippingApi:
  def clipByValue[T: Fractional: ClassTag](value: T): GradientClipping[T] = 
    _.clipInRange(-value, value)

  def noClipping[T]: GradientClipping[T] = t => t