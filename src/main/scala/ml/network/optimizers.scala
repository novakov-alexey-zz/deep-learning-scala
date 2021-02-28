package ml.network

import ml.tensors.api._
import ml.tensors.ops._
import ml.transformation.castFromTo

import scala.reflect.ClassTag
import scala.math.Numeric.Implicits._

// suported Optimizers
type Adam
type SimpleGD

sealed trait Optimizer[U]:

  def updateWeights[T: ClassTag](
      weights: List[Layer[T]],
      activations: List[Activation[T]],
      error: Tensor[T],
      cfg: OptimizerCfg[T],
      timestep: Int
  )(using n: Fractional[T]): List[Layer[T]]

  def initState[T: ClassTag: Numeric](w: Tensor[T], b: Tensor[T]): Option[OptimizerState[T]] = None

object optimizers:
  given Optimizer[Adam] with        

    override def initState[T: ClassTag: Numeric](w: Tensor[T], b: Tensor[T]): Option[OptimizerState[T]] =
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
          List.empty[Layer[T]],
          error,
          None: Option[Tensor[T]]          
        ) {             
            case (
                  (Layer(w, b, f, u, Some(AdamState(mw, vw, mb, vb))), Activation(x, z, _)),
                  (ls, prevDelta, prevWeight)
                ) =>            
              val delta = (prevWeight match 
                case Some(pw) => prevDelta * pw.T
                case None     => prevDelta
              ) multiply f.derivative(z)        
              val wGradient = c.clip(x.T * delta)
              val bGradient = c.clip(delta).sum
              
              // Adam                        
              val (corrW, weightM, weightV) = correction(wGradient, mw, vw)
              val newWeight = w - corrW

              val (corrB, biasM, biasV) = correction(bGradient.asT, mb, vb)
              val newBias = b - corrB

              val adamState = Some(AdamState(weightM, weightV, biasM, biasV))
              val updated = Layer(newWeight, newBias, f, u, adamState) +: ls              
              (updated, delta, Some(w))
            
            case s => sys.error(s"Adam optimizer require state, but was:\n$s")
        }
        ._1    

  given Optimizer[SimpleGD] with
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
          List.empty[Layer[T]],
          error,
          None: Option[Tensor[T]]
        ) {
          case (
                (l @ Layer(w, b, f, u, s), Activation(x, z, _)),
                (ls, prevDelta, prevWeight)
              ) =>            
            val delta = (prevWeight match {
              case Some(pw) => prevDelta * pw.T
              case None     => prevDelta
            }) multiply f.derivative(z)

            val wGradient = cfg.clip(x.T * delta)
            val bGradient = cfg.clip(delta).sum
            val newWeight = w - (cfg.learningRate * wGradient)
            val newBias = b - (cfg.learningRate * bGradient)
            val updated = l.copy(w = newWeight, b = newBias) +: ls
            (updated, delta, Some(w))
        }
        ._1    

case class OptimizerCfg[T: ClassTag: Fractional](
  learningRate: T,
  clip: GradientClipping[T] = GradientClippingApi.noClipping[T],
  adam: AdamCfg[T]
)

sealed trait OptimizerState[T]

case class AdamState[T](mw: Tensor[T], vw: Tensor[T], mb: Tensor[T], vb: Tensor[T]) extends OptimizerState[T]

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