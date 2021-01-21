// scala 2.13.3

import $file.tensor
import $file.converter

import tensor._
import converter._

import reflect.runtime.universe._

import java.nio.file.Path
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.Using

case class TextLoader(
    path: Path,
    header: Boolean = true,
    delimiter: String = ",",
    data: Array[Array[String]] = Array.empty[Array[String]]
) {

  def load(): TextLoader = copy(
    data = Using.resource(Source.fromFile(path.toFile)) { s =>
      val lines = s.getLines()//.take(10) // TODO remove this take(10)
      (if (header && lines.nonEmpty) lines.toArray.tail else lines.toArray)
        .map(_.split(delimiter))
    }
  )

  def cols[T: ClassTag: TypeTag](range: (Int, Int)): Tensor2D[T] =
    transform[T](Tensor2D.slice(data, None, Some(range)))

  def col[T: ClassTag: TypeTag](i: Int): Tensor1D[T] = {
    val col = Tensor2D.col(data, i)
    Tensor1D(transformArr[T](col))
  }
}
