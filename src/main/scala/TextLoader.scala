import converter.{transform, transformArr}

import java.io.File
import java.nio.file.Path
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.Using

object TextLoader {
  val defaultDelimiter: String = ","

  def apply(rows: String*): TextLoader = {
    TextLoader(data = rows.toArray.map(_.split(defaultDelimiter).toArray))
  }
}

case class TextLoader(
    path: Path = new File("data.csv").toPath,
    header: Boolean = true,
    delimiter: String = TextLoader.defaultDelimiter,
    data: Array[Array[String]] = Array.empty[Array[String]]
) {

  def load(): TextLoader = copy(
    data = Using.resource(Source.fromFile(path.toFile)) { s =>
      val lines = s.getLines()
      (if (header && lines.nonEmpty) lines.toArray.tail else lines.toArray)
        .map(_.split(delimiter))
    }
  )

  def cols[T: ClassTag](range: (Int, Int)): Tensor2D[T] =
    transform[T](Tensor2D.slice(data, None, Some(range)))

  def col[T: ClassTag](i: Int): Tensor1D[T] = {
    val col = Tensor2D.col(data, i)
    Tensor1D(transformArr[T](col))
  }
}
