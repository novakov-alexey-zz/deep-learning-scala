import ml.network.Model

import scala.util.Using

import java.io.File
import java.io.PrintWriter
import java.nio.file.Path

def store(filename: String, header: String, data: List[List[String]]) =    
  Using.resource(new PrintWriter(new File(filename))) { w =>
    w.write(header)
    data.foreach { row =>      
      w.write(s"\n${row.mkString(",")}")
    }
  }

def storeMetrics[T](model: Model[T], path: Path) =
  val values = model.metricValues
  val header = s"epoch,loss,${values.map(_._1.name).mkString(",")}"
  val acc = values.headOption.map(_._2).getOrElse(Nil)
  val lrData = model.history.losses.zip(acc).zipWithIndex.map { 
    case ((loss, acc), epoch) => List(epoch.toString, loss.toString, acc.toString)      
  } 
  store(path.toString, header, lrData) 