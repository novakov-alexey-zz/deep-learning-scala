import scala.util.Using
import java.io.File
import java.io.PrintWriter

def store(filename: String, header: String, data: List[List[String]]) =    
  Using.resource(new PrintWriter(new File(filename))) { w =>
    w.write(header)
    data.foreach { row =>      
    w.write(s"\n${row.mkString(",")}")        
    }
  }