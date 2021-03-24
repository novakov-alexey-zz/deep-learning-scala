package ml.tensors.api

import scala.reflect.ClassTag

case class NDArray[T: ClassTag: Numeric](data: Array[Any], shape: List[Int]):  
  private def printArray(a: Array[Any], level: Int = 1): Array[String] = 
    a.map { e =>
      e match 
        case ar: Array[Any] =>
          val start = s"\n${" " * level}["
          val body = printArray(ar, level + 1).mkString(",")          
          val end = if body.last == ']' then s"\n${" " * level}]" else "]"
          s"$start$body$end"
        case _ => s"$e"
    }     

  override def toString: String = 
    val str = printArray(data).mkString(", ")
    "[" + str + (if str.last == ']' then "\n" else "") + "]"
    

object NDArray:
  def init[T](shape: List[Int], v: T)(using n: Numeric[T]): Array[Any] =
    shape match 
      case Nil => Array(v)
      case h :: Nil =>  Array.fill(h)(v)
      case h :: t => Array.fill(h)(init(t, v))

  def zeros[T: ClassTag](shape: Int*)(using n: Numeric[T]): NDArray[T] =     
    NDArray[T](init(shape.toList, n.zero), shape.toList)

  def ones[T: ClassTag](shape: Int*)(using n: Numeric[T]): NDArray[T] =     
    NDArray[T](init(shape.toList, n.one), shape.toList)

extension [T: ClassTag: Numeric](a: NDArray[T])
  def reshape(shape: Int*): NDArray[T] =
    val newShape = shape.toList
    assert(a.shape.reduce(_ * _) == newShape.reduce(_ * _), s"Current shape ${a.shape} does not fit new shape = $shape")
    def group(ar: Array[Any], shape: List[Int]): Array[Any] =
      shape match
        case h :: Nil => ar.grouped(h).toArray
        case h :: t => group(ar.grouped(h).toArray, t)
        case _ => ar
        
    NDArray[T](group(a.data, newShape.reverse), newShape)      


@main 
def test =  
  val ones = NDArray.ones[Int](16)
  println(ones)  
  println(ones.reshape(2, 2, 2, 2))
  
  NDArray[Int](Array(Array(Array(1))), List(1,1,1))