package main.com.inaction

import java.io.File

object PathGetter {

  def main(args: Array[String]): Unit = {
    pathGetter()
  }

  private def pathGetter(): Unit = {
    println(System.getProperty("user.dir"))
    val directory = new File(".")
    println(directory.getCanonicalPath)
    println(directory.getAbsolutePath)
    val directory2 = new File("../resource")
    println(directory2.getCanonicalPath)
    println(directory2.getAbsolutePath)
  }
}
