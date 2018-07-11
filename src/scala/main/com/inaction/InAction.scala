package main.com.inaction

import org.apache.log4j.{Level, Logger}

object InAction {
  def main(args: Array[String]): Unit = {
    // 准备环境
    System.setProperty("hadoop.home.dir", "D:/hadoop3")
    Logger.getLogger("org").setLevel(Level.ERROR)
    MapReduce.readDataFromCsv()
  }
}
