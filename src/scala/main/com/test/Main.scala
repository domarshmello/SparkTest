package main.com.test

import main.com.luxury_taobao.TaobaoReduce
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object Main {

  def main(args: Array[String]): Unit = {
    // 关闭其他输出日志
    //System.setProperty("hadoop.home.dir", "D:/hadoop3")
    Logger.getLogger("org").setLevel(Level.ERROR)
    TaobaoReduce.provinceCollect()
  }

  def simple(): Unit = {
    val conf = new SparkConf().setAppName("Combine").setMaster("local")
    val sc = new SparkContext(conf)
    val data = sc.parallelize(Array(("company-1", 92), ("company-1", 92), ("company-1", 82),
      ("company-2", 78), ("company-2", 96), ("company-2", 85), ("company-3", 88),
      ("company-3", 94), ("company-3", 80)), 3)
    val res = data.combineByKey(
      income => (income, 1),
      (acc: (Int, Int), income) => (acc._1 + income, acc._2 + 1),
      (acc1: (Int, Int), acc2: (Int, Int)) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
    ).foreach(println)
//      .map { case (key, value) => (key, value._1, value._1 / value._2.toFloat) }
//    res.repartition(1).saveAsTextFile("./result")
  }
}
