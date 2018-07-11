package main.com.test

import org.apache.spark.{SparkConf, SparkContext}

object SimpleRDDTest {
  def wordCount(): Unit = {
    val conf = new SparkConf()
      .setAppName("WordCount")
    val sc = new SparkContext(conf)
    val file = sc.textFile("file:///usr/local/spark/README.md")
    val word_count = file
      .flatMap { line => line.split(" ") }
      .map { word => (word, 1) }
      .reduceByKey {
        _ + _
      }
    word_count
      .foreach(word_map => println(s"${word_map._1} : ${word_map._2}"))
  }

  def listCount(): Unit = {
    val conf = new SparkConf().setAppName("WordCount")
    val list = List("hadoop", "spark", "hive", "spark")
    val sc = new SparkContext(conf)
    val rdd = sc.parallelize(list)
  }
}
