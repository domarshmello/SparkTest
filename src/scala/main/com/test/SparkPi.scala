package main.com.test

import java.util.Scanner

import org.apache.spark.sql.SparkSession

import scala.math.random
import org.apache.spark.{SparkConf, SparkContext}

object SparkPi {

  def sparkPi(): Unit = {
    val sparkConf = new SparkConf().setAppName("正经测试")
      //设置Master_IP
      .setMaster("spark://Master:7077")
    //设置driver端的ip,这里是你本机的ip
    //      .setIfMissing("spark.driver.host", "localhost")
    val spark = new SparkContext(sparkConf)
    val slices = 2
    val n = 100000 * slices
    val count = spark.parallelize(1 to n, slices).map { i =>
      val x = Math.random * 2 - 1
      val y = Math.random * 2 - 1
      if (x * x + y * y < 1) 1 else 0
    }.reduce(_ + _)
    println("*****Pi is roughly " + 4.0 * count / n)
    spark.stop()
  }

  def sparkRDD(): Unit = {
    val spark = SparkSession.builder().master("spark://118.89.232.32:7077").appName("JsonReader").getOrCreate()
    val df = spark.read.json("file://118.89.232.32:/usr/local/spark/examples/src/main/resources/people.json")
    df.show()
    spark.close()
  }
}