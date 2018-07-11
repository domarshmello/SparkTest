package main.com.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{KMeans,KMeansModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Encoder, SparkSession}

object KMeans {

  private val sourceFile = "file:/home/ubuntu/iris.data"

  case class model_instance(features: Vector)

  def HelloKMeans(): Unit = {

    // 开启 spark 会话
    val spark = SparkSession.builder()
      .appName("KMeansTest")
      .master("local")
      .getOrCreate()

    // 启动隐式转换系统
    import spark.implicits._

    // 获取 spark 上下文（以后上下文直接用 spark 会话拿）

    val sc = spark.sparkContext

    // 设置解码器
    implicit val encoder: Encoder[String] = ExpressionEncoder()

    // 读取原始数据
    val df = sc.textFile(sourceFile).map(line => {
      model_instance(Vectors.dense(line.split(",")
        .filter(p => p.matches("\\d*(\\.?)\\d*"))
        .map(_.toDouble)))
    }).toDF()
    //    用于保存训练后模型的KMeansModel类则属于Transformer
    val kmeansmodel = new KMeans()
      .setK(3).setFeaturesCol("features")
      .setPredictionCol("prediction")
      .fit(df)


    // 检查训练结果  KMeansModel作为一个Transformer
    // 提供了一致性的transform()方法，用于将存储在DataFrame中的给定数据集进行整体处理，生成带有预测簇标签的数据集：
    // 使用collect()方法，该方法将DataFrame中所有的数据组织成一个Array对象进行返回

    val results = kmeansmodel.transform(df)

    results.collect().foreach(
      row => {
        println(row(0) + " is predicted as cluster " + row(1))
      })

    kmeansmodel.clusterCenters.foreach(
      center => {
        println("Clustering Center:" + center)
      })

    println(kmeansmodel.computeCost(df))
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    HelloKMeans()
  }

}
