package main.com.test

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession


object KMeans2 {

  private val sourceFile = "H:\\LangLearn\\hadoop_projects\\SparkTest\\src\\scala\\main\\com\\resource\\iris.data"

  case class model_instance(features: Vector)

  def HelloKMeans(): Unit = {
    val session = SparkSession.builder()
      .master("local")
      .appName("KMeans")
      .getOrCreate()

    import session.implicits._

    val context = session.sparkContext
    //    val sqlContext = session.sqlContext
    //    import sqlContext.implicits._

    val rawData = context.textFile(sourceFile)

    val df = rawData.map(line => {
      model_instance(Vectors.dense(line.split(",")
        .filter(p => p.matches("\\d*(\\.?)\\d*"))
        .map(_.toDouble)))
    }).toDF()

    val kmeansModel = new KMeans()
      .setK(3)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .fit(df)

    val results = kmeansModel.transform(df)

    results.collect().foreach(
      row => {
        println(row(0) + " is predicted as cluster " + row(1))
      })

    kmeansModel.clusterCenters.foreach(
      center => {
        println("Clustering Center:" + center)
      })

    println(kmeansModel.computeCost(df))
  }

  def main(args: Array[String]): Unit = {
    HelloKMeans()
  }
}
