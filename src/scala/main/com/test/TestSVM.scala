package main.com.test


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{Encoder, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import java.util.Properties

import org.apache.spark.sql.types.{StringType, StructField, StructType}

import org.apache.spark.mllib.optimization.L1Updater


object TestSVM {
  private val sourceFile ="/Users/apple/Desktop/SparkTest/src/scala/main/com/resource/iris.data"

  def SVM(): Unit = {

    // 开启 spark 会话
    val spark = SparkSession.builder()
      .appName("SVMTest")
      .master("local")
      .getOrCreate()

    // 获取 spark 上下文（以后上下文直接用 spark 会话拿）

    val sc = spark.sparkContext
    val data = sc.textFile(sourceFile)
    val parsedData = data.map { line =>
          val parts = line.split(',')
           LabeledPoint(if(parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)
        =="Iris-versicolor") 1.toDouble else
               2.toDouble, Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts
    (2).toDouble,parts(3).toDouble))
          }

    val splits = parsedData.filter { point => point.label != 2 }.randomSplit(
      Array(0.6, 0.4), seed = 11L)

    val training = splits(0).cache()

    val test = splits(1)

    val numIterations = 1000

    val model = SVMWithSGD.train(training, numIterations)


    model.clearThreshold()


    val scoreAndLabels = test.map { point =>
             val score = model.predict(point.features)
             (score, point.label)
           }

    scoreAndLabels.foreach(println)

    model.setThreshold(0.0)

    scoreAndLabels.foreach(println)

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)

    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    val svmAlg = new SVMWithSGD()


    svmAlg.optimizer.
             setNumIterations(2000)
            .setRegParam(0.1)
            .setUpdater(new L1Updater)

    val modelL1 = svmAlg.run(training)

    println("result:")
    parsedData.map{data =>
      val score = modelL1.predict(data.features)

      score + " " + data.label
    }.foreach(println)

  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    SVM()
  }


}
