package main.com.luxury_taobao

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql.SparkSession

object FinalPreX {

  def forHage(): Unit = {
    val spark = SparkSession.builder()
      .appName("淘宝")
      .master("local")
      .getOrCreate()

    val sc = spark.sparkContext

    println(
      """
        |                     ★
        |                    ／＼
        |                   ／⸛⸛＼
        |                  ／    ＼
        |                 ／ ⸮   ⸛＼
        |                ／     & @＼
        |               ／&⸛ &&  @  ＼
        |              ／⁂⁂  @       ＼
        |             ／i⸛  &     ⸛ ｡⸮＼
        |            ／ ⸮   &  ｡       ＼
        |           ／     &｡i  &    i@i＼
        |          ／  ⸮ ⸛@⸮⁂    &⸛⸛@&⸮  ＼
        |         ／  ⸛   &  ⸛⁂i   ⸮  &⸮  ＼
        |        ／   &     ｡   & ⸛⁂   ⸛   ＼
        |       ／  & ⸮    @     ⸛ i｡   &｡  ＼
        |      ／    @@   ⸮⸛⸮｡      ⁂@  @    ＼
        |     ／   &      &&      i    ⸮i@    ＼
        |    ／  ⁂    @⸛ i⁂i ⸮⸮ & ⁂i  ⸛ i    ⸛ ＼
        |   ／ ｡  @&⁂ ⁂ ⸛⁂   ⸮  @｡  i ⁂i      &&＼
        |  ／  ⁂⁂｡ ｡   @i⸮ @  & ｡⸛ @   ⸛    &   ⸛＼
        | ／   ｡ ⸮i ⸮  ⸮ ｡ @ ⸛      ⁂⸛ &｡  @ @⸮i  ＼
        | ^^^^^^^^^^^^^^^^^^^|  |^^^^^^^^^^^^^^^^^^^
        |                    |  |
      """.stripMargin)

    import spark.implicits._
//数据存放的地址 你看看你放到哪个位置 对应改成自己的
    val train_data = sc.textFile("/usr/local/dbtaobao/dataset/train_do.csv")
    val test_data = sc.textFile("/usr/local/dbtaobao/dataset/train_test.csv")

    val train = train_data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(4).toDouble, Vectors.dense(parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))
    }.toDF()

    val test = test_data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(4).toDouble, Vectors.dense(parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))
    }.toDF()

    val model = new NaiveBayes().fit(train)


    val predictions = model.transform(test)
    predictions.show()


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"测试集准确率为 = ${accuracy * 100}")
  }

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:/hadoop3")
    Logger.getLogger("org").setLevel(Level.ERROR)
    forHage()
  }


}
