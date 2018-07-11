package main.com.test

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Encoder, Row, SparkSession}

object MLIibTest {

  def mLTest(): Unit = {
    // 创建连接
    val spark = SparkSession.builder().
      appName("MlTest").
      getOrCreate()

    // 设置数据集
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)))
      .toDF("id", "text", "label")

    // 设置训练流程
    val tokenizer = new Tokenizer().
      setInputCol("text").
      setOutputCol("words")
    val hashingTF = new HashingTF().
      setNumFeatures(1000).
      setInputCol(tokenizer.getOutputCol).
      setOutputCol("features")
    val lr = new LogisticRegression().
      setMaxIter(10).
      setRegParam(0.01)


    val pipeline = new Pipeline().
      setStages(Array(tokenizer, hashingTF, lr))

    val model = pipeline.fit(training)

    // 测试
    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark a"),
      (7L, "apache hadoop")))
      .toDF("id", "text")

    // 转换
    model.transform(test).
      select("id", "text", "probability", "prediction").
      collect().
      foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }
  }

  def mLTestWithFeature(): Unit = {
    val spark = SparkSession.builder().
      appName("MLTest").
      getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0, "I heard about Spark and I love Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")))
      .toDF("label", "sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")

    val wordsData = tokenizer.transform(sentenceData)

    wordsData.show(false)

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(2000)

    val featuredData = hashingTF.transform(wordsData)

    featuredData.select("rawFeatures")
      .show(false)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val idfModel = idf.fit(featuredData)

    val rescaledData = idfModel.transform(featuredData)

    // 特征，权重表
    rescaledData.select("features", "label")
      .take(3)
      .foreach(println)

  }

  def testCountVectorizer(): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("MlTest")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))))
      .toDF("id", "words")

    val cvModel: CountVectorizerModel = new CountVectorizer().
      setInputCol("words").
      setOutputCol("features").
      setVocabSize(3).
      setMinDF(2).
      fit(df)

    println(cvModel.vocabulary)

    cvModel.transform(df).show(false)
    implicit val encoder: Encoder[String] = ExpressionEncoder()
    val cvm = new CountVectorizerModel(Array("a", "b", "c")).
      setInputCol("words").
      setOutputCol("features")
    cvm.transform(df).select("features").show()
  }


}
