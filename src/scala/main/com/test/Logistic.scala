package main.com.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Encoder, Row, SparkSession}


object Logistic {
//  private val sourceFile = "H:\\LangLearn\\hadoop_projects\\SparkTest\\src\\scala\\main\\com\\resource\\iris.data"
  private val sourceFile ="/Users/apple/Desktop/SparkTest/src/scala/main/com/resource/iris.data"


  case class Iris(features: Vector, label: String)

  def letsLogistic(): Unit = {

    // 开启 spark 会话
    val spark = SparkSession.builder()
      .appName("MlTest")
//      .master("local")
//    8080是管理器 8080是管理器
        .master("spark:/118.89.232.32:7077")
      .getOrCreate()
    // 启动隐式转换系统
    import spark.implicits._

    // 获取 spark 上下文（以后上下文直接用 spark 会话拿）
    val sc = spark.sparkContext

    // 设置解码器
    implicit val encoder: Encoder[String] = ExpressionEncoder()

    // 读取原始数据
    val data = sc.textFile(sourceFile)
      .map(_.split(","))
      .map(p => Iris(Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble), p(4)))
      .toDF()

    // 创建虚拟数据表并搜索
    data.createOrReplaceTempView("iris")
    val df = spark.sql("select * from iris where label != 'Iris-setosa'")
    df.map(t => t(1) + ":" + t(0))(encoder)
      .collect()

    // 获取标签列
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)

    // 获取特征列
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(df)

    // 分成训练集，测试集
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    // 设置逻辑斯特方程参数
    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // 把预测数据转化为字符串
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // 开管道
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))

    // 训练
    val model = pipeline.fit(trainingData)

    // 检查训练结果
    val predictions = model.transform(testData)
    predictions.
      select("predictedLabel", "label", "features", "probability")
      .collect()
      .foreach { case Row(predictedLabel: String, label: String, features: Vector, prob: Vector) => println(s"($label, $features) --> prob=$prob, predictedLabel=$predictedLabel")
      }
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logistic.letsLogistic()
  }
}
