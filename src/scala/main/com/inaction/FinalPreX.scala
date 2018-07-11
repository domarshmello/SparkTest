package main.com.inaction

import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
//哈哥你自己看吧 朴素贝叶斯模型训练准点 95%左右吧  0和1阈值很接近   实际数据有问题  直接匹配拟合到样本数量最多的0的区域里
object FinalPreX {


  def forTheFreedomAndGlory(): Unit = {
    // 启动会话，并获取上下文备用
    val spark = SparkSession.builder()
      .appName("淘宝")
      .master("local")
      .getOrCreate()
    val sc = spark.sparkContext

    // 输出一个霸气的 Logo 代表自己是最强的女人
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

    // 开启 SparkSession 的隐式转换系统
    // 当Scala编译器进行类型匹配时，如果找不到合适的候选，那么隐式转化提供了另外一种途径来告诉编译器如何将当前的类型转换成预期类型
    import spark.implicits._

    // 从硬盘加载数据


    val train_data = sc.textFile("file:/usr/local/dbtaobao/dataset/train_do.csv")
    val test_data = sc.textFile("file:/usr/local/dbtaobao/dataset/train_test.csv")

    // 格式化训练，测试数据
    val train = train_data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(4).toDouble, Vectors.dense(parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))
    }.toDF()


    val test = test_data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(4).toDouble, Vectors.dense(parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))
    }.toDF()

    // 启动朴素贝叶斯模型，并传入训练数据进行训练
    val model = new NaiveBayes().fit(train)

    // 用测试数据对模型进行测试，得到结果数据并输出
    val predictions = model.transform(test)
    predictions.show()

    // 启动评估对象，利用评估对象计算精准度
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"测试集准确率为 = ${accuracy * 100}")

    // 准备数据库连接信息
    val prop = new Properties()
    prop.put("user", "root")
    prop.put("password", "1234")
    prop.put("driver", "com.mysql.jdbc.Driver")

    // 设置表头
    val fields = "label,features,rawPrediction,probability,prediction".split(",")
      .map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    // 启动行编码器，以使类型推断系统正确推断数据类型
    val encoder = RowEncoder(schema)

    // 格式化结果数据
    val rawRDD = predictions.map(x =>
      Row(x(0).toString, x(1).toString, x(2).toString, x(3).toString, x(4).toString)
    )(encoder).rdd
    spark.createDataFrame(rawRDD, schema)
      // 写入数据库
      .write
      .mode("append")
      //如果跑的有问题将下面的useSSl=false改成true  ps:我这里的mysql-connectot-jar包版本低
      .jdbc("jdbc:mysql://localhost:3306/dbtaobao?useSSL=false", "dbtaobao.rebuy", prop)

  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    forTheFreedomAndGlory()
  }


}
