package main.com.newsPredictions

import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import scala.reflect.io.File
/**
  *
  * data from   https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
  * 9. num_imgs：图片数量 10. num_videos：视频数量  11. average_token_length：内容中单词的平均长度
  * 46. global_rate_positive_words：内容中正面词语的比率 47. global_rate_negative_words：内容中负面词语的比率 ---统计
  * 50。 avg_positive_polarity：平均 正面词的极性    57. title_sentiment_polarity：标题极性
  * Created by apple on 2018/6/13.
  */
object newsDo {
  // 输出一个霸气的 Logo 代表自己是最爱大数据的marshmello
  println(
    """
      |              .,;>>%%%%%>>;,.
      |           .>%%%%%%%%%%%%%%%%%%%%>,.
      |         .>%%%%%%%%%%%%%%%%%%>>,%%%%%%;,.
      |       .>>>>%%%%%%%%%%%%%>>,%%%%%%%%%%%%,>>%%,.
      |     .>>%>>>>%%%%%%%%%>>,%%%%%%%%%%%%%%%%%,>>%%%%%,.
      |   .>>%%%%%>>%%%%>>,%%>>%%%%%%%%%%%%%%%%%%%%,>>%%%%%%%,
      |  .>>%%%%%%%%%%>>,%%%%%%>>%%%%%%%%%%%%%%%%%%,>>%%%%%%%%%%.
      | .>>%%%%%%%%%%>>,>>>>%%%%%%%%%%'..`%%%%%%%%,;>>%%%%%%%%%>%%.
      |.>>%%%>>>%%%%%>,%%%%%%%%%%%%%%.%%%,`%%%%%%,;>>%%%%%%%%>>>%%%%.
      |>>%%>%>>>%>%%%>,%%%%%>>%%%%%%%%%%%%%`%%%%%%,>%%%%%%%>>>>%%%%%%%.
      |>>%>>>%%>>>%%%%>,%>>>%%%%%%%%%%%%%%%%`%%%%%%%%%%%%%%%%%%%%%%%%%%.
      |>>%%%%%%%%%%%%%%,>%%%%%%%%%%%%%%%%%%%'%%%,>>%%%%%%%%%%%%%%%%%%%%%.
      |>>%%%%%%%%%%%%%%%,>%%%>>>%%%%%%%%%%%%%%%,>>%%%%%%%%>>>>%%%%%%%%%%%.
      |>>%%%%%%%%;%;%;%%;,%>>>>%%%%%%%%%%%%%%%,>>>%%%%%%>>;";>>%%%%%%%%%%%%.
      |`>%%%%%%%%%;%;;;%;%,>%%%%%%%%%>>%%%%%%%%,>>>%%%%%%%%%%%%%%%%%%%%%%%%%%.
      | >>%%%%%%%%%,;;;;;%%>,%%%%%%%%>>>>%%%%%%%%,>>%%%%%%%%%%%%%%%%%%%%%%%%%%%.
      | `>>%%%%%%%%%,%;;;;%%%>,%%%%%%%%>>>>%%%%%%%%,>%%%%%%'%%%%%%%%%%%%%%%%%%%>>.
      |  `>>%%%%%%%%%%>,;;%%%%%>>,%%%%%%%%>>%%%%%%';;;>%%%%%,`%%%%%%%%%%%%%%%>>%%>.
      |   >>>%%%%%%%%%%>> %%%%%%%%>>,%%%%>>>%%%%%';;;;;;>>,%%%,`%     `;>%%%%%%>>%%
      |   `>>%%%%%%%%%%>> %%%%%%%%%>>>>>>>>;;;;'.;;;;;>>%%'  `%%'          ;>%%%%%>
      |    >>%%%%%%%%%>>; %%%%%%%%>>;;;;;;''    ;;;;;>>%%%                   ;>%%%%
      |    `>>%%%%%%%>>>, %%%%%%%%%>>;;'        ;;;;>>%%%'                    ;>%%%
      |     >>%%%%%%>>>':.%%%%%%%%%%>>;        .;;;>>%%%%                    ;>%%%'
      |     `>>%%%%%>>> ::`%%%%%%%%%%>>;.      ;;;>>%%%%'                   ;>%%%'
      |      `>>%%%%>>> `:::`%%%%%%%%%%>;.     ;;>>%%%%%                   ;>%%'
      |       `>>%%%%>>, `::::`%%%%%%%%%%>,   .;>>%%%%%'                   ;>%'
      |        `>>%%%%>>, `:::::`%%%%%%%%%>>. ;;>%%%%%%                    ;>%,
      |         `>>%%%%>>, :::::::`>>>%%%%>>> ;;>%%%%%'                     ;>%,
      |          `>>%%%%>>,::::::,>>>>>>>>>>' ;;>%%%%%                       ;%%,
      |            >>%%%%>>,:::,%%>>>>>>>>'   ;>%%%%%.                        ;%%
      |             >>%%%%>>``%%%%%>>>>>'     `>%%%%%%.
      |             >>%%%%>> `@@a%%%%%%'     .%%%%%%%%%.
      |             `a@@a%@'    `%a@@'       `a@@a%a@@a'
      |                    marshmello使用机器学习预测新闻热度
    """.stripMargin)


  private val trainFile = "file:/Users/apple/Desktop/SparkTest/src/scala/main/com/resource/在线新闻数据/train_do.csv"
  private val testFile = "file:/Users/apple/Desktop/SparkTest/src/scala/main/com/resource/在线新闻数据/train_test.csv"
  // 模型保存文件夹
  private val modelSavePath = "/Users/apple/Desktop/SparkTest/src/scala/main/com/resource/mod"

  Logger.getLogger("org").setLevel(Level.ERROR)

  private val spark = SparkSession.builder()
    .master("local")
    .appName("TestNews")
    .getOrCreate()

  import spark.implicits._

  private val rawTrainSet = spark.read.csv(trainFile).toDF()
  private val rawTestSet = spark.read.csv(testFile).toDF()

  implicit val encoder: Encoder[LabeledPoint] = ExpressionEncoder()

  private val trainSet = rawTrainSet.map { line =>
    var label = 0
    if (line.get(60).toString.trim.toLong > 9000)
    // 很多人分享
      label = 1
    else
    // 冷门新闻
      label = 0
    LabeledPoint(label,
      Vectors.dense(
        line.get(9).toString.trim.toDouble,
        line.get(10).toString.trim.toDouble,
        line.get(11).toString.trim.toDouble,
        line.get(46).toString.trim.toDouble,
        line.get(47).toString.trim.toDouble,
        line.get(50).toString.trim.toDouble,
        line.get(57).toString.trim.toDouble
      ))
  }.toDF()
  //定义label  根据数据60列的数量选定超过5000就是 label=1很多人share   label=0很少人分享
  // label 可以理解为一个向量对应的一个特殊值，这个值的具体内容可以由用户指定，比如你开发了一个算法A，这个算法对每个向量处理之后会得出一个特殊的标记值p，你就可以把p作为向量标签。
  private val testSet = rawTestSet.map { line =>
    var label = 0
    if (line.get(60).toString.trim.toLong > 5000)
    // 很多人分享1
      label = 1
    else
    // 冷门新闻
      label = 0
    //向量标签和向量是一起的(label,向量特征组）
    // 创建一个稠密向量 Vectors.dense -->很直观，你要创建什么，就加入什么，其函数声明为Vector.dense(values : Array[Double])
    LabeledPoint(label,
      Vectors.dense(
        line.get(9).toString.trim.toDouble,
        line.get(10).toString.trim.toDouble,
        line.get(11).toString.trim.toDouble,
        line.get(46).toString.trim.toDouble,
        line.get(47).toString.trim.toDouble,
        line.get(50).toString.trim.toDouble,
        line.get(57).toString.trim.toDouble
      ))
  }.toDF()

  //!离散  logistics 找阈值分开的两个区域块进行预测 faster than svm
  def selfPreLogistic(): DataFrame = {
    //开始训练
    val model = new LogisticRegression().fit(trainSet.union(testSet))
    if (!File(modelSavePath).exists) {
      println("模型以保存")
      model.save(modelSavePath)
    }
    //预测结果
    val pre = model.transform(testSet)
    showPredictionAccruracy(pre)
    //返回预测结果pre
    pre
  }

  def selfPreLogisticTest(): DataFrame = {
    //开始训练 伴生对象读取已经保存的模型
    val model = LogisticRegressionModel.load(modelSavePath)
    val input = scala.io.StdIn.readLine("请输入训练数据(用,隔开)\n").split(",")
    input.length match {
      case 8 =>
        println("开始测试\n")
      case _ =>
        println("输入格式有误，请重新新输入\n")
        System.exit(0)
    }

    var label = 0
    if (input(0).toLong > 5000)
    // 很多人分享
      label = 1
    else
    // 冷门新闻
      label = 0
    val tmpTrain = LabeledPoint(label,
      Vectors.dense(
        input(1).toDouble,
        input(2).toDouble,
        input(3).toDouble,
        input(4).toDouble,
        input(5).toDouble,
        input(6).toDouble,
        input(7).toDouble
      ))
    val pre = model.transform(Seq(tmpTrain).toDS())
    //预测结果
    showPredictionAccruracy(pre)
    //返回预测结果pre
    pre
  }
/**
  * svm  ：找到一线把平面的点分为2部分  直线上侧 为一种 直线下为另一种  训练  线附近（正则化参数）的数据  but  logistics是训练线两边的数据
  * */
  //非离散(1 label --->n features) svm  找阈值附近的两个点预测  news的热度与好坏都有关
  def selfSvm(): DataFrame = {
    val model = new LinearSVC().fit(trainSet.union(testSet))
    val pre = model.transform(testSet)
    showPredictionAccruracy(pre)
    pre
  }

  def showPredictionAccruracy(data: DataFrame): Unit = {
    data.show(20)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accruracy = evaluator.evaluate(data)
    println(s"准确率为：${accruracy * 100}%")
  }

  // 数据库保存方法 scala强类型语言
  def saveToDatabase(data: DataFrame, tableName: String, isSVM: Boolean = false): Unit = {
    // 准备数据库连接信息
    val prop = new Properties()
    prop.put("user", "root")
    prop.put("password", "1234")
    prop.put("driver", "com.mysql.jdbc.Driver")
    // 设置表头   label feature 预测 预测
    val fields = if (isSVM) "label,features,rawPrediction,prediction".split(",")
      .map(fieldName => StructField(fieldName, StringType, nullable = true))
    else "label,features,rawPrediction,probability,prediction".split(",")
      .map(fieldName => StructField(fieldName, StringType, nullable = true))

    val schema = StructType(fields)

    // 启动行编码器，以使类型推断系统正确推断数据类型
    val encoder = RowEncoder(schema)
    // 格式化结果数据
    val rawRDD = if (isSVM) data.map(x =>
      Row(x(0).toString, x(1).toString, x(2).toString, x(3).toString))(encoder).rdd
    else data.map(x =>
      Row(x(0).toString, x(1).toString, x(2).toString, x(3).toString, x(4).toString))(encoder).rdd

    spark.createDataFrame(rawRDD, schema)
      // 写入数据库
      .write
      .mode("append")
      .jdbc("jdbc:mysql://localhost:3306/dbtaobao?createDatabaseIfNotExist=true&useSSL=false",
        s"dbtaobao.$tableName", prop)
  }


  def main(args: Array[String]): Unit = {
    println("欢迎使用新闻预测系统")
    while (true) {
      val tmp = scala.io.StdIn.readLine("输入N开始训练\n输入Y训练并存入数据库\n输入T进行测试\n输入任意其他字符退出\n>>").toLowerCase()
      tmp match {
        case "y" =>
          saveToDatabase(selfSvm(), "NewsSVM", isSVM = true)
          saveToDatabase(selfPreLogistic(), "NewsLog")
        case "t" =>
          selfPreLogisticTest()
        case "n" =>
//          selfSvm()
          selfPreLogistic()
        case _ =>
          System.exit(0)
      }
    }
  }
}