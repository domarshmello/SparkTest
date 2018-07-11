package main.com.luxury_taobao

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Encoder, SparkSession}


//厦大教程机器学习案例  数据选取的有问题 1.0代表回头客
object dbTaobaoX {

  // 输出一个霸气的 Logo 代表自己是最强的女人
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
      |                    杜杜的大象
    """.stripMargin)

  case class TestDataTrain(user_id: String,
                           item_id: String,
                           cat_id: String,
                           merchant_id: String,
                           brand_id: String,
                           month: String,
                           day: String,
                           action: String,
                           age_range: String,
                           gender: String,
                           province: String)

  val logger = Logger.getLogger("回头客日志")
  def Testdb(): Unit = {

    // 开启 spark 会话
    val spark = SparkSession.builder()
      .appName("dbTaobaoTest")
      //      127
      .master("local")
      .getOrCreate()

    // 启动隐式转换系统

    // 获取 spark 上下文（以后上下文直接用 spark 会话拿）

    val sc = spark.sparkContext

    // 设置解码器
    implicit val encoder: Encoder[String] = ExpressionEncoder()

    // 读取训练数据

    val train_data = sc.textFile("/usr/local/dbtaobao/dataset/train_do.csv")
    val test_data = sc.textFile("/usr/local/dbtaobao/dataset/train_test.csv")

    val train = train_data.map { line =>
      val parts = line.split(',')
      // 模式匹配类：标签，特征
      LabeledPoint(parts(4).toDouble,Vectors.dense(parts(1).toDouble,parts(2).toDouble,parts(3).toDouble))
    }

    val test = test_data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(4).toDouble,Vectors.dense(parts(1).toDouble,parts(2).toDouble,parts(3).toDouble))
    }

    //训练集构建模型SVMWithSGD  训练次数1000次
    val numIterations = 1000
    val model = SVMWithSGD.train(train, numIterations)

//    //    评估模型
//    model.clearThreshold()
//    var scoreAndLabels = test.map { point =>
//      val score = model.predict(point.features)
//      score + " " + point.label
//    }
//    scoreAndLabels.take(5).foreach(println)

    //    没有设定阀值的测试集结果存入到MySQL数据中
    model.clearThreshold()
//    model.setThreshold(0.0)

    // 模型↑
    // ----------------------------------------------------
    // 结果↓
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    scoreAndLabels.foreach(println)




//    val resultX = test1_data.map{line =>
//      val parts = line.split(",")
//      val dataTrain = TestDataTrain(parts(0),parts(1),parts(2),parts(3),
//        parts(4),parts(5),parts(6),parts(7),parts(8),parts(9),parts(10))
//      if (dataTrain.action.toInt != 2) 0 else 1
//    }
//
//    logger.info("匹配结果如下")
//    logger.info("0代表不会再次购买，1代表会再次购买")
//    logger.info("----------|-----------|---------|")
//    scoreAndLabels.zip(resultX).foreach{x =>
//      if (x._1._1 != x._2) logger.error(s"预测结果:${x._1._1} | 实际结果:${x._2}|结果：失败|")
//      else logger.info(s"预测结果:${x._1._1} | 实际结果:${x._2}|结果：成功|")
//      logger.info("----------|---------|---------|")
//    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//    areaUnderROC曲线下面积 预测成功的区域位置  精度是模型的拟合程度 准确度就是整个样本训练的的精确度
    println(s"准确度 = ${metrics.areaUnderROC()}")
    println(s"精度：${metrics.areaUnderPR()}")
//    模型的权重 weights
    println(model.weights)
    println("model.weights.size:"+model.weights.size)

//    //设置回头客数据
//    val rebuyRDD = scoreAndLabels.map(x => List(x._1.toString, x._2.toString))
//    //下面要设置模式信息
//    val schema = StructType(List(StructField("score", StringType, true), StructField("label", StringType, true)))
//    //下面创建Row对象，每个Row对象都是rowRDD中的一行
//    val rowRDD = rebuyRDD.map(p => Row(p.head, p(1)))
//    //建立起Row对象和模式之间的对应关系，也就是把数据和模式对应起来
//    val rebuyDF = spark.createDataFrame(rowRDD, schema)
//    //下面创建一个prop变量用来保存JDBC连接参数
//    val prop = new Properties()
//    prop.put("user", "root") //表示用户名是root
//    prop.put("password", "1234") //表示密码是hadoop
//    prop.put("driver", "com.mysql.jdbc.Driver") //表示驱动程序是com.mysql.jdbc.Driver
//    //下面就可以连接数据库，采用append模式，表示追加记录到数据库dbtaobao的rebuy表中
//    rebuyDF.write.mode("append").jdbc("jdbc:mysql://localhost:3306/dbtaobao?useSSL=true", "dbtaobao.rebuy", prop)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Testdb()
  }
}
