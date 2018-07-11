package main.com.inaction

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Encoder, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object MapReduce {
  private val csv1Path = "H:/LangLearn/hadoop_projects/SparkTest/src/scala/main/com/resource/2012年注册用户.csv"
  private val csv2Path = "H:/LangLearn/hadoop_projects/SparkTest/src/scala/main/com/resource/2017新注册用户.csv"
  private val csv3Path = "H:/LangLearn/hadoop_projects/SparkTest/src/scala/main/com/resource/元老级用户.csv"
  private val title = "mid,name,sex,rank,face,regtime,spacesta,birthday,sign,level,OfficialVerifyType,OfficialVerifyDesc,vipType,vipStatus,toutu,toutuId,coins,following,fans,archiveview,article"

  def readDataFromCsv(): Unit = {
    // 准备连接
    val session = SparkSession.builder()
      .master("local")
      .appName("从csv文件读取要分析的数据")
      .getOrCreate()

    // 建立表头
    val fields = title.split(",")
      .map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    // 产生 RawDataFrame
    val rawDF = session.read.csv(csv3Path)

    // 精修 DataFrame
    val ripeDF = session.createDataFrame(rawDF.rdd, schema)
    ripeDF.createOrReplaceTempView("User")

    // 解码器
    implicit val encoder: Encoder[(String, Int)] = ExpressionEncoder()

    val result = session.sql("SELECT sex FROM User".stripMargin)
      .map(attributes => (attributes(0).toString, 1))(encoder)
      .rdd
      .reduceByKey {
        _ + _
      }
    result.foreach(println(_))

  }
}
