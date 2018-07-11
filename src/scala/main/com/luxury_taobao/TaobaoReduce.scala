package main.com.luxury_taobao


import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Encoder, Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

/**
  * Created by apple on 2018/6/8.
  */
object TaobaoReduce {

  private val data_front = "/usr/local/dbtaobao/dataset/user_log_front.csv"
  private val data_back = "/usr/local/dbtaobao/dataset/user_log_back.csv"
  private val data = "/usr/local/dbtaobao/dataset/user_log.csv"

  def provinceCollect(): Unit = {
    val spark = SparkSession.builder()
      .appName("ProvinceCollect")
      .master("local")
      .getOrCreate()

    import spark.implicits._
    val sc = spark.sparkContext

    val peopleRDD = spark.read.csv(data_front)

    val schemaString = "user_id,item_id,cat_id,merchant_id,brand_id,month,day,action,age_range,gender,province"
    val fields = schemaString.split(',')
      .map{fieldName =>
        StructField(fieldName, StringType, nullable = true)
      }

    val schema = StructType(fields)

    val provinceActionTable = spark.createDataFrame(peopleRDD.rdd, schema).createOrReplaceTempView("Pro_table")

    implicit val encoder: Encoder[(String, Int)] = ExpressionEncoder()

    val result = spark.sql("SELECT province FROM Pro_table WHERE action == 2".stripMargin)
      .map(attr => (attr(0).toString, 1))(encoder)
      .rdd
      .reduceByKey(_ + _)
      .sortBy(_._2)

    val totalNumber = spark.sql("SELECT province FROM Pro_table WHERE action != 0").count()

    val finalResult = result.map(a => (a._1, a._2.toDouble / totalNumber))
    finalResult.foreach(println)







  }

}
