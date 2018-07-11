package main.com.test

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Encoder, Row, SparkSession}

object DataFrameTest {
  def dfFromJson(): Unit = {
    val session = SparkSession.builder().getOrCreate()
    val dataFrame = session.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
    dataFrame.show
    dataFrame.select(dataFrame("name").as("username"), dataFrame("age"))
      .show()
  }

  def dfFromRawText(): Unit = {
    val session = SparkSession.builder()
      .appName("dfFromRawText")
      .getOrCreate()
    val peopleRDD = session.sparkContext
      .textFile("file:///usr/local/spark/examples/src/main/resources/people.txt")
    // 理论上是从文件读取
    val schemaString = "name age"
    val fields = schemaString
      .split(" ")
      .map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    val rowRDD = peopleRDD
      .map(_.split(","))
      .map(attributes => Row(attributes(0), attributes(1).trim))

    val peopleDF = session.createDataFrame(rowRDD, schema)
    peopleDF.createOrReplaceTempView("people")

    implicit val encoder: Encoder[String] = ExpressionEncoder()
    session
      .sql("SELECT name,age FROM people".stripMargin)
      .map { attributes => s"name: ${attributes(0)}, age: ${attributes(1)}" }(encoder)
      .show()

    peopleDF
      .select("name", "age")
      .write.format("csv")
      .save("file:///home/ubuntu/newpeople.csv")
  }

}
