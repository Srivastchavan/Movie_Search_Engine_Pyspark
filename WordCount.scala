// Databricks notebook source
import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

// COMMAND ----------

val input = sc.textFile("/FileStore/tables/The_Adventures_of_Tom_Sawyer.txt")
val words = input.flatMap(x=> x.split("""\W+"""))
val longwords = words.filter(x=>x.length > 2)
var sentence = longwords.collect.mkString(" ")

// COMMAND ----------

val spark = SparkNLP.start()
val pipeline = PretrainedPipeline("explain_document_dl", lang="en")

// COMMAND ----------

val result = pipeline.annotate(sentence)

// COMMAND ----------

val count_words = result("entities")

// COMMAND ----------

val Pairs = count_words.map(x=>(x,1))
val Freq = Pairs.groupBy(_._1).map{ case (key, list) => key -> list.map(_._2).reduce(_+_)}

// COMMAND ----------

val top10_counts = Freq.toSeq.sortBy(-_._2).take(10) 
