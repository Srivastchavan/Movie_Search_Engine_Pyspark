# Databricks notebook source
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import explode
import math
from math import sqrt

# COMMAND ----------

#1. tokenizing the plot summary words 
moviePlots = spark.read.option("header", "false").option("delimiter","\t").csv("/FileStore/tables/plot_summaries.txt").toDF("id","summary")
tokenizer = RegexTokenizer().setInputCol("summary").setOutputCol("words").setPattern("\\W+")
wordsData = tokenizer.transform(moviePlots)
#2. removing stop words
remover = StopWordsRemover().setInputCol("words").setOutputCol("Word Collection")
plotWords = remover.transform(wordsData).select("id","Word Collection").toDF("movie_id","plot_summary")
plotWords.show()

# COMMAND ----------

#3. Converting to RDD after exploding the rows
plotWords_Split = plotWords.select(plotWords.movie_id,explode(plotWords.plot_summary)).rdd
plotWords_Split.collect()

# COMMAND ----------

#performing map-reduce to get the word count per movie per word
movWord_Init = plotWords_Split.map(lambda x:((x[0],x[1]),1))
movWord_Count = movWord_Init.reduceByKey(lambda x,y:x+y)
movWord_Count.take(5)

# COMMAND ----------

#performing map-reduce to find the movie frequency for the particular word
total_Docs = plotWords_Split.count()
docs_Count = movWord_Count.map(lambda x:(x[0][1],1))
tf = docs_Count.reduceByKey(lambda x,y: x + y)
tf.take(5)

# COMMAND ----------

#computing tf-idf 
idf = tf.map(lambda x:(x[0],math.log((total_Docs/x[1]))))
word_Movie = movWord_Count.map(lambda x:(x[0][1],(x[0][0],x[1])))
tfidf = word_Movie.join(idf)
final_tfidf = tfidf.map(lambda x: (x[0],((x[1][0][0],x[1][0][1]),x[1][1],x[1][1]*x[1][0][1])))
final_tfidf.take(5)

# COMMAND ----------

#reading movie metadata for getting the movie name from movieId
movie_metadata = spark.read.csv('/FileStore/tables/movie_metadata.tsv', header=None, sep = '\t')
movie_details = movie_metadata.select("_c0","_c2").toDF("movie_id","movie_name").rdd
movie_details.take(5)

# COMMAND ----------

#4a. single input file searching
one_input = sc.textFile("/FileStore/tables/single_Name_File.txt").map(lambda x : x.lower())
search_terms = one_input.collect()
for term in search_terms:
  print("The Related Movies are below for:",term)
  rel_movies = sc.parallelize(final_df.filter(lambda x: x[0] == term).sortBy(lambda x: -x[1][2]).map(lambda x : (x[1][0][0], x[1][2])).take(10))
  result = movie_details.join(rel_movies)   
  result = result.sortBy(lambda x : -x[1][1]).map(lambda x : [x[0],x[1][0],x[1][1]]).toDF(["movie_id", "movie_name", "TF-IDF value"])
  result.show(truncate=False) 
  

# COMMAND ----------

#4b. query term search 
query = sc.textFile("/FileStore/tables/search_Query.txt").collect()[0].split()
query = [x.lower() for x in query]
freq_terms = sc.parallelize(query).map(lambda x : (x, 1)).reduceByKey(lambda x,y : x+y)
freq_terms.collect()

final_tfidf_mod = final_tfidf.map(lambda x :(x[0], x[1][2]))
terms_present = freq_terms.join(final_tfidf_mod)
terms_present.take(5)
merge_terms = terms_present.map(lambda x : (x[0], x[1][1]))
tf_movies = final_tfidf.map(lambda x : (x[0], (x[1][0][0], x[1][2]))).join(merge_terms).map(lambda x : (x[1][0], x[1][1], x[1][0][1]))
tf_movies.take(5)

# COMMAND ----------

#computing the cosine similarity
cos_val_init = tf_movies.map(lambda x : (x[0], (x[1] * x[2], x[2] * x[2], x[1] * x[1])))
cos_val = cos_val_init.reduceByKey(lambda x,y : ((x[0] + y[0], x[1] + y[1], x[2] + y[2])))
cos_score = cos_val.map(lambda x : (x[0], x[1][0]/(sqrt(x[1][1]) * sqrt(x[1][2]))))
cos_score.sortBy(lambda x : -x[1]).take(5)
RDD_res =  cos_score.map(lambda x : (x[0][0], 1)).reduceByKey(lambda x,y : x+y)
RDD_res.take(10)

# COMMAND ----------

#5. printing the final results based on top cosine similarity scores
movie_list = RDD_res.join(movie_details).map(lambda x : (x[0], x[1][1]))
movie_cos_score = cos_score.map(lambda x : (x[0][0], x[1]))
movie_cos_score = movie_cos_score.join(movie_list).distinct().sortBy(lambda x : -x[1][0])
sorted_movie_cos_score = movie_cos_score.map(lambda x : [x[0],x[1][1],x[1][0]]).toDF(["Movie Id", "Movie Name", "Cosine Similarity"])
sorted_movie_cos_score.show(truncate=False)
