from pyspark.sql import SparkSession

# Kreirati Apache Spark program koji korišdenjem RDD-a broji i prikazuje
# koliko se puta svaka od reči nalazi u tekstu.


spark = SparkSession.builder.appName("Brojnje_reči_1").getOrCreate()

data = [
    "RDD Like Demo",
    "Abstract RDD Like Demo",
    "RDD Demo",
    "Double RDD Demo",
    "Pair RDD Demo",
    "Hadoop RDD Demo",
    "New Hadoop RDD Demo"
]

rdd = spark.sparkContext.parallelize(data)

word_counts = rdd.flatMap(lambda line: line.split())\
    .map(lambda word: (word, 1))\
    .reduceByKey(lambda a, b: a+b)

output = word_counts.collect()

for(word, count) in output:
    print("%s: %i" % (word, count))

spark.sparkContext.stop()