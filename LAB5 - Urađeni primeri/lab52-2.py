from pyspark.sql import SparkSession

# Kreirati Apache Spark program koji korišdenjem RDD-a broji i prikazuje
# koliko se puta svaka od reči nalazi u tekstu.
# Rešenje ukoliko se tekst čita iz fajla.

spark = SparkSession.builder.appName("brojanje_reci_2").getOrCreate()

lines = spark.read.text("input.txt").rdd.map(lambda r: r[0])

word_counts = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

word_counts.saveAsTextFile("output.txt")

spark.sparkContext.stop()
