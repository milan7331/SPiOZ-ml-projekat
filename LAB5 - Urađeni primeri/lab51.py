from pyspark import SparkContext, SparkConf

# Kreirati Apache Spark program koji filtrira brojeve manje od 10.
# Pronalazi kvadrate filtriranih brojeva i u konzoli prikazuje sumu dobivenih kvadrata.

conf = SparkConf().setAppName("test").setMaster("local")

sc = SparkContext(conf=conf)

data = [1, 2, 3, 4, 50, 61, 72, 8, 9, 19, 31, 42, 53, 6, 7, 23]
# data = [(1.5, 2.2, 3.1, 4.2, 50.1, 61.3, 72.8, 8.2, 9.5, 19.6, 31.7, 42.8, 53.3, 6.6, 7.4, 23.1]

rdd = sc.parallelize(data)

filteredRDD = rdd.filter(lambda x: x > 10)

transformedRDD = filteredRDD.map(lambda x: x*x)

numbers_sum = transformedRDD.reduce(lambda x, y: x+y)

print("Suma kvadrata brojeva većih od 10:", numbers_sum)
