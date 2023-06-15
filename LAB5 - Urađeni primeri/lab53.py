from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# Kreirati Apache Spark program za klasifikaciju koji na osnovu zadatih atributa koji opisuju profil pacijanta
# određuje da li de rezultati testa na dijabetes biti pozitivni ili negativni. Profil pacijanta je opisan sa
# ukupno 8 numeričnih atributa, deveti atribut predstavlja klasu (0 – ako nema dijabetes, 1 – ako ima dijabetes).

spark = SparkSession.builder.appName('PrimerKlasifikacije').getOrCreate()

data = spark.read.csv('diabetes.csv', header=True, inferSchema=True)

assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")

data = assembler.transform(data).select("features", "label")

data.show(5)

splits = data.randomSplit([0.7, 0.3], 1234)

train = splits[0]

test = splits[1]

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

model = nb.fit(train)

predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

predictionAndLabels = predictions.select("prediction", "label").rdd.map(tuple)

metrics = MulticlassMetrics(predictionAndLabels)

confusionMatrix = metrics.confusionMatrix()
print("Confusion Matrix:")
print(confusionMatrix)

spark.stop()
