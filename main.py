from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, stddev_samp, expr, udf

from pyspark.sql.types import DoubleType

from pyspark.mllib.evaluation import MulticlassMetrics

# 1 - Definicija kolona odabranog dataset-a
columns = ["precipitation", "temp_max", "temp_min", "wind"]
columns_scaled = ["precipitation_S", "temp_max_S", "temp_min_S", "wind_S"] # cemu ovo koji k

# 2 - Definicija klasa
labels = ["drizzle", "rain", "sun", "snow", "fog"]

# 3 - Kreiranje spark sesije
spark = SparkSession.builder.appName("Weather Prediction").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 4 - Putanja do dataseta u csv formatu
path = "./weather.csv"

# 5- Učitavanje podataka
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)

########################  2) Preprocesiranje (koristeći Spark)
# Drop unnecessary columns
data = data.drop("date")

print("\nData after dropping unnecessary columns:")
data.show()

# Count and remove duplicate rows
duplicate_count = data.dropDuplicates().count() - data.count()
print(f"\nNumber of duplicate rows: {duplicate_count}\n")

# Check for null values in each column
print("\nNumber of null values in each column:")
data.select([sum(col(column).isNull().cast("int")).alias(column) for column in data.columns]).show()

# Check class balance
print("\nClass distribution:")
data.groupBy("weather").count().show()

# eliminacija outliers za kolone??
statistics = data.select([stddev_samp(column).alias(column) for column in data.columns]).first()

# Define the lower and upper bounds for outliers (moze i 3) (sta moze bre 3 tf?)
lower_bounds = [(statistics[column] - 2 * statistics[column]) for column in columns]
upper_bounds = [(statistics[column] + 2 * statistics[column]) for column in columns]

#filtriranje redova / slogova / cega god kojima se kriterijumi ne poklapaju??
data.filter(expr(" AND ".join([f"({column} >= {lower_bound} AND {column} <= {upper_bound})" for column, lower_bound, upper_bound in zip(columns, lower_bounds, upper_bounds)]))).show()


# normalizacija meme
unlist = udf(lambda x: round(float(list(x)[0]), 15), DoubleType())

for i in columns:
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect") # VectorAssembler Transformation - Converting column to vector type
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_S") # MinMaxScaler Transformation
    pipeline = Pipeline(stages=[assembler, scaler])  # Pipeline of VectorAssembler and MinMaxScaler
    data = pipeline.fit(data).transform(data).withColumn(i+"_S", unlist(i+"_S")).drop(i+"_Vect")  # Fitting pipeline on dataframe

# Drop the first 4 columns
drop_columns = data.columns[:4]
data = data.drop(*drop_columns)
data = data.select(*([col(c) for c in data.columns[1:]] + [col(data.columns[0])])) # Move the 11th column to the end
data.show()

# # # # chat gpt koristio drugi neki scaller koja je razlika samo bog zna??
# # # # Perform feature scaling
# # # assembler = VectorAssembler(inputCols=columns[1:], outputCol="features")
# # # scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# # # pipeline = Pipeline(stages=[assembler, scaler])
# # # data = pipeline.fit(data).transform(data)

# # # # Drop the original features column and keep the scaled features
# # # data = data.drop("features").withColumnRenamed("scaledFeatures", "features")

# # # print("\nData after feature scaling:")
# # # data.show()


# Optional: Select specific attributes for training
# data = data.select("features", "label")

#############################################################################################################

# # Convert string labels to numeric format
indexer = StringIndexer(inputCol="weather", outputCol="label")
indexedData = indexer.fit(data).transform(data)

# odabir atributa i pretvaranje u vektor tf??
assembler = VectorAssembler(inputCols=columns_scaled, outputCol="features")
assembledData = assembler.transform(indexedData)


# Split the data into training and testing sets
(trainingData, testData) = assembledData.randomSplit([0.8, 0.2], seed=99999999)

# Kreiranje ML modela
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = NaiveBayes()

# Treniranje modela
dtModel = dt.fit(trainingData)
rfModel = rf.fit(trainingData)
nbModel = nb.fit(trainingData)

# Testiranje modela
dtPredictions = dtModel.transform(testData)
rfPredictions = rfModel.transform(testData)
nbPredictions = nbModel.transform(testData)

#### OVAJ evaluator đavo može da izvuče F1 i ostalo pored accuracy vidi chat gpt sranje od pre

# Evaluation metrics
dtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
rfEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
nbEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")

# Accuracy for Decision Tree
dtAccuracy = dtEvaluator.evaluate(dtPredictions)
print("Decision Tree Accuracy:", dtAccuracy)

# Accuracy for Random Forest
rfAccuracy = rfEvaluator.evaluate(rfPredictions)
print("Random Forest Accuracy:", rfAccuracy)

# Accuracy for Naive Bayes
nbAccuracy = nbEvaluator.evaluate(nbPredictions)
print("Naive Bayes Accuracy:", nbAccuracy)

# Matrica konfuzije za - Decision Tree
dtPredictionAndLabels = dtPredictions.select("prediction", "label").rdd
dtMetrics = MulticlassMetrics(dtPredictionAndLabels)
dtConfusionMatrix = dtMetrics.confusionMatrix()
print("Decision Tree Confusion Matrix:")
print(dtConfusionMatrix)

# Matrica konfuzije za - Random Forest
rfPredictionAndLabels = rfPredictions.select("prediction", "label").rdd
rfMetrics = MulticlassMetrics(rfPredictionAndLabels)
rfConfusionMatrix = rfMetrics.confusionMatrix()
print("Random Forest Confusion Matrix:")
print(rfConfusionMatrix)

# Matrica konfuzije za - Naive Bayes
nbPredictionAndLabels = nbPredictions.select("prediction", "label").rdd
nbMetrics = MulticlassMetrics(nbPredictionAndLabels)
nbConfusionMatrix = nbMetrics.confusionMatrix()
print("Naive Bayes Confusion Matrix:")
print(nbConfusionMatrix)

spark.stop()
