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
assembled_data = assembler.transform(indexedData)


# Split the data into training and testing sets
(training_data, test_data) = assembled_data.randomSplit([0.8, 0.2], seed=99999999)

# Kreiranje ML modela
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = NaiveBayes()

# Treniranje modela
dt_model = dt.fit(training_data)
rf_model = rf.fit(training_data)
nb_model = nb.fit(training_data)

# Testiranje modela
dt_predictions = dt_model.transform(test_data)
rf_predictions = rf_model.transform(test_data)
nb_predictions = nb_model.transform(test_data)

# Evaluation metrics za Decision Tree
dt_evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
dt_evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
dt_evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Evaluation metrics za Random Forest
rf_evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
rf_evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
rf_evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Evaluation metrics za Naive Bayes
nb_evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nb_evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
nb_evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
nb_evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Accuracy, precision, recall i f1 za Decision Tree
dt_accuracy = dt_evaluator_accuracy.evaluate(dt_predictions)
dt_precision = dt_evaluator_precision.evaluate(dt_predictions)
dt_recall = dt_evaluator_recall.evaluate(dt_predictions)
dt_f1 = dt_evaluator_f1.evaluate(dt_predictions)
print(f"Decision Tree Stats:\n Accuracy: {dt_accuracy}\n Precision: {dt_precision}\n Recall: {dt_recall}\n F1: {dt_f1}\n\n")

# Accuracy, precision, recall i f1 za Random Forest
rf_accuracy = rf_evaluator_accuracy.evaluate(rf_predictions)
rf_precision = rf_evaluator_precision.evaluate(rf_predictions)
rf_recall = rf_evaluator_recall.evaluate(rf_predictions)
rf_f1 = rf_evaluator_f1.evaluate(rf_predictions)
print(f"Random Forest Stats:\n Accuracy: {rf_accuracy}\n Precision: {rf_precision}\n Recall: {rf_recall}\n F1: {rf_f1}\n\n")

# Accuracy, precision, recall i f1 za Naive Bayes
nb_accuracy = nb_evaluator_accuracy.evaluate(nb_predictions)
nb_precision = nb_evaluator_precision.evaluate(nb_predictions)
nb_recall = nb_evaluator_recall.evaluate(nb_predictions)
nb_f1 = nb_evaluator_f1.evaluate(nb_predictions)
print(f"Naive Bayes Stats:\n Accuracy: {nb_accuracy}\n Precision: {nb_precision}\n Recall: {nb_recall}\n F1: {nb_f1}\n\n")

# Matrica konfuzije za Decision Tree
dt_prediction_and_labels = dt_predictions.select("prediction", "label").rdd
dt_metrics = MulticlassMetrics(dt_prediction_and_labels)
dt_confusion_matrix = dt_metrics.confusionMatrix()
print(f"Decision Tree Confusion Matrix:\n {dt_confusion_matrix}")

# Matrica konfuzije za Random Forest
rf_prediction_and_labels = rf_predictions.select("prediction", "label").rdd
rf_metrics = MulticlassMetrics(rf_prediction_and_labels)
rf_confusion_matrix = rf_metrics.confusionMatrix()
print(f"Random Forest Confusion Matrix:\n {rf_confusion_matrix}")

# Matrica konfuzije za Naive Bayes
nb_prediction_and_labels = nb_predictions.select("prediction", "label").rdd
nb_metrics = MulticlassMetrics(nb_prediction_and_labels)
nb_confusion_matrix = nb_metrics.confusionMatrix()
print(f"Naive Bayes Confusion Matrix:\n {nb_confusion_matrix}")

spark.stop()
