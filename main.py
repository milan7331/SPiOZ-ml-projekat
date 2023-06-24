from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, sum, stddev_samp, expr, udf, month

# Dodaj logovanje u konzolu
# Eventualno ispisivanje u output fajl radi preglednosti??

# Putanja do dataseta u csv formatu
path = "./weather.csv"

# Kreiranje spark sesije
spark = SparkSession.builder.appName("Weather prediction").getOrCreate()
spark.sparkContext.setLogLevel("OFF")

# Učitavanje podataka
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)

# Ekstraktujemo mesec iz datuma jer dani i godine nisu od značaja
data = data.withColumnRenamed("date", "month")
data = data.withColumn("month", month("month"))

# Duplikati u ovom datasetu ne postoje ali za svaki slucaj otklanjamo
data = data.dropDuplicates()

# Izveštaj o null vrednostima u datasetu
print("\nBroj null vrednosti po kolonama, iste ce biti odbacene\n")
data.select([sum(col(column).isNull().cast("int")).alias(column) for column in data.columns]).show()

# Otklanjanje redova sa null vrednostima jer nisu pogodni za obradu
data = data.dropna()

# Izveštaj o distribuciji klasa u datasetu / balansiranosti podataka
print("\nDistribucija klasa: \n")
data.groupBy("weather").count().show()

# Eliminacija izuzetaka / odstupanja (outliers) po kolonama
for i in data.columns[:-1]:
    # Računamo srednju vrednost kolone i standardnu devijaciju
    mean_value = data.agg({i: "mean"}).collect()[0][0]
    std_dev = data.agg({i: "stddev"}).collect()[0][0]

    # Biramo gornju i donju granicu koje će biti korišćene prilikom filtriranja
    lower_bound = mean_value - 3 * std_dev
    upper_bound = mean_value + 3 * std_dev

    # Filtriramo kolone sa vrednostima koje odstupaju
    data = data.filter(col(i).between(lowerBound=lower_bound, upperBound=upper_bound))


# Izveštaj o distribuciji klasa u datasetu nekon filtriranja
# Note: Ovakav vid filtriranja najviše čisti "rain" klasu vrv nije najbolja ideja??
print("\nDistribucija klasa nakon filtriranja\n")
data.groupBy("weather").count().show()

# User defined funkcija koja konvertuje vector tip nazad u float sa 10 decimala
# Treba nam na kraju skaliranja kako bi povratili prvobitni format
udf_float10 = udf(lambda x: round(float(x[0]), 10), DoubleType())

# Skaliranje po kolonama
for i in data.columns[:-1]:
    # Transformacija kolone u vektor
    assembler = VectorAssembler(inputCols=[i], outputCol=i+"_v")

    # Skaliranje vrednosti iz kolone (vektora) na vrednosti od 0.0 do 1.0
    # Moguće je koristiti i StandardScaler
    scaler = MinMaxScaler(inputCol=i+"_v", outputCol=i+"_s")

    # Pipeline koji vrši sekvencu transformacija nad podacima
    pipeline = Pipeline(stages=[assembler, scaler])

    # Konačno vršimo skaliranje nad samim podacima
    data = pipeline.fit(data).transform(data)

    # Vršimo konverziju podataka nazad na float vrednosti
    data = data.withColumn(i+"_s", udf_float10(i+"_s"))

    # Odbacujemo originalne i vector vrednosti
    data = data.drop(i+"_v")
    data = data.drop(i)


# Nakon skaliranja premeštamo kolonu klase na kraj gde je bila i do sada
data = data.select(*(data.columns[1:] + data.columns[:1]))

# Datasetu dodajemo label kolonu koja sadrži klasu kao numeričku vrednost
indexer = StringIndexer(inputCol="weather", outputCol="label")
data = indexer.fit(data).transform(data)

# Transformišemo kolone od značaja u vektor pod imenom features
# Dakle bez kolona klase i label
assembler = VectorAssembler(inputCols=data.columns[0:-2], outputCol="features")
data = assembler.transform(data)

# Delimo dataset na treining i test setove u odnosu 80:20
(training_data, test_data) = data.randomSplit([0.8, 0.2], seed=1010)

# Kreiranje ML modela
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# Treniranje ML modela
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
