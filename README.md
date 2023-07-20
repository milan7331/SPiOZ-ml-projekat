# Apache Spark aplikacija za predviđanje vremena

Ovaj Python Spark ML program obavlja predviđanje vremena na odabranoj
lokaciji koristeći različite klasifikatore, uključujući Decision Tree,
Random Forest i Naive Bayes. Podaci se učitavaju iz CSV fajla koji sadrži
informacije o vremenskim uslovima. Program uključuje nekoliko koraka obrade
podataka, skaliranje osobina i evaluaciju obučenih modela.
Program je osmišljen da se pokreće na većem broju računara istovremeno,
koristeći Apache Spark za distribuiranu obradu podataka.

#### Obrada podataka

* Podaci se učitavaju iz CSV fajla, zatim se kreira Spark DataFrame.
* Kolona "date" se transformiše u kolonu "month" izdvajanjem informacije o mesecu.
* Duplirani redovi se uklanjaju iz skupa podataka.
* Nulte vrednosti se uklanjaju iz skupa podataka.
* Odstupanja se filtriraju primenom 3-sigma pravila za svaku kolonu.

#### Skaliranje osobina

* Osobine se skaliraju koristeći Min-Max skaliranje na opseg od 0.0 do 1.0.

#### Obuka i testiranje modela

* Skup podataka se deli na trening skup (80%) i test skup (20%).
* Koriste se tri klasifikatora: Decision Tree, Random Forest i Naive Bayes.
* Svaki model se obučava koristeći trening podatke, a zatim se testira na test podacima.

#### Evaluacija modela

* Različite metrike evaluacije se izračunavaju za svaki model, uključujući
tačnost, ~~preciznost, odziv~~ i F1-meru. Matrice konfuzije se generišu za svaki
model kako bi se vizualizovala performansa na različitim klasama.

# Izlaz programa:

![Inicijalni podaci učitani iz csv fajla](/assets/images/slika1.png)

![Broj null vrednosti po kolonama](/assets/images/slika2.png)

![Podaci nakon ekstrakcije datuma, odbacivanja duplikata i null vrednosti](/assets/images/slika3.png)

![Distribucija klasa](/assets/images/slika4.png)

![Podaci nakon skaliranja vrednosti na ospeg 0-1](/assets/images/slika5.png)

![Podaci nakon indeksiranja klase i pretvaranja kolona od znacaja u vektor features](/assets/images/slika6.png)

![Evaluacija modela](/assets/images/slika7.png)

![Matrice konfuzije](/assets/images/slika8.png)

# Verzije korišćenog alata, pokretanje aplikacije i korišćeni dataset:

Verzije:
* Apache Spark: 3.2.4 Pre-built for Apache Hadoop 2.7
* Hadoop: 2.7 winutils
* Java: JDK Java 8 Update 202 
* Python: 3.9.13

Aplikaciju pokrenuti iz terminala sledećom komandom:
```bash
spark-submit --master local[*] --py-files PROGRAM_NAME.py --conf spark.pyspark.python="C:\Users\Milan\AppData\Local\Programs\Python\Python39\python.exe" PROGRAM_NAME.py
```

Korišćeni dataset je moguće pronaći na sledećem linku: [Seatle Weather Prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction)
