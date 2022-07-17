import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max, mean, percentile_approx, expr, col, variance, count, countDistinct, when
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Create the spark session
spark = SparkSession.builder.appName('nuclear_plants').getOrCreate()

# make data frame from csv file
df = spark.read.option("header", True) \
    .option("inferSchema", True) \
    .csv("./nuclear_plants_small_dataset.csv")

df.printSchema()

############################################################

# Task 1

df = df.na.drop("any")

# Task 2
list_abnormal = []
list_normal = []
for i in df.columns:
    column = "Power_range_sensor_1"
    if i == "Status":
        continue
    column = i
    grouped = df.groupby("Status").agg(min(column).alias("min"),
                                       max(column).alias("max"),
                                       mean(column).alias("mean"),
                                       percentile_approx(
                                           column, 0.5).alias("median"),
                                       count(column).alias("mode"),
                                       variance(column).alias("variance")) \
        .withColumnRenamed("Status", "Status\\" + column)
    grouped.show(truncate=False)

    pdf = grouped.toPandas()
    list_abnormal.append(
        [column, pdf.iloc[0]["min"], pdf.iloc[0]["max"], pdf.iloc[0]["mean"], pdf.iloc[0]["median"],
         pdf.iloc[0]["mode"], pdf.iloc[0]["variance"]])
    list_normal.append([column, pdf.iloc[1]["min"], pdf.iloc[1]["max"], pdf.iloc[1]["mean"], pdf.iloc[1]["median"],
                        pdf.iloc[1]["mode"], pdf.iloc[1]["variance"]])

panda_df_abnormal = pd.DataFrame(list_abnormal,
                                 columns=["Name", "min", "max", "mean", "median", "mode", "variance"])
panda_df_normal = pd.DataFrame(list_normal, columns=[
                               "Name", "min", "max", "mean", "median", "mode", "variance"])

panda_df_normal.plot(title="Normal", kind="box")
panda_df_abnormal.plot(title="Abnormal", kind="box")
plt.show()

# Task3
print(panda_df_normal.corr())
print(panda_df_abnormal.corr())

############################################################
# Section 2

# Task 4
print("")
train, test = df.randomSplit([0.7, 0.3], 2021)
print("Train Dataset Count: " + str(train.count()))
train.groupby("Status").count().show()
print("Test Dataset Count: " + str(test.count()))
test.groupby("Status").count().show()

# Task 5
# Decision Tree Classifier

# prepare Data
df = df.withColumn("label", when(df.Status == "Normal", 0).when(
    df.Status == "Abnormal", 1).otherwise(1))
featureCols = []
for i in df.columns:
    if i == "Status":
        continue
    if i == "label":
        continue
    featureCols.append(i)
va = VectorAssembler(inputCols=featureCols, outputCol="features")
va_df = va.transform(df)

# select train and test data
train, test = va_df.randomSplit([0.7, 0.3], 2021)

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtcModel = dtc.fit(train)
predictions = dtcModel.transform(test)
predictions = predictions.withColumn(
    "prediction", predictions.prediction.cast("int"))


# Calculate error rate
testCount = predictions.count()
diffCount = predictions.filter(col("label") != col("prediction")).count()
error_rate = diffCount / testCount

# Calculate Sensitivity and Specificity
TP = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
FN = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
FP = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
TN = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
print("Decision Tree Model")
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(
    f"error rate: {error_rate}, sensitivity: {sensitivity}, specificity: {specificity}")

# Support Vector Machine Model
lsvc = LinearSVC()
lsvcModel = lsvc.fit(train)
predictions = lsvcModel.transform(test)
predictions = predictions.withColumn(
    "prediction", predictions.prediction.cast("int"))


# Calculate error rate
diffCount = predictions.filter(col("label") != col("prediction")).count()
error_rate = diffCount / testCount

# Calculate Sensitivity and Specificity
TP = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
FN = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
FP = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
TN = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
print("\nSupport Vector Machine")
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(
    f"error rate: {error_rate}, sensitivity: {sensitivity}, specificity: {specificity}")

# Artificial Neural Network
mpc = MultilayerPerceptronClassifier(layers=[len(featureCols), 20, 2])
mpcModel = mpc.fit(train)
predictions = mpcModel.transform(test)
predictions = predictions.withColumn(
    "prediction", predictions.prediction.cast("int"))

# Calculate error rate
diffCount = predictions.filter(col("label") != col("prediction")).count()
error_rate = diffCount / testCount

# Calculate Sensitivity and Specificity
TP = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
FN = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
FP = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
TN = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
print("\nArtificial Neural Network")
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(
    f"error rate: {error_rate}, sensitivity: {sensitivity}, specificity: {specificity}")


# Task 8
print("")
df = spark.read.option("header", True) \
    .option("inferSchema", True) \
    .csv("./nuclear_plants_big_dataset.csv") \
    .na.drop("any") \
    .drop("Status")


def func_min(x):
    temp = ("min",)
    temp += x
    return temp


def func_max(x):
    temp = ("max",)
    temp += x
    return temp


def func_mean(x):
    temp = ("mean",)
    temp += x
    return temp


columns = ["kind"]
for i in df.columns:
    columns.append(i)

rows = []
row = df.rdd.map(lambda x: func_min(x)).min()
rows.append(row)
row = df.rdd.map(lambda x: func_max(x)).max()
rows.append(row)
for row in df.select([mean(c).alias(c) for c in df.columns]).rdd.map(lambda x: func_mean(x)).collect():
    rows.append(row)

dataframe = spark.createDataFrame(rows, columns)
dataframe.show()

