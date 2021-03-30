# Databricks notebook source
#Importar librerías requeridas

import numpy as np
import pandas as pd
import pyspark

from pyspark.sql import SparkSession
from pyspark.ml.feature import (VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

#Importamos los datos

spark = SparkSession.builder.appName('Trabajo').getOrCreate()
#Leer data de un csv:
data = spark.read.format("csv").load("dbfs:/FileStore/shared_uploads/kasanro@gmail.com/customer_churn-2.csv",inferSchema=True,header=True)
#Describir un dataframe
data.describe().show()
#Ver las columnas
data.columns
#Imprimir cuantos datos importamos
print(data.count())
#Mostramos los primeros registros
data.show()


# COMMAND ----------

#Defininos los campos a usar
inputCols = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']

#Definimos el assembler y pasamos por parametro los campos de entrada
assembler = VectorAssembler(inputCols=inputCols,outputCol='features')

#Transformamos la data
output = assembler.transform(data)

print(output.columns)

#Seleccionamos los campos requeridos y la definimos los porcentajes de uso de data de entrenamiento y prueba
final_data = output.select('features','churn')
train, test = final_data.randomSplit([0.7, 0.3])




# COMMAND ----------

#Generamos la logistica de regresión

lr = LogisticRegression(labelCol='churn')
lr_model = lr.fit(train)
train_summary = lr_model.summary
train_summary.predictions.describe().show()

roc = train_summary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# COMMAND ----------

#Evaluamos el modelo con la data de prueba

pred_and_labels = lr_model.evaluate(final_data)
pred_and_labels.predictions.show()

#Evaluamos el área bajo la curva
my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='churn',metricName='areaUnderROC')
auc = my_eval.evaluate(pred_and_labels.predictions)
print(auc)

#Evaluamos la precisión
my_eval2 = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='churn',metricName='accuracy')
accuracy = my_eval2.evaluate(pred_and_labels.predictions)
print(accuracy)




# COMMAND ----------

#### Nueva data para predecir ###

#Generamos la logica de regresión con toda la data
lr = LogisticRegression(labelCol='churn')
lr_model = lr.fit(final_data)

#importamos el csv con los nuevos clientes
new_customers = spark.read.format("csv").load("dbfs:/FileStore/shared_uploads/kasanro@gmail.com/new_customers-1.csv",inferSchema=True,header=True)

#Imprimimos algunos datos de la nueva data
print(new_customers.count())
new_customers.show()

#transformamos los data de entrada con la definicion del assembler inicial
test_new_customers = assembler.transform(new_customers) #  usar el mismo assembler anterior

#Generamos el analisis de la nueva data de acuerdo a la logica de regresión
results = lr_model.transform(test_new_customers)

#Imprimimos los resultados.
results.select('Company', 'prediction').show()

