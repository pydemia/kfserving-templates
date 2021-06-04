#-- coding: utf-8 --
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col, lit, concat, last, avg, when, trim, ltrim, rtrim
from pyspark.ml.feature import *
from pyspark.ml import *
import json, csv, io, re, urllib

# conf = SparkConf().setAppName("pysparkApp").set("spark.driver.cores", 1).set("spark.driver.memory", "1G").set("spark.executor.cores", 1).set("spark.executor.memory", "1G").set("spark.executor.instances", 3)
# sqlContext = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
# sc = sqlContext.sparkContext
# sqlContext.sparkContext.addPyFile('asset.py')
conf = SparkConf().setAppName("pysparkApp")
spark = SparkSession.builder.master('local').getOrCreate()
sc = spark.sparkContext(conf=conf)
sc.addFile("asset.py")

from asset import *

# S3 불러오기 Credential
sc._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", "AKIA43N3NPLEIRWXEOTR")
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "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")
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")

# S3 불러오기 Custom Schema
customSchema_P5KY = StructType([StructField("Id", IntegerType(), True), StructField("MSSubClass", IntegerType(), True), StructField("MSZoning", StringType(), True), StructField("LotFrontage", StringType(), True), StructField("LotArea", IntegerType(), True), StructField("Street", StringType(), True), StructField("Alley", StringType(), True), StructField("LotShape", StringType(), True), StructField("LandContour", StringType(), True), StructField("Utilities", StringType(), True), StructField("LotConfig", StringType(), True), StructField("LandSlope", StringType(), True), StructField("Neighborhood", StringType(), True), StructField("Condition1", StringType(), True), StructField("Condition2", StringType(), True), StructField("BldgType", StringType(), True), StructField("HouseStyle", StringType(), True), StructField("OverallQual", IntegerType(), True), StructField("OverallCond", IntegerType(), True), StructField("YearBuilt", IntegerType(), True), StructField("YearRemodAdd", IntegerType(), True), StructField("RoofStyle", StringType(), True), StructField("RoofMatl", StringType(), True), StructField("Exterior1st", StringType(), True), StructField("Exterior2nd", StringType(), True), StructField("MasVnrType", StringType(), True), StructField("MasVnrArea", StringType(), True), StructField("ExterQual", StringType(), True), StructField("ExterCond", StringType(), True), StructField("Foundation", StringType(), True), StructField("BsmtQual", StringType(), True), StructField("BsmtCond", StringType(), True), StructField("BsmtExposure", StringType(), True), StructField("BsmtFinType1", StringType(), True), StructField("BsmtFinSF1", IntegerType(), True), StructField("BsmtFinType2", StringType(), True), StructField("BsmtFinSF2", IntegerType(), True), StructField("BsmtUnfSF", IntegerType(), True), StructField("TotalBsmtSF", IntegerType(), True), StructField("Heating", StringType(), True), StructField("HeatingQC", StringType(), True), StructField("CentralAir", StringType(), True), StructField("Electrical", StringType(), True), StructField("1stFlrSF", IntegerType(), True), StructField("2ndFlrSF", IntegerType(), True), StructField("LowQualFinSF", IntegerType(), True), StructField("GrLivArea", IntegerType(), True), StructField("BsmtFullBath", IntegerType(), True), StructField("BsmtHalfBath", IntegerType(), True), StructField("FullBath", IntegerType(), True), StructField("HalfBath", IntegerType(), True), StructField("BedroomAbvGr", IntegerType(), True), StructField("KitchenAbvGr", IntegerType(), True), StructField("KitchenQual", StringType(), True), StructField("TotRmsAbvGrd", IntegerType(), True), StructField("Functional", StringType(), True), StructField("Fireplaces", IntegerType(), True), StructField("FireplaceQu", StringType(), True), StructField("GarageType", StringType(), True), StructField("GarageYrBlt", StringType(), True), StructField("GarageFinish", StringType(), True), StructField("GarageCars", IntegerType(), True), StructField("GarageArea", IntegerType(), True), StructField("GarageQual", StringType(), True), StructField("GarageCond", StringType(), True), StructField("PavedDrive", StringType(), True), StructField("WoodDeckSF", IntegerType(), True), StructField("OpenPorchSF", IntegerType(), True), StructField("EnclosedPorch", IntegerType(), True), StructField("3SsnPorch", IntegerType(), True), StructField("ScreenPorch", IntegerType(), True), StructField("PoolArea", IntegerType(), True), StructField("PoolQC", StringType(), True), StructField("Fence", StringType(), True), StructField("MiscFeature", StringType(), True), StructField("MiscVal", IntegerType(), True), StructField("MoSold", IntegerType(), True), StructField("YrSold", IntegerType(), True), StructField("SaleType", StringType(), True), StructField("SaleCondition", StringType(), True), StructField("SalePrice", IntegerType(), True)])

# S3 불러오기
df_P5KY = sqlContext.read.option("delimiter", "").option("header", "true").csv(sc.textFile("s3a://accuinsight/tmp/longhong/housing.csv").map(lambda line : "".join(line.split(","))), schema = customSchema_P5KY)

# Activate Smart Parser
df_P5KY = smartParser(df_P5KY, "none", 0)

# drop
df_NB33 = df_P5KY.drop("Id", "ExterQual", "BsmtQual", "GrLivArea", "FireplaceQu", "GarageCond", "PoolQC")

# dropNaCol
df_k076 = df_NB33.select("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition", "SalePrice")

# nearZeroVar
df_DzoD = df_k076.select("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "PavedDrive", "Fence", "MiscFeature", "MoSold", "YrSold", "SaleType", "SaleCondition", "SalePrice")

# fillna
df_T97V = df_DzoD.fillna("None")

# cast
df_B65E = df_T97V.withColumn("SalePrice",df_T97V.SalePrice.cast(DoubleType()))

# S3 내보내기
# S3 내보내기 Credential
sc._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem")
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", "AKIA43N3NPLEIRWXEOTR")
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "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")
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")

# S3 내보내기
df_B65E.repartition(1).write.format("csv").mode('overwrite').option("header", "true").option("escape", "\"").option("quoteMode", "NONE").save("s3a://accuinsight/tmp/longhong/result")

import boto3, datetime
s3 = boto3.client("s3",endpoint_url="https://s3.ap-northeast-2.amazonaws.com", aws_access_key_id="AKIA43N3NPLEIRWXEOTR", aws_secret_access_key="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")
bucket_name = "accuinsight"
prefix_path = "tmp/longhong/result"
res=s3.list_objects(Bucket=bucket_name, Prefix=prefix_path, MaxKeys=30)
name=res["Contents"][1]["Key"]
target_source = {"Bucket": bucket_name, "Key": name}
target_key ="tmp/longhong/result/housing_pre.csv"
s3.copy(CopySource=target_source, Bucket=bucket_name, Key=target_key)
s3.delete_object(Bucket=bucket_name, Key=name)


sc.stop()