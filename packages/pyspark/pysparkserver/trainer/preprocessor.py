from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col, lit, concat, last, avg, when, trim, ltrim, rtrim
from pyspark.ml.feature import *
from pyspark.ml import *
import json, csv, io, re, urllib

from typing import Dict, List
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import from_json, col

__all__ = ['prep_func']


conf = SparkConf().setAppName("pysparkApp")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
sc.addFile("../../pipeline/asset.py")

from pysparkserver.pipeline.asset import *
# from asset import *


customSchema = StructType([StructField("Id", IntegerType(), True), StructField("MSSubClass", IntegerType(), True), StructField("MSZoning", StringType(), True), StructField("LotFrontage", StringType(), True), StructField("LotArea", IntegerType(), True), StructField("Street", StringType(), True), StructField("Alley", StringType(), True), StructField("LotShape", StringType(), True), StructField("LandContour", StringType(), True), StructField("Utilities", StringType(), True), StructField("LotConfig", StringType(), True), StructField("LandSlope", StringType(), True), StructField("Neighborhood", StringType(), True), StructField("Condition1", StringType(), True), StructField("Condition2", StringType(), True), StructField("BldgType", StringType(), True), StructField("HouseStyle", StringType(), True), StructField("OverallQual", IntegerType(), True), StructField("OverallCond", IntegerType(), True), StructField("YearBuilt", IntegerType(), True), StructField("YearRemodAdd", IntegerType(), True), StructField("RoofStyle", StringType(), True), StructField("RoofMatl", StringType(), True), StructField("Exterior1st", StringType(), True), StructField("Exterior2nd", StringType(), True), StructField("MasVnrType", StringType(), True), StructField("MasVnrArea", StringType(), True), StructField("ExterQual", StringType(), True), StructField("ExterCond", StringType(), True), StructField("Foundation", StringType(), True), StructField("BsmtQual", StringType(), True), StructField("BsmtCond", StringType(), True), StructField("BsmtExposure", StringType(), True), StructField("BsmtFinType1", StringType(), True), StructField("BsmtFinSF1", IntegerType(), True), StructField("BsmtFinType2", StringType(), True), StructField("BsmtFinSF2", IntegerType(), True), StructField("BsmtUnfSF", IntegerType(), True), StructField("TotalBsmtSF", IntegerType(), True), StructField("Heating", StringType(), True), StructField("HeatingQC", StringType(), True), StructField("CentralAir", StringType(), True), StructField("Electrical", StringType(), True), StructField("1stFlrSF", IntegerType(), True), StructField("2ndFlrSF", IntegerType(), True), StructField("LowQualFinSF", IntegerType(), True), StructField("GrLivArea", IntegerType(), True), StructField("BsmtFullBath", IntegerType(), True), StructField("BsmtHalfBath", IntegerType(), True), StructField("FullBath", IntegerType(), True), StructField("HalfBath", IntegerType(), True), StructField("BedroomAbvGr", IntegerType(), True), StructField("KitchenAbvGr", IntegerType(), True), StructField("KitchenQual", StringType(), True), StructField("TotRmsAbvGrd", IntegerType(), True), StructField("Functional", StringType(), True), StructField("Fireplaces", IntegerType(), True), StructField("FireplaceQu", StringType(), True), StructField("GarageType", StringType(), True), StructField("GarageYrBlt", StringType(), True), StructField("GarageFinish", StringType(), True), StructField("GarageCars", IntegerType(), True), StructField("GarageArea", IntegerType(), True), StructField("GarageQual", StringType(), True), StructField("GarageCond", StringType(), True), StructField("PavedDrive", StringType(), True), StructField("WoodDeckSF", IntegerType(), True), StructField("OpenPorchSF", IntegerType(), True), StructField("EnclosedPorch", IntegerType(), True), StructField("3SsnPorch", IntegerType(), True), StructField("ScreenPorch", IntegerType(), True), StructField("PoolArea", IntegerType(), True), StructField("PoolQC", StringType(), True), StructField("Fence", StringType(), True), StructField("MiscFeature", StringType(), True), StructField("MiscVal", IntegerType(), True), StructField("MoSold", IntegerType(), True), StructField("YrSold", IntegerType(), True), StructField("SaleType", StringType(), True), StructField("SaleCondition", StringType(), True), StructField("SalePrice", IntegerType(), True)])


def _df_from_json(json_str: str) -> DataFrame:
    return spark.read.json(sc.parallelize([json_str]), schema=customSchema)


def _df_from_dict(_dict: Dict) -> DataFrame:
    return spark.createDataFrame([_dict], schema=customSchema)


def _df_from_list(dict_list: List[Dict]) -> DataFrame:
    return spark.createDataFrame(dict_list, schema=customSchema)


def _df_from_pandas_df(pandas_df: pd.DataFrame) -> DataFrame:
    return spark.createDataFrame(pandas_df, schema=customSchema)

def _is_valid_json(s: str) -> bool:
  try:
    json.loads(s)
  except ValueError as e:
    return False
  return True


def prep_func(inputs) -> pd.DataFrame:
    if isinstance(inputs, str):
        if _is_valid_json:
            df = _df_from_json(inputs)
        else:
            raise ValueError(f'data in "instances": invalid json format')
    elif isinstance(inputs, dict):
        df = _df_from_dict(inputs)
    elif isinstance(inputs, list):
        df = _df_from_list(inputs)
    elif isinstance(inputs, pd.DataFrame):
        df = _df_from_pandas_df(inputs)
    else:
        raise ValueError(f'data in "instances": not supported format in [json, Dict, List[Dict], pd.DataFrame]')

    # Activate Smart Parser
    df_P5KY = smartParser(df, "none", 0)

    # drop
    df_NB33 = df_P5KY.drop("Id", "ExterQual", "BsmtQual",
                        "GrLivArea", "FireplaceQu", "GarageCond", "PoolQC")

    # dropNaCol
    df_k076 = df_NB33.select("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "Heating",
                            "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition", "SalePrice")

    # nearZeroVar
    df_DzoD = df_k076.select("MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                            "BsmtFinType2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir", "Electrical", "1stFlrSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "PavedDrive", "Fence", "MiscFeature", "MoSold", "YrSold", "SaleType", "SaleCondition", "SalePrice")

    # fillna
    df_T97V = df_DzoD.fillna("None")

    # cast
    df_B65E = df_T97V.withColumn("SalePrice", df_T97V.SalePrice.cast(DoubleType()))

#     # preprocessing start
#     # sort
#     df_mKx4 = df_vmC0.sort(["Passen gerId"], ascending=[1])

#     # dataset
#     df_ltXK = df_mKx4

#     # Wrangler
#     df_MouT = df_ltXK

#     # Wrangler - cast
#     df_MouT = df_ltXK.withColumn("SibSp",df_ltXK.SibSp.cast(StringType()))

#     # Wrangler - dropna
#     df_MouT = df_MouT.dropna("all", subset=("Name"))

#     # Wrangler - dropna
#     df_MouT = df_MouT.dropna("all", subset=("Cabin"))

#     # dataset
#     df_r7Lu = df_MouT

#     # preprocessing end
    prep = df_B65E

    return prep.toPandas()
