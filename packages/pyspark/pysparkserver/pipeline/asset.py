# -*- coding: utf-8 -*-
import operator, datetime, calendar
import xml.etree.ElementTree as ET
import pandas as pd
import re, math, json, numpy
import requests

from collections import defaultdict
from random import randint, sample, choice
from pandas_profiling import ProfileReport

from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf, regexp_replace, col, when, lit, count, isnull
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql import Row
from pygrok import Grok

def isNumber(num):
    try:
        res = str(float(num))
        return False if(res == "nan" or res == "inf" or res == "-inf") else True
    except ValueError:
        return False
    except TypeError:
        return False

def naCheck(t, p):
    naCnt = 0
    for l in t[1]:
        if type(l) is str:
            if l == None or l == "":
                naCnt = naCnt + 1
        else:
            if isNumber(l) == False:
                naCnt = naCnt + 1
    if naCnt / float(len(t[1])) >= p / 100.0:
        print("[" + str(t[0]) + "] is checked " + str(p) + "% NA.")
        return (t[0], True)
    else:
        return (t[0], False)

def constantCheck(t, n, p):
    cons = {}
    for l in t[1]:
        if l in cons:
            cons[l] = cons[l] + 1
        else:
            cons[l] = 1
    if len(cons) < n:
        n = len(cons)
    cons_sorted = sorted(cons.items(), key=operator.itemgetter(1), reverse=True)
    total = 0
    index = 0
    constantList = []
    for (k, v) in cons_sorted:
        total = total + v
        index = index + 1
        constantList.append(k)
        if index == n:
            break
    if total / float(len(t[1])) >= p / 100.0:
        print("[" + str(t[0]) + "] is checked " + str(p) + "% constant: " + ",".join(str(c) for c in constantList))
        return (t[0], True)
    return (t[0], False)

def saturationCheck(t, p):
    existSaturation = []
    try:
        vals = {}
        for l in t[1]:
            if l in vals:
                vals[l] = vals[l] + 1
            else:
                vals[l] = 1
        vals_sorted = sorted(vals.items(), key=operator.itemgetter(0), reverse=True)
        saturationKey = vals_sorted[0][0]
        saturationVal = vals_sorted[0][1]
        if saturationVal / float(len(t[1])) >= p / 100.0:
            print("[" + str(t[0]) + "] is checked " + str(p) + "% MAX saturation: " + str(saturationKey))
            existSaturation.append(True)
        vals_sorted = sorted(vals.items(), key=operator.itemgetter(0), reverse=False)
        saturationKey = vals_sorted[0][0]
        saturationVal = vals_sorted[0][1]
        if saturationVal / float(len(t[1])) >= p / 100.0:
            print("[" + str(t[0]) + "] is checked " + str(p) + "% MIN saturation: " + str(saturationKey))
            existSaturation.append(True)
    except:
        print("ERROR: [" + str(t[0]) + "]")
    return (t[0], any(existSaturation))

def addColumn(dict, col):
    keyTuple = {}
    for (key, value) in dict.items():
        keyTuple[(dict[col], key)] = value
    return keyTuple

def filterNaRow(dict, col, t, p):
    if dict[col] / float(t) >= p / 100.0:
        return False
    else:
        return True

def nearZeroVar(t, freqCut, uniqueCut):
    uniqueVal = set()
    countVal = defaultdict(lambda: 0)
    total = 0
    for l in t[1]:
        try:
            total += l
            uniqueVal.add(l)
            countVal[l] = countVal[l] + 1
        except TypeError:
            print("[" + str(t[0]) + "] TypeError")
            return (t[0], False)
    isNzv = False
    isZero = False
    countVal_sorted = sorted(countVal.items(), key=operator.itemgetter(1), reverse=True)
    if len(countVal_sorted) > 1:
        freqRatio = countVal_sorted[0][1] / float(countVal_sorted[1][1])
        if freqRatio >= freqCut:
            isNzv = True
    else:
        freqRatio = 0
        isZero = True
        isNzv = True
    uniqueRatio = len(uniqueVal) / float(len(t[1]))
    if uniqueRatio >= uniqueCut:
        isNzv = True
    print("[" + str(t[0]) + "]" + " freqRatio=" + str("%.6f" % freqRatio) + ", percentUnique=" + str("%.7f" % uniqueRatio) + ", zeroVar=" + str(isZero) + ", nzv=" + str(isNzv))
    return (t[0], isNzv)

def convertDT(timeCal, timeFormat):
    timeCal = timeCal.replace(" ","").replace("now","")
    calTarget = list(filter(None, re.split('([hdwM])',timeCal)))
    dateDict = dict(h='hours', d='days', w='weeks', M='months')
    formatDict = dict(yyyy='%Y', MM='%m', dd='%d', HH='%H', mm='%M', ss='%S')
    for key, value in formatDict.items():
        timeFormat = timeFormat.replace(key, value)
    resultDT = datetime.datetime.now()
    for i in range(0,len(calTarget),2):
        if calTarget[i+1] == 'h':
            resultDT = resultDT + datetime.timedelta(hours=int(calTarget[i]))
        elif calTarget[i+1] == 'd':
            resultDT = resultDT + datetime.timedelta(days=int(calTarget[i]))
        elif calTarget[i+1] == 'w':
            resultDT = resultDT + datetime.timedelta(weeks=int(calTarget[i]))
        else:
            resultDT = datetime.datetime.combine(addMonths(resultDT, int(calTarget[i].replace("M",""))), resultDT.time())
    resultDTString = resultDT.strftime(timeFormat)
    return resultDTString

def addMonths(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)

def getSimilarity(df1, df2, columnName1, columnName2):
    df1_column=df1.select(columnName1).dropDuplicates()
    df2_column=df2.select(columnName2).dropDuplicates()
    
    subCount=df2_column.subtract(df1_column).count()
    
    df2_column_count=df2_column.count()
    return (df2_column_count - subCount)/df2_column_count * 100

def getSimilarityByPandas(df1, df2, columnName1, columnName2):
    df1_column=df1.select(columnName1).toPandas().rename(columns={columnName1: "target"}).drop_duplicates()
    df2_column=df2.select(columnName2).toPandas().rename(columns={columnName2: "target"}).drop_duplicates()
    
    df = pd.concat([df1_column, df2_column])
    df = df.reset_index(drop=True)
    df_gpby = df.groupby(list(df.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) != 1]
    df = df.reindex(idx)
    
    return (df.count()/df2_column.count() * 100).target

def getSimilarities(df1, df2, count):
    result=[]
    result_item={}
    df1_size=df1.count()
    df2_size=df2.count()

    if (df1_size < 1000):
        df1_sample=df1
    else:
        df1_sample=df1.sample(False, 1000/df1_size, 1)

    if (df2_size < 1000):
        df2_sample=df2
    else:
        df2_sample=df2.sample(False, 1000/df2_size, 1)

    for df1_column_name in df1.columns:
        df_1column_name_type=getType(df1_sample, df1_column_name)
        for df2_column_name in df2.columns:
            result_item={}
            if(df_1column_name_type==getType(df2_sample, df2_column_name)):
                result_item["col1"]=df1_column_name
                result_item["col2"]=df2_column_name
                result_item["similarity"]=getSimilarityByPandas(df1_sample, df2_sample, df1_column_name, df2_column_name)
                result.append(result_item)

    return sorted(result, key=lambda t: t["similarity"], reverse=True)[0:count]

def getType(df, colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]

def systematicSampling(df, count, sqlContext):
    result = []
    samplingInterval = int(df.count() / count)
    startingIndex = randint(0, samplingInterval - 1)
    for idx in range(count):
        if (idx == 0):
            result.append(startingIndex)
        else:
            result.append(startingIndex+(samplingInterval*idx))
    return sqlContext.createDataFrame(df.rdd.zipWithIndex().filter(lambda r: True if r[1] in result else False).map(lambda r: r[0]), df.schema)

def stratifiedByRatio(df, col, replacement, fraction, seed, sqlContext):
    fractions = df.rdd.map(lambda x: x[col]).distinct().map(lambda x: (x, fraction)).collectAsMap()
    stratum = df.rdd.keyBy(lambda x: x[col])
    sampled = stratum.sampleByKey(replacement, fractions, seed)
    if sampled.isEmpty():
        return sqlContext.createDataFrame(sc.emptyRDD(), df.schema)
    else:
        return sqlContext.createDataFrame(sampled.map(lambda x: x[1]), df.schema)

def stratifiedSampling(df, col, replacement, count, sqlContext):
    stratum = df.rdd.map(lambda r: (r[col], r)).groupByKey().mapValues(list)
    sampled = stratum.map(lambda t: sampledList(t, replacement, count)).flatMapValues(lambda x: x)
    if sampled.isEmpty():
        return sqlContext.createDataFrame(sc.emptyRDD(), df.schema)
    else:
        return sqlContext.createDataFrame(sampled.map(lambda x: x[1]), df.schema)

def clusterSampling(df, col, fraction):
    result = []
    list_group = df.select(col).distinct().collect()
    samplingSize = math.ceil(df.select(col).distinct().count() * fraction)
    selectGroup = sample(list_group,samplingSize)
    for i in range(samplingSize):
        if i != 0:
            result = result | (df[col]==selectGroup[i][col] )
        else :
            result = (df[col]==selectGroup[i][col])
    return df.filter(result)

def sampledList(t, replacement, count):
    if type(count) is float:
        count = math.ceil(count * len(t[1]))
    else:
        if count > len(t[1]):
            count = len(t[1])
    if replacement:
        list = []
        for i in range(count):
            list.append(choice(t[1]))
        return (t[0], list)
    else:
        return (t[0], sample(t[1], count))

# Outlier detection using Boxplots
def getOutliers(df, cols, sqlContext):
    for col in cols:
        quantiles = df.stat.approxQuantile(col, [0.25, 0.75], 0.001)
        Q1 = quantiles[0]
        Q3 = quantiles[1]
        IQR = Q3 - Q1
        lowerWhisker = Q1 - 1.5*IQR
        upperWhisker = Q3 + 1.5*IQR
        tmpDf=df.filter((df[col] < lowerWhisker) | (df[col] > upperWhisker))
        outliers = tmpDf.rdd.map(lambda r: (r[col])).distinct().collect()
    return sqlContext.createDataFrame(df.rdd.map(lambda r: (r[col], r)).filter(lambda t: False if t[0] in outliers else True).map(lambda t: t[1]), df.schema)

def getOutliersCustom(df, cols, lower, upper, sqlContext):
    for col in cols:
        tmpDf=df.filter((df[col] < lower) | (df[col] > upper))
        outliers = tmpDf.rdd.map(lambda r: (r[col])).distinct().collect()
    return sqlContext.createDataFrame(df.rdd.map(lambda r: (r[col], r)).filter(lambda t: False if t[0] in outliers else True).map(lambda t: t[1]), df.schema)

def minMaxScale(df, min, max, targetColumns):
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = MinMaxScaler(min = min, max = max, inputCol = vectCol, outputCol = outputCol)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    result = pipeline.fit(df).transform(df).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result

def minMaxScaleTest(df, dfTest, min, max, targetColumns):
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = MinMaxScaler(min = min, max = max, inputCol = vectCol, outputCol = outputCol)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    result = model.transform(df).drop(*dropCols)
    test = model.transform(dfTest).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
            test = test.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result, test

def standardScale(df, withMean, withStd, targetColumns):
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = StandardScaler(withMean = withMean, withStd = withStd, inputCol = vectCol, outputCol = outputCol)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    result = pipeline.fit(df).transform(df).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result

def standardScaleTest(df, dfTest, withMean, withStd, targetColumns):
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = StandardScaler(withMean = withMean, withStd = withStd, inputCol = vectCol, outputCol = outputCol)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    result = model.transform(df).drop(*dropCols)
    test = model.transform(dfTest).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
            test = test.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result, test

def robustScaler(df, withCentering, withScaling, targetColumns):
    for col in targetColumns:
        if getType(df, col)!="string":
            quantiles = df.approxQuantile( col, [0.25, 0.75], 0.001 )
            median = df.approxQuantile( col, [0.5], 0.001 )
            IQR = quantiles[1] - quantiles[0]
            if withCentering:
                if withScaling:
                    df = df.withColumn(col, (df[col] - median[0])/IQR)
                else:
                    df = df.withColumn(col, df[col] - median[0])
            else:
                if withScaling:
                    df = df.withColumn(col, df[col]/IQR)
                else:
                    df = df.withColumn(col, df[col])
    return df

def robustScalerTest(df, dfTest, withCentering, withScaling, targetColumns):
    for col in targetColumns:
        if getType(df, col)!="string":
            quantiles = df.approxQuantile( col, [0.25, 0.75], 0.001 )
            median = df.approxQuantile( col, [0.5], 0.001 )
            IQR = quantiles[1] - quantiles[0]
            if withCentering:
                if withScaling:
                    df = df.withColumn(col, (df[col] - median[0])/IQR)
                    dfTest = dfTest.withColumn(col, (dfTest[col] - median[0])/IQR)
                else:
                    df = df.withColumn(col, df[col] - median[0])
                    dfTest = dfTest.withColumn(col, dfTest[col] - median[0])
            else:
                if withScaling:
                    df = df.withColumn(col, df[col]/IQR)
                    dfTest = dfTest.withColumn(col, dfTest[col]/IQR)
                else:
                    df = df.withColumn(col, df[col])
                    dfTest = dfTest.withColumn(col, dfTest[col])
    return df, dfTest

def normalizer(df, targetColumns, norm):
    if norm == "L1":
        float_norm = 1.0
    elif norm == "L2":
        float_norm = 2.0
    else:
        float_norm = float("inf")

    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = Normalizer(inputCol=vectCol, outputCol=outputCol, p=float_norm)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    result = pipeline.fit(df).transform(df).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result

def normalizerTest(df, dfTest, targetColumns, norm):
    if norm == "L1":
        float_norm = 1.0
    elif norm == "L2":
        float_norm = 2.0
    else:
        float_norm = float("inf")

    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = Normalizer(inputCol=vectCol, outputCol=outputCol, p=float_norm)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    result = model.transform(df).drop(*dropCols)
    test = model.transform(dfTest).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
            test = test.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result, test

def maxAbsScale(df, targetColumns):
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = MaxAbsScaler(inputCol = vectCol, outputCol = outputCol)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    result = pipeline.fit(df).transform(df).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result

def maxAbsScaleTest(df, dfTest, targetColumns):
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    stageList = []
    dropCols = []
    for col in targetColumns:
        if getType(df, col)!="string":
            vectCol = col + "_Vect"
            outputCol = col + "_output"
            assembler = VectorAssembler(inputCols=[col], outputCol=vectCol)
            stageList.append(assembler)
            scaler = MaxAbsScaler(inputCol = vectCol, outputCol = outputCol)
            stageList.append(scaler)
            dropCols.append(vectCol)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    result = model.transform(df).drop(*dropCols)
    test = model.transform(dfTest).drop(*dropCols)
    for col in targetColumns:
        if getType(df, col)!="string":
            result = result.withColumn(col, unlist(col+"_output")).drop(col+"_output")
            test = test.withColumn(col, unlist(col+"_output")).drop(col+"_output")
    return result, test

def oneHotEncoder(df, col, newCols, handleInvalid, dropLast):
    inputCols = []
    stringCols = []
    dropCols = []
    stageList = []
    for (name) in list(col):
        if getType(df, name)=="string":
            inputCols.append(name+"_index")
            stringCols.append(name)
            dropCols.append(name+"_index")
        else:
            raise TypeError
    for column in list(stringCols):
        stringIndexer = StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep", stringOrderType="alphabetAsc")
        stageList.append(stringIndexer)
    oneHotEncoderEstimator = OneHotEncoderEstimator(inputCols=inputCols, outputCols=newCols,handleInvalid=handleInvalid, dropLast=dropLast)
    stageList.append(oneHotEncoderEstimator)
    pipeline = Pipeline(stages=stageList)
    return pipeline.fit(df).transform(df).drop(*dropCols)

def oneHotEncoderTest(df, dfTest, col, newCols, handleInvalid, dropLast):
    inputCols = []
    stringCols = []
    dropCols = []
    stageList = []
    for (name) in list(col):
        if getType(df, name)=="string":
            inputCols.append(name+"_index")
            stringCols.append(name)
            dropCols.append(name+"_index")
        else:
            raise TypeError
    for column in list(stringCols):
        stringIndexer = StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep", stringOrderType="alphabetAsc")
        stageList.append(stringIndexer)
    oneHotEncoderEstimator = OneHotEncoderEstimator(inputCols=inputCols, outputCols=newCols,handleInvalid=handleInvalid, dropLast=dropLast)
    stageList.append(oneHotEncoderEstimator)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    train = model.transform(df).drop(*dropCols)
    test = model.transform(dfTest).drop(*dropCols)
    return train, test

def pca(df, cols, newCols, k):
    inputCols = []
    stringCols = []
    dropCols = []
    stageList = []
    newCols = newCols.split(',')
    for name in list(cols):
        if getType(df, name)=="string":
            inputCols.append(name+"_index")
            stringCols.append(name)
            dropCols.append(name+"_index")
        else:
            inputCols.append(name)
    for column in list(stringCols):
        stringIndexName = StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep", stringOrderType="alphabetAsc")
        stageList.append(stringIndexName)
    vecCol = "pca_Vect"
    pcaOutputCol = "pca_output"
    assembler = VectorAssembler(inputCols=inputCols,outputCol=vecCol)
    pca = PCA(k=k, inputCol=vecCol, outputCol=pcaOutputCol)
    stageList.append(assembler)
    stageList.append(pca)
    pipeline = Pipeline(stages=stageList)
    result = pipeline.fit(df).transform(df).drop(vecCol).drop(*dropCols)
    for i in range(k):
        vectorParse = udf(lambda x: float(list(x)[i]), DoubleType())
        result = result.withColumn(newCols[i], vectorParse(pcaOutputCol))
    return result.drop(pcaOutputCol)

def pcaTest(df, dfTest, cols, newCols, k):
    inputCols = []
    stringCols = []
    dropCols = []
    stageList = []
    newCols = newCols.split(',')
    for name in list(cols):
        if getType(df, name)=="string":
            inputCols.append(name+"_index")
            stringCols.append(name)
            dropCols.append(name+"_index")
        else:
            inputCols.append(name)
    for column in list(stringCols):
        stringIndexName = StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep", stringOrderType="alphabetAsc")
        stageList.append(stringIndexName)
    vecCol = "pca_Vect"
    pcaOutputCol = "pca_output"
    assembler = VectorAssembler(inputCols=inputCols,outputCol=vecCol)
    pca = PCA(k=k, inputCol=vecCol, outputCol=pcaOutputCol)
    stageList.append(assembler)
    stageList.append(pca)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    result = model.transform(df).drop(vecCol).drop(*dropCols)
    test = model.transform(dfTest).drop(vecCol).drop(*dropCols)
    for i in range(k):
        vectorParse = udf(lambda x: float(list(x)[i]), DoubleType())
        result = result.withColumn(newCols[i], vectorParse(pcaOutputCol))
        test = test.withColumn(newCols[i], vectorParse(pcaOutputCol))
    return result.drop(pcaOutputCol), test.drop(pcaOutputCol)

def chiSqSelector(df, cols, labelCol, numTopFeatures):
    inputCols = []
    stringCols = []
    stageList = []
    for name in list(cols):
        if getType(df, name)=="string":
            inputCols.append("_index_"+name)
            stringCols.append(name)
        else:
            inputCols.append(name)
    for column in list(stringCols):
        stringIndexName = StringIndexer(inputCol=column, outputCol="_index_"+column, handleInvalid="keep", stringOrderType="alphabetAsc")
        stageList.append(stringIndexName)
    vecCol = "_temp_Vect"
    assembler = VectorAssembler(inputCols=inputCols,outputCol=vecCol)
    chiSqSelector = ChiSqSelector(numTopFeatures=numTopFeatures, featuresCol=vecCol, outputCol='_temp', labelCol=labelCol)
    stageList.append(assembler)
    stageList.append(chiSqSelector)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    importantFeatures = model.stages[len(stageList)-1].selectedFeatures
    return df.drop(*[item for idx, item in enumerate(cols) if idx not in importantFeatures])

def chiSqSelectorTest(df, dfTest, cols, labelCol, numTopFeatures):
    inputCols = []
    stringCols = []
    stageList = []
    for name in list(cols):
        if getType(df, name)=="string":
            inputCols.append("_index_"+name)
            stringCols.append(name)
        else:
            inputCols.append(name)
    for column in list(stringCols):
        stringIndexName = StringIndexer(inputCol=column, outputCol="_index_"+column, handleInvalid="keep", stringOrderType="alphabetAsc")
        stageList.append(stringIndexName)
    vecCol = "_temp_Vect"
    assembler = VectorAssembler(inputCols=inputCols,outputCol=vecCol)
    chiSqSelector = ChiSqSelector(numTopFeatures=numTopFeatures, featuresCol=vecCol, outputCol='_temp', labelCol=labelCol)
    stageList.append(assembler)
    stageList.append(chiSqSelector)
    pipeline = Pipeline(stages=stageList)
    model = pipeline.fit(df)
    importantFeatures = model.stages[len(stageList)-1].selectedFeatures
    train = df.drop(*[item for idx, item in enumerate(cols) if idx not in importantFeatures])
    test = dfTest.drop(*[item for idx, item in enumerate(cols) if idx not in importantFeatures])
    return train, test

def corrSelector(df, targetColumns, label, num):
    columnsForVect = [label]
    stageList = []
    for col in targetColumns:
        if getType(df, col)=="string":
            columnsForVect.append("_corr_"+col)
            strIdx = StringIndexer(inputCol=col, outputCol="_corr_"+col, handleInvalid="keep", stringOrderType="alphabetAsc")
            stageList.append(strIdx)
        else:
            columnsForVect.append(col)
    assembler = VectorAssembler(inputCols=columnsForVect, outputCol="features")
    stageList.append(assembler)
    pipeline = Pipeline(stages=stageList)
    dfTmp = pipeline.fit(df).transform(df)
    dfTmpForCorr = dfTmp.select('features')
    correlation = Correlation.corr(dfTmpForCorr, "features", "pearson").collect()[0][0].toArray()
    corrDict = {}
    #correlation[0][1:] remove label column,
    for i, corr in enumerate(correlation[0][1:]):
        # print('{} : {} : {}'.format(i, columnsForVect[i+1], abs(corr)))
        # because remove label column, column value + 1
        corrDict[columnsForVect[i+1]] = abs(corr)
    sortedCorrDict = sorted(corrDict.items(), key=lambda x: x[1], reverse=True)
    selectedFeatures = []
    for i in range(num):
        selectedFeatures.append(sortedCorrDict[i][0].replace("_corr_", ""))
    return df.drop(*[item for item in targetColumns if item not in selectedFeatures])

def corrSelectorTest(df, dfTest, targetColumns, label, num):
    columnsForVect = [label]
    stageList = []
    for col in targetColumns:
        if getType(df, col)=="string":
            columnsForVect.append("_corr_"+col)
            strIdx = StringIndexer(inputCol=col, outputCol="_corr_"+col, handleInvalid="keep", stringOrderType="alphabetAsc")
            stageList.append(strIdx)
        else:
            columnsForVect.append(col)
    assembler = VectorAssembler(inputCols=columnsForVect, outputCol="features")
    stageList.append(assembler)
    pipeline = Pipeline(stages=stageList)
    dfTmp = pipeline.fit(df).transform(df)
    dfTmpForCorr = dfTmp.select('features')
    correlation = Correlation.corr(dfTmpForCorr, "features", "pearson").collect()[0][0].toArray()
    corrDict = {}
    #correlation[0][1:] remove label column,
    for i, corr in enumerate(correlation[0][1:]):
        # print('{} : {} : {}'.format(i, columnsForVect[i+1], abs(corr)))
        # because remove label column, column value + 1
        corrDict[columnsForVect[i+1]] = abs(corr)
    sortedCorrDict = sorted(corrDict.items(), key=lambda x: x[1], reverse=True)
    selectedFeatures = []
    for i in range(num):
        selectedFeatures.append(sortedCorrDict[i][0].replace("_corr_", ""))
    train = df.drop(*[item for item in targetColumns if item not in selectedFeatures])
    test = dfTest.drop(*[item for item in targetColumns if item not in selectedFeatures])
    return train, test

def treeSelector(df, targetColumns, label, num, numTrees, maxDepth, seed, maxBins):
    columnsForVect = []
    stageList = []
    for col in targetColumns:
        if getType(df, col) == "string":
            columnsForVect.append("_stx_"+col)
            strIdx = StringIndexer(inputCol=col, outputCol="_stx_"+col, handleInvalid="keep", stringOrderType="alphabetAsc")
            stageList.append(strIdx)
        else:
            columnsForVect.append(col)
    assembler = VectorAssembler(inputCols=columnsForVect, outputCol="features")
    stageList.append(assembler)
    pipeline = Pipeline(stages=stageList)
    dfTmp = pipeline.fit(df).transform(df).select('features', label)
    rf = RandomForestClassifier(numTrees=numTrees, maxDepth=maxDepth, labelCol=label, seed=seed, maxBins=maxBins)
    model = rf.fit(dfTmp)
    importanceDict = {}
    for i, importance in enumerate(model.featureImportances.toArray()):
        importanceDict[targetColumns[i]] = importance
    sortedImportanceDict = sorted(importanceDict.items(), key=lambda x: x[1], reverse=True)
    selectedFeatures = []
    for i in range(num):
        selectedFeatures.append(sortedImportanceDict[i][0].replace("_stx_", ""))
    return df.drop(*[item for item in targetColumns if item not in selectedFeatures])

def treeSelectorTest(df, dfTest, targetColumns, label, num, numTrees, maxDepth, seed, maxBins):
    columnsForVect = []
    stageList = []
    for col in targetColumns:
        if getType(df, col) == "string":
            columnsForVect.append("_stx_"+col)
            strIdx = StringIndexer(inputCol=col, outputCol="_stx_"+col, handleInvalid="keep", stringOrderType="alphabetAsc")
            stageList.append(strIdx)
        else:
            columnsForVect.append(col)
    assembler = VectorAssembler(inputCols=columnsForVect, outputCol="features")
    stageList.append(assembler)
    pipeline = Pipeline(stages=stageList)
    dfTmp = pipeline.fit(df).transform(df).select('features', label)
    rf = RandomForestClassifier(numTrees=numTrees, maxDepth=maxDepth, labelCol=label, seed=seed, maxBins=maxBins)
    model = rf.fit(dfTmp)
    importanceDict = {}
    for i, importance in enumerate(model.featureImportances.toArray()):
        importanceDict[targetColumns[i]] = importance
    sortedImportanceDict = sorted(importanceDict.items(), key=lambda x: x[1], reverse=True)
    selectedFeatures = []
    for i in range(num):
        selectedFeatures.append(sortedImportanceDict[i][0].replace("_stx_", ""))
    train = df.drop(*[item for item in targetColumns if item not in selectedFeatures])
    test = dfTest.drop(*[item for item in targetColumns if item not in selectedFeatures])
    return train, test

def convertNum(s):
    try:
        int(s)
        return int(s)
    except ValueError:
        try:
            float(s)
            return float(s)
        except ValueError:
            return s

def hasXmlns(s, root):
 if 'xmlns=' in s and '}' in root.tag:
  pl = root.tag.split('}')
  p = pl[0][1:]
  return ET.XML(s.replace('xmlns=\''+p+'\'', ''))
 else:
  return ET.XML(s)

def xmlParsing(root, tags):
    res = []
    for i,child in enumerate(root):
     rec = {}
     for tag in tags:
      if child.iter(tag) != None:
       tmp = ''
       for a in child.iter(tag):
        if a.text != None and len(a.text.strip()) > 0:
         tmp += a.text + ','
        else:
          for val in a.attrib.values():
           tmp += val + '|'
       tmp = tmp[0:-1]
       rec[tag] = convertNum(tmp)
     res.append(rec)
    return res

def getColumnsProfile(df, start, length):
    df = df.select(df.columns[start:start+length])
    pdf = df.toPandas()
    profile = ProfileReport(pdf, minimal=True, vars={'cat':{'check_composition':True}})
    result = profile.get_description()
    refinedDict = refine(result)
    return json.dumps(refinedDict, default=json_default)

def getColumnProfileByName(df, col):
    df = df.select(col)
    pdf = df.toPandas()
    profile = ProfileReport(pdf, minimal=True, vars={'cat':{'check_composition':True}})
    result = profile.get_description()
    refinedDict = refine(result)
    return json.dumps(refinedDict, default=json_default)

def stringify_keys(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise
            del d[key]
    return d

# Gets called for objects that can't otherwise be serialized.
def json_default(value):
    if isinstance(value, pd.core.series.Series):
        return value.to_json()
    elif isinstance(value, numpy.int64):
        return int(value)
    else:
        return str(value)

# To delete unnecessary key values
def refine(dic):
    for key in dic['variables']:
        col = dic['variables'][key]
        if 'chi_squared' in col.keys():
            col['chi_squared'] = [(None if math.isnan(i) else i) for i in col['chi_squared']]
        if 'value_counts_without_nan' in col.keys():
            del col['value_counts_without_nan']
        if 'value_counts_with_nan' in col.keys():
            del col['value_counts_with_nan']
        if (str(col['type']) != "Variable.TYPE_NUM"):
            col['value_counts'] = dict(list(col['value_counts'].items())[0:10])
        if (str(col['type']) == "Variable.TYPE_PATH"):
            #del col['parent_counts']
            col['stem_counts'] = dict(list(col['stem_counts'].items())[0:10])
            col['name_counts'] = dict(list(col['name_counts'].items())[0:10])
            col['suffix_counts'] = dict(list(col['suffix_counts'].items())[0:10])
            col['parent_counts'] = dict(list(col['parent_counts'].items())[0:10])
            stringify_keys(col)
        elif (str(col['type']) == "Variable.TYPE_DATE"):
            stringify_keys(col)
            del col['histogram_data']
        elif (str(col['type']) == "Variable.TYPE_URL"):
            col['scheme_counts'] = dict(list(col['scheme_counts'].items())[0:10])
            col['netloc_counts'] = dict(list(col['netloc_counts'].items())[0:10])
            col['path_counts'] = dict(list(col['path_counts'].items())[0:10])
            col['query_counts'] = dict(list(col['query_counts'].items())[0:10])
            col['fragment_counts'] = dict(list(col['fragment_counts'].items())[0:10])
        elif (str(col['type']) == "Variable.TYPE_NUM"):
            #if 'scatter_data' in col.keys():
            del col['scatter_data']
            del col['histogram_data']
    for key in list(dic.keys()):
        if key == 'table':
            del dic['table']
        elif key == 'scatter':
            del dic['scatter']
        elif key == 'correlations':
            del dic['correlations']
        elif key == 'missing':
            del dic['missing']
        elif key == 'package':
            del dic['package']
        elif key == 'messages':
            del dic['messages']
    return dic

def fill_missing_value(df, targetColumns, constantVale=""):
    newConstantValue = ''
    for column in targetColumns:
        if(constantVale == ""):
            newConstantValue = df.toPandas()[column].mode()[0]
        else:
            newConstantValue = constantVale
        df = df.withColumn(column, when(df[column].isNull()==True, lit(newConstantValue)).otherwise(df[column]).cast(getType(df,column)))
    return df

def detectDelimiter(target):
    delimiters = [',', '\t', '|', ';', '^', '\u0001']
    selection = ''
    for d in delimiters:
        before_cnt = -1
        for l in target:
            res = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1','',l)
            current_cnt = res.count(d)
            if current_cnt == 0:
                selection = ''
                break
            if before_cnt != -1 and before_cnt != current_cnt:
                selection = ''
                break
            selection = d
            before_cnt = current_cnt
        if selection != '':
            break
    return selection

def smartParser(df, sampling, limit):
    if limit > 10000000:
        limit = 10000000
    columns = df.dtypes
    if sampling == 'random':
        cell_count = len(columns) * df.count()
        if cell_count > limit:
            ratio = limit / cell_count
            df = df.sample(True, ratio)
    elif sampling == 'limit':
        row_count = limit / len(columns)
        df = df.limit(int(row_count))
    return df

def getDatasetInfo(df):
    p_cells_missing = 0.0
    p_duplicates = 0.0
    df_size = df.count()

    column_list = df.columns
    column_size = len(column_list)
    n_duplicates = df_size - df.distinct().count()
    n_cells_missing = df.rdd.map(lambda x: getMissingCells(x, column_size)).sum()

    if n_duplicates > 0:
        p_duplicates = n_duplicates/df_size
    if n_cells_missing > 0:
        p_cells_missing = n_cells_missing/(df_size * len(column_list))

    result = {}
    result["column_list"] = column_list
    result["type_list"] = [i[1] for i in df.dtypes]
    result["n_duplicates"] = n_duplicates
    result["n_cells_missing"] = n_cells_missing
    result["p_duplicates"] = p_duplicates
    result["p_cells_missing"] = p_cells_missing
    return result

def getMissingCells(row, column_size):
    missing_cnt = 0
    for i in range(column_size):
        if row[i] == None:
            missing_cnt = missing_cnt + 1
    return missing_cnt

def createNewColumnByGrok(text, grok, columnNames):
    row_list = []
    matchResult = grok.match(text)
    if isinstance(matchResult, type(None)):
        for c in columnNames:
            row_list.append("")
    else:
        for columnName in matchResult:
            row_list.append(matchResult[columnName])
    return row_list

def getMissingValueColumns(df, dfString):
    target_columns = []
    for k in df.columns:
        if getType(df, k).startswith(('double', 'int', 'float', 'bigint', 'smallint', 'decimal')):
            target_columns.append(k)
    df_null_count = df.select([count(when(isnull(c), c)).alias(c) for c in target_columns])
    result = {}
    result["missing_columns"] = [dfString+'.'+ key for (key, value) in df_null_count.collect()[0].asDict().items() if value > 0]
    return result

def getCorrelationByPandas(df, cols, methodType):
    if cols != None:
        pdf = df.select(cols).toPandas()
    else:
        pdf = df.toPandas()
    corr = pdf.corr(method = methodType)
    return corr

def getCorrelation(df, cols, methodType):
    result = {}
    colList = []
    finalColList = []
    stageList = []
    dropCols = []
    df = df.dropna("any")
    if cols == None:
        return result
    for col in cols:
        colType = getType(df, col)
        if colType == "tinyint" or colType == "smallint" or colType == "int" or colType == "bigint" or colType == "float" or colType == "double" or "decimal" in colType:
            colList.append(col)
            finalColList.append(col)
        elif colType == "string":
            colList.append("_corr_"+col)
            strIdx = StringIndexer(inputCol=col, outputCol="_corr_"+col, handleInvalid="keep", stringOrderType="alphabetAsc")
            stageList.append(strIdx)
            dropCols.append(col)
            finalColList.append(col)
    assembler = VectorAssembler(inputCols=colList, outputCol="features")
    stageList.append(assembler)
    pipeline = Pipeline(stages=stageList)
    dfTmp = pipeline.fit(df).transform(df)
    for renameCol in dropCols:
        dfTmp = dfTmp.drop(renameCol).withColumnRenamed("_corr_"+renameCol,renameCol)
    corr = Correlation.corr(dfTmp, "features", methodType)
    result["column_list"] = finalColList
    result["corr"] = corr.head()[0].toArray().tolist()
    return result

def castForImputer(df, inputCols):
    target_columns = []
    for k in inputCols:
        if getType(df, k).startswith(('int', 'bigint', 'smallint', 'decimal')):
            target_columns.append(k)
    for col_name in target_columns:
        df = df.withColumn(col_name, col(col_name).cast('double'))
    return df

def upDownSampling(df, target, label_0, label_1, method):
    df_label_0 = df.filter(df[target] == label_0)
    df_label_1 = df.filter(df[target] == label_1)
    df_label_x = df.filter((df[target] != label_0) & (df[target] != label_1))
    cnt_0 = df_label_0.count()
    cnt_1 = df_label_1.count()
    if cnt_0 < cnt_1:
        if method == 'up':
            return df_label_0.sample(withReplacement=True, fraction=cnt_1/cnt_0).union(df_label_1).union(df_label_x)
        else:
            return df_label_1.sample(withReplacement=False, fraction=cnt_0/cnt_1).union(df_label_0).union(df_label_x)
    elif cnt_0 > cnt_1:
        if method == 'up':
            return df_label_1.sample(withReplacement=True, fraction=cnt_0/cnt_1).union(df_label_0).union(df_label_x)
        else:
            return df_label_0.sample(withReplacement=False, fraction=cnt_1/cnt_0).union(df_label_1).union(df_label_x)
    else:
        return df

def getMlresult(path, type, sqlContext):
    df = sqlContext.read.format(type).option("header", "false").option("inferSchema", "true").load(path)
    df1 = df.toDF(*list(map(lambda x: x.replace(".", "_"), df.columns)))
    return ",".join(df1.na.fill("null").na.fill(float("nan")).toJSON().take(100))

def exportDataCatalog(url, body, userId, df):
    headers = {
        "userId": userId,
        "Content-Type": "application/json"
    }
    schemaList = []
    schema = {}

    for filed in df.schema:
        schema['name'] = filed.name
        schema['type'] = filed.dataType.simpleString()
        schemaList.append(schema)
        schema = {}

    body["schemas"] = schemaList

    try:
        response = requests.post(url, headers=headers, data=json.dumps(body))
    except:
        raise DataCatalogException("Can not connect to DataCatalog server. URL = " + url)
    if(response.status_code != 200):
        raise DataCatalogException("DataCatalog Http error. HTTP status = " + str(response.status_code))

    print(response.text)

def getInteractions(df, col_x, cols_y1, cols_y2, value_type):

    totoal_list = {}
    limitResult = 100000

    # y axis 2
    if cols_y1 != "" and cols_y2 != "":
        df = df.select(col_x,cols_y1,cols_y2).dropna("any")

        if value_type == "SUM":
            grouped = df.groupBy(col_x).agg(F.sum(cols_y1).alias(cols_y1), F.sum(cols_y2).alias(cols_y2)).orderBy(col_x)
        elif value_type == "MEAN":
            grouped = df.groupBy(col_x).agg(F.avg(cols_y1).alias(cols_y1), F.avg(cols_y2).alias(cols_y2)).orderBy(col_x)
        elif value_type == "MIN":
            grouped = df.groupBy(col_x).agg(F.min(cols_y1).alias(cols_y1), F.min(cols_y2).alias(cols_y2)).orderBy(col_x)
        elif value_type == "MAX":
            grouped = df.groupBy(col_x).agg(F.max(cols_y1).alias(cols_y1), F.max(cols_y2).alias(cols_y2)).orderBy(col_x)
        elif value_type == "MEDIAN":
            expression1 = 'percentile(' + cols_y1.replace('"','') + ', array(0.5))'
            expression2 = 'percentile(' + cols_y2.replace('"','') + ', array(0.5))'
            grouped = df.groupBy(col_x).agg(F.expr(expression1)[0].alias(cols_y1), F.expr(expression2)[0].alias(cols_y2)).orderBy(col_x)

        for colResult in grouped.collect():
            if len(totoal_list) * 2 < limitResult :
                count_list = {}
                count_list[cols_y1] = round(colResult[cols_y1],2)
                count_list[cols_y2] = round(colResult[cols_y2],2)
                totoal_list[str(colResult[col_x])] = count_list
    # y axis 1
    elif cols_y1 != "":
        df = df.select(col_x,cols_y1).dropna("any")

        #  y axis type 'timestamp', 'boolean', 'vector','string'
        if getType(df, cols_y1).startswith(('timestamp', 'boolean', 'vector','string')):

            count_list = {}
            default_list = {}
            if df.select(cols_y1).distinct().count() <= 20 :
                for y_values in df.select(cols_y1).distinct().rdd.flatMap(lambda x: x).collect():
                    if y_values not in default_list.keys():
                        default_list[y_values] = 0
            else :
                return 'Column y (StringType) values must be less than 20'

            grouped = df.groupBy(col_x,cols_y1).agg(F.count("*").alias("count")).orderBy(col_x,cols_y1)

            for colResult in grouped.collect():
                if len(totoal_list) * len(default_list) < limitResult :
                    if colResult[col_x] not in totoal_list.keys():
                        count_list = default_list.copy()
                    count_list[colResult[cols_y1]] = round(colResult["count"],2)
                    totoal_list[str(colResult[col_x])] = count_list
        #  y axis type Numeric
        else:
            if value_type == "SUM":
                grouped = df.groupBy(col_x).agg(F.sum(cols_y1).alias(cols_y1)).orderBy(col_x)
            elif value_type == "MEAN":
                grouped = df.groupBy(col_x).agg(F.avg(cols_y1).alias(cols_y1)).orderBy(col_x)
            elif value_type == "MIN":
                grouped = df.groupBy(col_x).agg(F.min(cols_y1).alias(cols_y1)).orderBy(col_x)
            elif value_type == "MAX":
                grouped = df.groupBy(col_x).agg(F.max(cols_y1).alias(cols_y1)).orderBy(col_x)
            elif value_type == "MEDIAN":
                expression = 'percentile(' + cols_y1.replace('"','') + ', array(0.5))'
                grouped = df.groupBy(col_x).agg(F.expr(expression)[0].alias(cols_y1)).orderBy(col_x)

            for colResult in grouped.collect():
                if len(totoal_list) < limitResult :
                    count_list = {}
                    count_list[cols_y1] = round(colResult[cols_y1],2)
                    totoal_list[str(colResult[col_x])] = count_list
    # x axis only
    else:
        df = df.select(col_x).fillna(0).fillna("None")
        grouped = df.groupBy(col_x).count()

        for colResult in grouped.collect():
            if len(totoal_list) < limitResult :
                count_list = {}
                count_list["count"] = round(colResult["count"],2)
                totoal_list[str(colResult[col_x])] = count_list

    return totoal_list

def getInteractionsForScatter(df, col_x, cols_y):
    if col_x != cols_y:
        df = df.select(col_x,cols_y)
    else:
        df = df.select(col_x)
    pdf = df.toPandas()
    profile = ProfileReport(pdf, minimal=True, interactions={'continuous': True})
    result = profile.get_description()
    refinedDict = ''
    for key in result['scatter']:
        if key == col_x:
            y_values = result['scatter'][key]
            if cols_y in y_values.keys():
                refinedDict = y_values[cols_y]
    return refinedDict

def getInteractionsForBoxPlot(df, col_x, cols_y, cols):
    totoal_list = []
    limitResult = 100000

    # y axis 1
    if cols_y != "":
        df = df.select(col_x,cols_y).dropna("any")

        expression1 = 'percentile(' + cols_y.replace('"','') + ', array(0.75))'
        expression2 = 'percentile(' + cols_y.replace('"','') + ', array(0.50))'
        expression3 = 'percentile(' + cols_y.replace('"','') + ', array(0.25))'
        grouped = df.groupBy(col_x).agg(F.expr(expression1)[0].alias("75%"),F.expr(expression2)[0].alias("median"),F.expr(expression3)[0].alias("25%")).orderBy(col_x)

        for colResult in grouped.collect():
            if len(totoal_list) < limitResult :
                count_list = {}
                count_list[col_x] = str(colResult[col_x])
                count_list["upperWhisker"] = round(colResult["75%"] + 1.5*(colResult["75%"] - colResult["25%"]),2)
                count_list["75%"] = round(colResult["75%"],2)
                count_list["median"] = round(colResult["median"],2)
                count_list["25%"] = round(colResult["25%"],2)
                count_list["lowerWhisker"] = round(colResult["25%"] - 1.5*(colResult["75%"] - colResult["25%"]),2)
                totoal_list.append(count_list)

    # x axis only
    else:
        for col in cols:
            quantiles = df.stat.approxQuantile(col, [0.25, 0.5, 0.75], 0.001)
            if len(totoal_list) < limitResult :
                count_list = {}
                count_list["Column"] = col
                count_list["upperWhisker"] = round(quantiles[2] + 1.5*(quantiles[2] - quantiles[0]),2)
                count_list["75%"] = round(quantiles[2],2)
                count_list["median"] = round(quantiles[1],2)
                count_list["25%"] = round(quantiles[0],2)
                count_list["lowerWhisker"] = round(quantiles[0] - 1.5*(quantiles[2] - quantiles[0]),2)
                totoal_list.append(count_list)

    return totoal_list

class DataCatalogException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
