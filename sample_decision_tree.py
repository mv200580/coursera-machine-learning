import pandas as pd
import numpy as np
import pyodbc

desired_width = 1024
pd.set_option('display.width', desired_width)

# db = pyodbc.connect(
#     'Driver={Adaptive Server Enterprise};Server=161.8.44.175;Port=4100;Database=kkc;Uid=starodubtsev_mv;Pwd=107538;',
#     autocommit=True)
# cursor = db.cursor()
#
# store = pd.HDFStore('sample_decision_tree.h5')
#
# # получаем исходные данные из БД плавки
# df = pd.read_sql("""
# select mark,
#         (select sum(weight_t)*1000 from  mtl.ferr_fact f where m.melt_num=f.melt_num and m.melt_year=f.melt_year /*and dept='КО'*/ and mtl='ФМН78') as FMN78,
#         (select sum(weight_t)*1000 from  mtl.ferr_fact f where m.melt_num=f.melt_num and m.melt_year=f.melt_year /*and dept='КО'*/ and mtl in ('СМН17', 'ФСМнС17_ДР', 'СМН18')) as SMN17,
#         0 as SMN18,
#         (select sum(weight_t)*1000 from  mtl.ferr_fact f where m.melt_num=f.melt_num and m.melt_year=f.melt_year /*and dept='КО'*/ and mtl in ('ФМН95', 'ФМН95ДР')) as FMN95,
#         (select sum(weight_t)*1000 from  mtl.ferr_fact f where m.melt_num=f.melt_num and m.melt_year=f.melt_year /*and dept='КО'*/ and mtl='ФС65') as FS65
#     from    mtl.ferr_melt m
#     where 1=1
#         --and melt_year=2016
#         and date_uchet >= '2015.06.01' and date_uchet < '2016.02.01'
# """, db)
# store["data"] = df
# print('Done')
#
# store.close()

store = pd.HDFStore('sample_decision_tree.h5')
data = store["data"]
data = data.fillna(0)
store.close()

# разделяем признаки и целевую переменную
ferr = pd.DataFrame(data, columns = ['FMN78', 'SMN17', 'SMN18', 'FMN95', 'FS65'])
mark = pd.DataFrame(data, columns = ['mark'])

print(data[data.mark == '09Г2С'])

# преобразуем mark в массив индексов
from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer(min_df=1, lowercase=False, token_pattern = '.+' )
mark_vect = countVectorizer.fit_transform(mark['mark'])
mark_ind = mark_vect.indices
mark_name = countVectorizer.get_feature_names()

print(mark['mark'].__len__())
print(mark_ind.__len__())



from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(ferr,mark_ind)

# определяем важность признаков
importances = clf.feature_importances_
print(importances)

i = clf.predict([[386, 578, 0, 0, 0]])  # 08пс
print(i)
print(mark_name[i])

i = clf.predict([[321, 728, 0, 0, 0]])  # SAE 1006
print(i)
print(mark_name[i])

i = clf.predict([[422, 0, 0, 0, 0]])    # 006/IF
print(i)
print(mark_name[i])

i = clf.predict([[2627, 5353, 0, 0, 1950]])  # 09Г2С
print(i)
print(mark_name[i])

i = clf.predict([[312, 801, 0, 0, 0]])  # SAE 1006-А
print(i)
print(mark_name[i])

i = clf.predict([[1212, 568, 0, 0, 0]])  # DC01
print(i)
print(mark_name[i])

i = clf.predict([[324, 646, 0, 0, 0]])  # SAE 1006-А
print(i)
print(mark_name[i])

i = clf.predict([[579, 693, 0, 0, 0]])  # 08пс
print(i)
print(mark_name[i])

# d = [
#     '08пс 34 34 44',
#     '08пс',
#     'Ст3сп',
#     '08ю',
#     '08ю',
#     'Ст3сп',
#     'Ст3сп',
#     '08пс',
#     '08пс',
#     'Ст3сп',
#     '08ю',
#     '08ю',
#     'Ст3сп',
#     'Ст3сп'
# ]
#
# from sklearn.feature_extraction.text import CountVectorizer
# countVectorizer = CountVectorizer(min_df=1, lowercase=False, token_pattern = '.+' )
# mark_vect = countVectorizer.fit_transform(d)
# print(mark_vect)
# print(d.__len__())
# print(mark_vect.indices)
# print(mark_vect.indices.__len__())
# print(countVectorizer.get_feature_names())
#
#




