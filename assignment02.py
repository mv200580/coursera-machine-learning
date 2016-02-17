# Assignment: Важность признаков
# https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/DVbEI/vazhnost-priznakov

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')

# удаляем данные с NaN значениями
data_cleaned = data[np.isfinite(data.Pclass) & np.isfinite(data.Fare) & np.isfinite(data.Age)]

# выделяем нужные столбцы
data_limited = pd.DataFrame(data_cleaned, columns = ['Pclass', 'Fare', 'Age', 'Sex'])

# преобразуем sex=['male', 'female'] в [1, 0]
data_limited['Sex'] = data_limited['Sex'].apply(lambda x: 1 if x == 'male' else 0)
print(data_limited)

# целевая переменная
data_target = pd.DataFrame(data_cleaned, columns = ['Survived'])
print(data_target)

# обучение решаюшего дерева
clf = DecisionTreeClassifier()
clf.fit(data_limited,data_target)

# определяем важность признаков
importances = clf.feature_importances_
print(importances)

# визуализация решающего дерева
from sklearn import tree
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file='dot_data.dot')


# предсказание по новому объекту
print(clf.predict([[3, 20, 30, 0]]))
print(clf.predict_proba([[3, 20, 30, 0]]))
