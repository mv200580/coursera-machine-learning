# Assignment: Предобработка данных в Pandas
# https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie/programming/u08ys/priedobrabotka-dannykh-v-pandas

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


print(data[:10])
print(data[data.Age == 28])

print('#1')
print(data['Sex'].value_counts())

print('#2')
print(data['Survived'].value_counts() / data['Survived'].count() * 100)

print('#3')
print(data['Pclass'].value_counts() / data['Pclass'].count() * 100)

print('#4')
print(data['Age'].describe())

print('#5')
print(data.corr('pearson'))

print('#6')
#female = data[data.Sex == 'female']
#names = female['Name']
#print(names)
#print(names.str.extract('Miss.\\W+(\\w+)').value_counts() )
#print(data['Name'].str.contains('Mr'))
#print(data[data.Name.str.contains('Mrs')])
misses = data[data.Name.str.contains('Miss')]['Name']
misses_names = misses.str.extract('Miss\.\\W+(\\w+)');
print(misses_names.value_counts())

mrss = data[data.Name.str.contains('Mrs')]['Name']
mrss_names = mrss.str.extract('\((\\w+)')
print(mrss_names.value_counts())

female_names = misses_names.append(mrss_names)
print(female_names.value_counts())

