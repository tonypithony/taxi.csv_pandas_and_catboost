# source CAT/bin/activate
# conda activate CAT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, 
							confusion_matrix, 
							classification_report, 
							matthews_corrcoef,
							roc_curve, 
							roc_auc_score,
							auc,
							cohen_kappa_score)
from math import sqrt
from catboost import CatBoostClassifier, Pool

# no_space_csv = 'noSpace-v7.csv'
# space_csv = 'space-v7.csv'

# df = pd.read_csv(space_csv)
# n_balance = df.shape[0] # 3176 - количество строк в space.csv

# df = pd.read_csv(no_space_csv)
# df = df.iloc[0:n_balance, :] # выбираем количество строк = в space.csv для сбалансированности
# df.to_csv('notspace-v7.csv', index=False)

# # merging two csv files 
# dataframe = pd.concat( 
# 	map(pd.read_csv, [space_csv, 'notspace-v7.csv']), ignore_index=True) 

dataframe = pd.read_csv('taxis.csv')

print(dataframe.head(), '\n')
# print(dataframe.shape, '\n')
print(dataframe.info(), '\n')
print(dataframe.describe(), '\n')

#specify that all columns should be shown
pd.set_option('display.max_columns', None)
print(dataframe.describe(include="all"), '\n') 

# for columnname in list(dataframe.columns):
# 	print(dataframe[columnname].describe(), '\n')

print(dataframe['payment'].value_counts())
# dataframe = dataframe.dropna()
# print(dataframe['payment'].value_counts())
# # до dropna()
# payment
# credit card    4577
# cash           1812
# # после dropna()
# payment
# credit card    4546
# cash           1795



corr_matrix = dataframe[['passengers', 'distance', 'fare', 'tip', 'tolls', 'total']]
print(corr_matrix.corr(), '\n')

plt.figure(figsize=(13, 13))
sns.heatmap(corr_matrix.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Taxi Matrix')
plt.savefig('corr_taxi_matrix.png')
# plt.show()


# dataframe['radiusGL'] = dataframe['radiusGL'].fillna(dataframe['radiusGL'].mean()) # замена NaN на среднее по столбцу
# dataframe['limitDown'] = dataframe['limitDown'].fillna(dataframe['limitDown'].min())
# dataframe['limitUp'] = dataframe['limitUp'].fillna(dataframe['limitUp'].max())
# dataframe['long'] = dataframe['long'].fillna(dataframe['long'].mode()[0]) # замена NaN на моду по столбцу
# dataframe['scope'] = dataframe['scope'].fillna(" ") # замена NaN на пустую строку (или любое заданное значение)


print(dataframe.isna().sum(), '\n')
# pickup              0
# dropoff             0
# passengers          0
# distance            0
# fare                0
# tip                 0
# tolls               0
# total               0
# color               0
# payment            44
# pickup_zone        26
# dropoff_zone       45
# pickup_borough     26
# dropoff_borough    45


target = 'payment'

dataframe = dataframe.dropna()
# print(dataframe.shape) # diff = 218721-201671 = 17050


# Create the feature matrix (X) and target vector (y)
X = dataframe.drop(target, axis=1)

y = dataframe[target]
y = y.map({'cash': 1, 'credit card': 0}).astype(int) # для построения ROC-кривой и использования CrossEntropy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# specifying categorical features
categorical_features = ['pickup', 
						'dropoff',
						'color', 
						'pickup_zone', 
						'dropoff_zone', 
						'pickup_borough',
                        'dropoff_borough']
# create and train the CatBoostClassifier
# https://www.geeksforgeeks.org/catboost-tree-parameters/
# model = CatBoostClassifier(iterations=100, depth=8, learning_rate=0.1, cat_features=categorical_features,
#                            loss_function='CrossEntropy', # or 'LogLoss'
#                            custom_metric=['Accuracy', 'AUC'], random_seed=42)
# model.fit(X_train, y_train)


train_data = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_data = Pool(data=X_test, label=y_test, cat_features=categorical_features)
class_counts = np.bincount(y_train)
 
model = CatBoostClassifier(iterations=500,  # Number of boosting iterations
                           learning_rate=0.1,  # Learning rate
                           depth=8,  # Depth of the tree
                           verbose=100,  # Print training progress every 50 iterations
                           early_stopping_rounds=10,  # stops training if no improvement in 10 consequtive rounds
                           loss_function='CrossEntropy',
                           custom_metric=['Accuracy', 'AUC'], random_seed=42)  # used for Multiclass classification tasks
 
# Train the CatBoost model and collect training progress data
model.fit(train_data, eval_set=test_data)

'''
0:      learn: 0.4782965        test: 0.4736305 best: 0.4736305 (0)     total: 162ms    remaining: 1m 21s
Stopped by overfitting detector  (10 iterations wait)

bestTest = 0.09514272028
bestIteration = 68
'''
 
# Extract the loss values from the evals_result_ dictionary
evals_result = model.get_evals_result()
train_loss = evals_result['learn']['CrossEntropy']
test_loss = evals_result['validation']['CrossEntropy']

# Plot the training progress
iterations = np.arange(1, len(train_loss) + 1)
 
plt.figure(figsize=(7, 4))
plt.plot(iterations, train_loss, label='Training Loss', color='blue')
plt.plot(iterations, test_loss, label='Validation Loss', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Loss (CrossEntropy)')
plt.title('CatBoost Taxi Training Progress')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('Training_Progress_Taxi_Matrix.png')


model.save_model('catboost_classification_taxi.model')


model_name = CatBoostClassifier()      # parameters not required.
model_name.load_model('catboost_classification_taxi.model')


y_pred = model_name.predict(X_test) # predicting accuracy
# print(y_pred)
X_test['predicted'] = y_pred
print(X_test.head(11))

# saving the dataframe
X_test.to_csv('taxi-predicted.csv')
'''
                        dropoff_zone pickup_borough dropoff_borough  predicted
742                      Murray Hill      Manhattan       Manhattan          1 cash
4824                 Lenox Hill East      Manhattan       Manhattan          1
3108                  Midtown Center      Manhattan       Manhattan          0 credit card
4985             Little Italy/NoLiTa      Manhattan       Manhattan          0
219                    Alphabet City      Manhattan       Manhattan          0
4154    Penn Station/Madison Sq West      Manhattan       Manhattan          0
2280                  Yorkville East      Manhattan       Manhattan          0
5456                     Marble Hill         Queens       Manhattan          0
1684  Long Island City/Hunters Point      Manhattan          Queens          0
4889                Manhattan Valley      Manhattan       Manhattan          0
5395                        Union Sq      Manhattan       Manhattan          0
'''


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") # Accuracy: 0.99

Kappa = cohen_kappa_score(y_test, y_pred)
'''
Kappa - это фактический показатель соответствия между фактическими метками 
предсказания и фактическими метками в "получено". Но здесь также упоминается, 
что не следует забывать о другом вероятном результате – точном предсказании 
благодаря чистой случайности. Это также означало, что чем выше или чем ближе 
значение Kappa к 1, тем лучше соответствие между прогнозируемыми значениями 
и фактическими метками.
'''
print(f'Kappa = {Kappa:.2f}', '\n') # 0.93 = 0.9262691828096996

# Plot the confusion matrix as a heatmap
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
# [[356   9]		True Positive (TP)	False Negative (FN)
#  [ 31 873]]		False Positive (FP)	True Negative (TN)

# Коэффициент корреляции Мэтьюса (MCC) — это показатель, который мы можем использовать 
# для оценки эффективности модели классификации. 
# Он рассчитывается как: MCC = (TP*TN – FP*FN) / √ (TP+FP) (TP+FN) (TN+FP) (TN+FN) 
# TP: количество истинных положительных результатов 
# TN: количество истинных отрицательных результатов 
# FP: Количество ложных срабатываний 
# FN: количество ложноотрицательных результатов.

'''
TP = 356, FN = 9, FP = 31, TN = 873

True Positive, True Negative - истинное утверждение и отрицание соответственно; 
False Nagative - если факт отрицается, а на самом деле есть; 
False Positive - факт утверждается, на самом деле ничего не произошло.
'''

TP, FN, FP, TN = confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]
try:
	MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	print(f'TP = {TP}, FN = {FN}, FP = {FP}, TN = {TN}')
	print(f"Коэффициент корреляции Мэтьюса (MCC) = {MCC:.2f}") # = 0.93
except ZeroDivisionError:
	print("MCC = NaN")

print(f"Коэффициент корреляции Мэтьюса (sklearn.metrics) = {matthews_corrcoef(y_test, y_pred):.2f}") # 0.9727766584990545


plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
			xticklabels=['credit card', 'cash'], 
			yticklabels=['credit card', 'cash'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for taxi')
# plt.show()
plt.savefig('Confusion_Taxi_Matrix.png')



importances = model_name.get_feature_importance()
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
 
plt.figure(figsize=(15, 12))
plt.bar(range(len(feature_names)), importances[sorted_indices])
plt.xticks(range(len(feature_names)), feature_names[sorted_indices], rotation=90)
plt.title("Feature Importance for taxi")
# plt.show()
plt.grid()
plt.savefig('Feature_taxi.png')


print("Classification Report for taxi:")
print(classification_report(y_test, y_pred))
# https://habr.com/ru/articles/821547/#classification_metrics
# Classification Report for taxi:
#               precision    recall  f1-score   support

#         cash       0.92      0.98      0.95       365
#  credit card       0.99      0.97      0.98       904

#     accuracy                           0.97      1269
#    macro avg       0.95      0.97      0.96      1269
# weighted avg       0.97      0.97      0.97      1269

'''
0. Precision :
Характеризует долю правильно предсказанных положительных классов среди 
всех образцов, которые модель спрогнозировала как положительный класс.

1. Точность : процент правильных положительных прогнозов по отношению 
к общему количеству положительных прогнозов.

2. Отзыв recall : процент правильных положительных прогнозов по отношению 
к общему количеству фактических положительных результатов.

3. Оценка F1 : средневзвешенное гармоническое значение точности и полноты. 
Чем ближе к 1, тем лучше модель.

Оценка F1: 2 * (Точность * Отзыв) / (Точность + Отзыв)

4. Поддержка support: эти значения просто говорят нам, сколько "игроков" принадлежало 
к каждому классу в тестовом наборе данных.

5. Макро-усреднение (macro-averaging) представляет собой среднее арифметическое 
подсчитанной метрики для каждого класса и используется при дисбалансе классов, 
когда важен каждый класс. В таком случае все классы учитываются равномерно 
независимо от их размера.

6. Взвешенное усреднение (weighted averaging) рассчитывается как взвешенное среднее 
и также применяется в случае дисбаланса классов, но только когда важность класса 
учитывается в зависимости от количества объектов с таким классом, то есть когда 
важны наибольшие классы. При таком подходе важность каждого класса учитывается с 
присвоением им весов. Вес класса w_k может устанавливаться по-разному, например, 
как доля примеров этого класса в обучающей выборке.
'''

# Plot ROC curves for each class
class_labels = np.unique(y) # Get unique class labels
plt.figure(figsize=(8, 6))
all_fpr, all_tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
plt.plot(all_fpr, all_tpr, 'orange')

plt.xlabel(r'False Positive Rate ($FPR = \frac{FP}{FP + TN}$)')
'''
FPR представляет собой долю случаев, когда модель предсказала 
положительный исход, но фактический исход был отрицательным.
False positive rate является важным показателем при разработке 
и оценке моделей машинного обучения, особенно в ситуациях, где 
последствия ложного положительного предсказания могут быть серьёзными.

На высокую false positive rate в моделях машинного обучения могут 
влиять следующие факторы:
* Качество и баланс обучающих данных.
* Тип используемой модели.

Для снижения false positive rate важно тщательно выбирать и предварительно 
обрабатывать обучающие данные, выбирать подходящую модель для конкретной 
задачи и изменять порог модели для прогнозирования благоприятного исхода.
'''
plt.ylabel(r'True Positive Rate ($TPR = \frac{TP}{TP + FN}$)')
'''
TPR представляет собой пропорцию реальных положительных случаев, 
которые были правильно идентифицированы или классифицированы 
моделью как положительные. TPR также известен как чувствительность, 
отзывчивость или частота попадания.
'''

roc_auc = auc(all_fpr, all_tpr)
plt.title(f'ROC curves for taxi CatBoost-classification (area = {roc_auc:.2f})')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1.05)
# plt.show()
plt.savefig('ROC-taxi.png')
# https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
print(f'ROC-AUC area = {roc_auc:.2f}') # = 0.99

'''
В идеальном случае ROC-кривая будет стремиться в верхний левый угол (TPR=1 и FPR=0), 
а площадь под ней (AUC) будет равна 1. 
При значении площади 0.5 качество прогнозов модели будет сопоставимо случайному 
угадыванию, ну а если это значение меньше 0.5, то, модель лучше предсказывает 
результаты, противоположные истинным — в таком случае нужно просто поменять 
целевые метки местами для получения площади больше 0.5.
'''