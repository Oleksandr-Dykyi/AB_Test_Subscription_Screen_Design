#!/usr/bin/env python
# coding: utf-8

# # A/B Test Subscription Screen Design

# ### Test hypothesis
# Якщо змінити дизайну екрану з пропозицією підписки на тиждень, де користувачам пропонується знижка 50%, то збільшиться конверсія придбання підписки на 10%.

# ### Test description
# До екрану з пропозицією підписки на тиждень додати надпис "знижка 50%".

# ### Confidence and potential risks
# ##### Confidence: 80%
# Тому що показується прибавлива знижка.
# 
# ##### Potential impact on key metrics: High
# Тому що прибавливі знижки спонукають до підписки.
# 
# ##### Risks: Low
# Тому що фактично ціна не піднялась, а теоретично діє знижка.

# ### Affected metrics
# ##### Primary metrics:
# 1. Конверсія з інстала в придбання тижневої підписки: 6,1%.
# 
# ##### Secondary metrics:
# 1. Дохід на користувача.
# 2. Час, проведений на екрані з пропозицією підписки.
# 
# Для зручності в цьому тесті припускатимемо, що додаткові метрики не зазнали значних змін, тому аналізувати результати тесту будемо, спираючись на конверсію з інстала в платіж.

# ### Statistical significance
# Для розрахунку розміру вибірки ми скористаємося такими параметрами:
# 
# ![image-4.png](attachment:image-4.png)
# 
# За допомогою [калькулятора](https://cxl.com/ab-test-calculator/) для розрахунку розміру вибірки для A/B тестів визначили, що нам потрібно приблизно 21 284 користувачі для виявлення мінімальної зміни в конверсії.

# ### Audience and Duration
# Учасники тесту будуть нові користувачі застосунку, які досягли екрану з пропозицією підписки після онбордингу, двох груп:
# 1) Група A. Користувачі, які встановили застосунок в період, коли був запущений тест (цим користувачам ми пропонуємо купити підписку за 4.99, щоб отримати доступ до преміум-функцій застосунку).
# 2) Група B. Користувачі, що встановили версію застосунку, в якій реалізовано A/B тест (пропонуємо ту ж підписку, але з припискою про те, що це пропозиція зі знижкою 50%).
# 
# Користувач повинен бути новачком і перейти на цей екран. Не варто включати в тест користувачів, які вже мають підписку або відмовились від неї раніше.
# 
# Для розрахунку тривалості тесту ми скористаємося такими параметрами:
# 
# ![image.png](attachment:image.png)
# 
# Враховуючи, що кожного дня застосунок встановлюють близько 2 тис. користувачів, за допомогою [калькулятора](https://cxl.com/ab-test-calculator/) визначили, що проводимо тест 22 дні, розбивка становитиме 50:50.

# ### Potential outcomes
# Якщо конверсія до підписки в групі обробки збільшується на рівні значущості 5% і з потужністю тесту 80% та інші метрики не мають статистично негативного впливу, ми розглянемо можливість впровадження нового дизайну.
# 
# Якщо конверсія знижується зі статистичною значущістю в групі обробки, або інші важливі метрики негативно впливають, ми не впроваджуватимемо новий дизайн.
# 
# Якщо немає статистично значущих результатів тесту, ми оцінимо результати та розглянемо можливі покращення альтернативного дизайну перед повторним тестуванням. Можна спробувати інший варіант, наприклад, змінити ціну, знижку або вміст пропозиції. У разі успіху також можливо розглянути подальше вдосконалення та оптимізацію нового дизайну для максимізації конверсії та доходу.

# ### Conclusions

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


test_data = pd.read_csv('ab_test_data.csv')


# In[3]:


test_data['date'] = pd.to_datetime(test_data['timestamp']).dt.date
test_data.head()


# In[4]:


test_data.groupby('test_group').describe()


# In[5]:


print("Count of conversions in groups A and B:")
test_data.groupby('test_group')['conversion'].sum()


# In[6]:


data_start = test_data['date'].min()
data_end = test_data['date'].max()
duration = (data_end - data_start).days
print(f"Test start: {data_start}, Test end: {data_end},  Duration: {duration} days")


# In[7]:


alpha = 0.01

statistic, pvalue = stats.ttest_ind(test_data[test_data['test_group'] == 'a']['conversion'],
                                    test_data[test_data['test_group'] == 'b']['conversion'], 
                                    alternative='less')

print(f'T-test: t-statistic: {round(statistic, 2)}, p-value: {round(pvalue, 2)}')
if pvalue < alpha:
    print('The difference is statistically significant, Null Hypothesis is rejected.')
else:
    print('The difference is insignificant, Null Hypothesis cannot rejected.')


# In[8]:


alpha = 0.01

observed = pd.crosstab(test_data['test_group'].values, test_data['conversion'].values)

statistic, pvalue, dof, expected_values = stats.chi2_contingency(observed)

print(f'Chi-squared: t-statistic: {round(statistic, 2)}, p-value: {round(pvalue, 2)}')
if pvalue < alpha:
    print('The difference is statistically significant, Null Hypothesis is rejected.')
else:
    print('The difference is insignificant, Null Hypothesis cannot rejected.')


# In[9]:


def statistic(x, y):
    return stats.ttest_ind(x, y).statistic

alpha = 0.05   

x = test_data[test_data['test_group'] == 'a']['conversion']
y = test_data[test_data['test_group'] == 'b']['conversion']

results = stats.permutation_test((x, y), statistic, n_resamples=100)

print(f'Permutation test: statistic: {round(results.statistic, 2)}, p-value: {round(results.pvalue, 2)}')
if results.pvalue < alpha:
    print('The difference is statistically significant, Null Hypothesis is rejected.')
else:
    print('The difference is insignificant, Null Hypothesis cannot rejected.')


# #### Conclusions
# Отже, за будь-яким критерієм ми відкидаємо нульову гіпотезу про те, що наша конверсія не залежить від тестової групи, в яку потрапив користувач і можемо приймати альтернативний дизайн. Іншими словами, різниця між групами є статистично значущою і можна впроваджувати новий дизайн (зі знижкою), для зручності припускачи, що додаткові метрики не зазнали значних змін.
# 
# Цільова метрика, конверсія з інстала в підписку збільшилась на 45.9%, з 6.1% до 8.9%.
# 
# Також візуалізуємо результати тесту нижче.

# In[10]:


plt.figure(figsize=(8, 6))
sns.barplot(x=test_data['test_group'], 
            y=test_data['conversion'], 
            errorbar=('ci', 95))
plt.title('A/B Test Results')
plt.xlabel('Group')
plt.ylabel('Mean')
plt.show()


# In[11]:


test_data_sorted = test_data.sort_values(by='timestamp')

cumulative_metric_a = test_data_sorted[test_data_sorted['test_group'] == 'a']['conversion'].expanding().mean().reset_index(drop=True)
cumulative_metric_b = test_data_sorted[test_data_sorted['test_group'] == 'b']['conversion'].expanding().mean().reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_metric_a, label='A')
plt.plot(cumulative_metric_b, label='B')

plt.title('Cumulative Сonversion Rate Comparison')
plt.xlabel('Time')
plt.ylabel('Cumulative Сonversion Rate')

plt.legend()
plt.show()


# In[12]:


test_data_sorted = test_data.sort_values(by='timestamp')

cumulative_metric_a = test_data_sorted[test_data_sorted['test_group'] == 'a'][['date', 'conversion']].set_index('date').expanding().mean()
cumulative_metric_b = test_data_sorted[test_data_sorted['test_group'] == 'b'][['date', 'conversion']].set_index('date').expanding().mean()

plt.figure(figsize=(10, 6))
plt.plot(cumulative_metric_a, label='A')
plt.plot(cumulative_metric_b, label='B')

plt.title('Cumulative Сonversion Rate Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Сonversion Rate')

plt.legend()
plt.show()

