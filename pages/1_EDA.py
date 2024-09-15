import streamlit as st

from Model_Data import load_data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data()
def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


@st.cache_data()
def describe_info(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


df = load_data()

st.title('Разведочный анализ данных')

### 2 Часть

with st.container(border=True):
    fig, ax = plt.subplots()
    ax = sns.scatterplot(df, x='WORK_AGE', y='AGE', ax=ax)
    ax.set(xlabel='Количество лет на работе', ylabel='Возраст', title='Распределённость возраста и количества отработанных лет')
    
    st.pyplot(fig)
    st.markdown('Количество лет проведённых на работе не зависит прямолинейно от возраста. На графике распределённости однако видна "линия" от (10, 50) до (15, 65) - это связано угадыванием количества отработанных лет по некоторым фичам, в том числе и возрасту.')

with st.container(border=True):
    fig, ax = plt.subplots()
    ax = sns.histplot(df, x='DEPENDANTS', y='CHILD_TOTAL', discrete=True, ax=ax)
    ax.set(ylabel='Количество детей', xlabel='Количество иждивенцов', title='Гистограмма по количеству детей и иждивенцев')
    
    st.pyplot(fig)
    st.markdown('Количество детей и иждивенцев коррелирует, при этом иждивенцев чаще меньше чем детей.')

with st.container(border=True):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    
    ax1 = sns.countplot(df, x='AGE', hue='SOCSTATUS_PENS_FL', native_scale=True, ax=ax1)
    ax1.legend(['Не пенсионеры', 'Пенсионеры'])
    ax1.set(ylabel='Количество людей', xlabel='Возраст')

    ax2 = sns.countplot(df, x='AGE', hue='SOCSTATUS_WORK_FL', native_scale=True, ax=ax2)
    ax2.legend(['Безработные', 'Работающие'])
    ax2.set(xlabel='Возраст')
    
    st.markdown('#### Распределение по годам работающих/пенсионеров')
    st.pyplot(fig)
    st.markdown('Как видно по 2-ум графикам выше работающие и не пенсионеры примерно одинаковые между собой. Только в правой части графиков наблюдаются заметные различия: очевидно не все пенсионеры выходя на пенсию прекращают работать.')

with st.container(border=True):
    fig, ax = plt.subplots()
    ax = sns.violinplot(df, x='RELIABILITY', hue='GENDER', split=True, ax=ax)
    h, _ = ax.get_legend_handles_labels()
    ax.legend([h[0], h[-1]], ['Женщины', 'Мужчины'])
    ax.set(xlabel='"Надёжность" выплаты ссуды', ylabel='Распределение людей', title='Распределение по "надёжности" выплаты ссуды')
    
    st.pyplot(fig)
    st.markdown('Видна мультимодальность в точках 0, 0.5 и 1. Больше всего людей не выплативших долги (или долг).')

with st.container(border=True):
    st.dataframe(corr_matrix(df).style.background_gradient(cmap='bone'))
    st.markdown('Все корреляции из матрицы были более-менее объяснены/продемонстрированы на графиках выше. Единственные корреляции, которые были пройдены стороной:\n'
                '1. Количество закрытых и всего кредитов - очевидно все клиенты ~~железного банка являются Ланистерами~~ не хотят оставаться с долгами или просто не могут не выплатить.\n'
                '2. Статусы работающего и пенсионера коррелируют с доходом, сроком работы и количеством иждивенцев: пенсия обычно меньше зарплаты, указать стаж работы видимо обычное дело в бланке вместо времени отработанного на работе, и когда становится мало денег, то количество иждивенцев само сокращается.'
                '3. Количество закрытых ссуд хорошо коррелирует с "надёжностью", т.к. "надёжность" это вероятность случайно выбрать из всех ссуд клиента закрытую.')

with st.container(border=True):
    fig, ax = plt.subplots()
    ax = sns.violinplot(df, x='AGE', hue='TARGET', split=True, ax=ax)
    h, _ = ax.get_legend_handles_labels()
    ax.legend([h[0], h[-1]], ['Не откликнулись', 'Откликнулись'])
    ax.set(xlabel='Возраст', ylabel='Распределение откликов', title='Распределение по откликам относительно возраста')
    
    st.pyplot(fig)
    st.markdown('Мультимодальность, которая видна на графике, позволяет сделать вывод, что в некоторые года человек более охотно даст положительный/отрицательный отклик. При этом видно, что более возрастные клиенты чаще отказываются от предложений.')

with st.container(border=True):
    fig, ax = plt.subplots()
    ax = sns.boxplot(df, x='PERSONAL_INCOME', hue='TARGET', ax=ax)
    ax.legend(['Не откликнулись', 'Откликнулись'])
    ax.set(xlabel='Доход', title='Распределение доходов среди тех, кто откликнулся/не откликнулся')
    
    st.pyplot(fig)
    st.markdown('Из графиков видно, что большая часть откликнувшихся имеют "примерно одинаковый" доход, выше среднего. У тех, кто не откликается больше выбросов и доход в среднем небольшой.')

with st.container(border=True):
    st.dataframe(describe_info(df))
    st.markdown('Видно, что в данных нет пропусков, каких-то бредовых значений и мусора. С данными можно работать дальше.')
