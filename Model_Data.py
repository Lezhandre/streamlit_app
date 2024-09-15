from typing import Self, Tuple
import streamlit as st

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DATAFRAMES_URL = 'https://github.com/aiedu-courses/linear_models/raw/main/datasets/clients/{}'
work_age_model_ = None


def work_age_model(X_train: pd.DataFrame = None, y_train: pd.Series = None) -> LinearRegression:
    global work_age_model_ ## должно увидеть модульную переменную
    if X_train is not None:
        model = LinearRegression(positive=True) ## чтобы не было отрицательных лет
        work_age_model_ = model.fit(X_train, y_train)
    return work_age_model_


@st.cache_data()
def load_data() -> pd.DataFrame:
    ''' load data and preprocessing it '''
    ### 1 Часть
    
    ## Загрузка таблиц и их соединение
    targets_df = pd.read_csv(DATAFRAMES_URL.format('D_target.csv'))
    
    clients_df = pd.read_csv(DATAFRAMES_URL.format('D_clients.csv'))
    last_credit_df = pd.read_csv(DATAFRAMES_URL.format('D_last_credit.csv'))
    clients_df = clients_df.merge(last_credit_df, how='left', left_on='ID', right_on='ID_CLIENT', copy=False)
    targets_df = targets_df.join(clients_df.set_index('ID_CLIENT'), on='ID_CLIENT', how='left')

    loans_df = pd.read_csv(DATAFRAMES_URL.format('D_loan.csv'))
    closed_loans_df = pd.read_csv(DATAFRAMES_URL.format('D_close_loan.csv'))
    loans_info_df = loans_df.merge(closed_loans_df, how='left', on='ID_LOAN')
    clients_loans_info_df = loans_info_df.groupby(by='ID_CLIENT').agg(
        LOAN_NUM_CLOSED=pd.NamedAgg(column='CLOSED_FL', aggfunc='sum'),
        LOAN_NUM_TOTAL=pd.NamedAgg(column='CLOSED_FL', aggfunc='count')
    )
    targets_df = targets_df.join(clients_loans_info_df, on='ID_CLIENT', how='left')
    
    assert (targets_df['LOAN_NUM_CLOSED'] <= targets_df['LOAN_NUM_TOTAL']).all()

    salaries_df = pd.read_csv(DATAFRAMES_URL.format('D_salary.csv'))
    targets_df = targets_df.join(salaries_df.set_index('ID_CLIENT'), on='ID_CLIENT', how='left')

    jobs_df = pd.read_csv(DATAFRAMES_URL.format('D_job.csv'))
    targets_df = targets_df.join(jobs_df.set_index('ID_CLIENT'), on='ID_CLIENT', how='left')

    targets_df = targets_df.drop_duplicates(ignore_index=True)
    
    ## Очистка странного слишком маленького дохода (предполагаем, что люди указавшие маленькие числа имели в виду тысячи)
    targets_df['PERSONAL_INCOME'] = targets_df['PERSONAL_INCOME'].mask(targets_df['PERSONAL_INCOME'] < 100, targets_df['PERSONAL_INCOME'] * 1_000)
    targets_df['PERSONAL_INCOME'] = np.maximum(targets_df['PERSONAL_INCOME'], targets_df['PERSONAL_INCOME'].quantile(0.001))
    
    ## Cколько лет отработал человек
    targets_df['WORK_AGE'] = targets_df['WORK_TIME'] / 12
    
    ## Построим модель для получения WORK_AGE для всех людей
    norm_work_age = (targets_df['WORK_AGE'] < 70) & (targets_df['WORK_AGE'] < targets_df['AGE'] - 16)
    targets_df['WORK_AGE'] = targets_df['WORK_AGE'].where(norm_work_age, np.nan)
    has_work_age = ~targets_df['WORK_AGE'].isna()
    model = work_age_model(targets_df.loc[has_work_age, ['AGE', 'PERSONAL_INCOME', 'GENDER']], targets_df.loc[has_work_age, 'WORK_AGE'])
    targets_df.loc[~has_work_age, 'WORK_AGE'] = model.predict(targets_df.loc[~has_work_age, ['AGE', 'PERSONAL_INCOME', 'GENDER']])
    
    assert (targets_df['WORK_AGE'] < targets_df['AGE']).all()

    ## "Надёжность" : вероятность выбрать наугад погашенную среди всех ссуд клиента
    targets_df['RELIABILITY'] = (targets_df['LOAN_NUM_CLOSED'] / targets_df['LOAN_NUM_TOTAL'])
    no_reliability = targets_df['RELIABILITY'].isna()
    targets_df.loc[no_reliability, 'RELIABILITY'] = targets_df.loc[~no_reliability, 'RELIABILITY'].mean()
    
    assert (~targets_df['RELIABILITY'].isna()).all()

    ## Больше нам ничего не нужно (наверно)
    targets_df = targets_df.drop(['FAMILY_INCOME', 'OWN_AUTO', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'JOB_DIR', 'WORK_TIME', 'EDUCATION', 'MARITAL_STATUS', 'ID', 'ID_CLIENT', 'CREDIT', 'TERM', 'FST_PAYMENT', 'FL_PRESENCE_FL', 'GEN_INDUSTRY', 'GEN_TITLE'], axis=1)

    return targets_df.set_index('AGREEMENT_RK')


class Model:
    ''' model for predicting client feedback '''
    def __init__(self):
        self.classifier_ = LogisticRegressionCV(
            max_iter=500,
            class_weight='balanced', ## иначе все ответы будут 0
            random_state=42,
            solver='newton-cholesky'
        )
        self.transformer_ = ColumnTransformer(
            [('age', QuantileTransformer(n_quantiles=8, random_state=42), ['WORK_AGE', 'AGE']),
            ('std', StandardScaler(), ['PERSONAL_INCOME', 'RELIABILITY']),
            ('pwr', PowerTransformer(), ['CHILD_TOTAL', 'DEPENDANTS', 'LOAN_NUM_CLOSED', 'LOAN_NUM_TOTAL'])],
            remainder='passthrough'
        )
    
    def preprocess(self, df: pd.DataFrame, target: str = 'TARGET') -> tuple:
        ''' runs preprocessing on dataset '''

        ## Разделение данных
        X, y = df, df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X[target])
        X_train, X_test = X_train.drop(target, axis=1), X_test.drop(target, axis=1)
        
        ## Масштабирование
        self.transformer_.fit(X_train)
        
        X_train, X_test = self.transformer_.transform(X_train), self.transformer_.transform(X_test)

        return X_train, X_test, y_train, y_test
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' transform input data '''
        
        return self.transformer_.transform(X)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        ''' fitting model '''
        
        self.classifier_.fit(X_train, y_train)
        return self
            
    def predict(self, X_transformed: np.ndarray):
        ''' predict answers '''
        
        return self.classifier_.predict(X_transformed)
            
    def predict_proba(self, X_transformed: np.ndarray):
        ''' predict probabilities '''
        
        return self.classifier_.predict_proba(X_transformed)


@st.cache_resource()
def load_model(data: pd.DataFrame) -> Tuple[Model, np.ndarray, np.ndarray]:
    model = Model()
    X_train, X_test, y_train, y_test = model.preprocess(data)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, y_test


def get_score(y_test: np.ndarray, y_pred: np.ndarray, metric: str):
    try:
        return globals()[f'{metric}_score'](y_test, y_pred)
    except KeyError:
        return ''


def get_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    return confusion_matrix(y_test, y_pred)
