import streamlit as st

import pandas as pd

from Model_Data import load_data, load_model, work_age_model

df = load_data()
model, *_ = load_model(df)

st.title('Предсказание отклика клиента по его данным')


class make_state_save_widget:
    __slots__ = ['function', 'full_state']
    
    def __init__(self, function, full_state: bool = False):
        self.function = function
        self.full_state = full_state
    
    def __call__(self, *args, **kwargs):
        key = kwargs['key']
        
        full_key = f'{key}_saved_state'
        if full_key in st.session_state:
            full_state = st.session_state[full_key]
            curr = full_state.get('value') if self.full_state else full_state
            keys = filter(lambda k: k != 'value', kwargs.keys())
            if key not in st.session_state or self.full_state and any(kwargs[key] != full_state[key] for key in keys):
                st.session_state[key] = curr
        
        ret = self.function(*args, **kwargs)
        
        st.session_state[full_key] = (kwargs | {'value': ret}) if self.full_state else ret
        return ret


number_input = make_state_save_widget(st.number_input)
checkbox = make_state_save_widget(st.checkbox)
selectbox = make_state_save_widget(st.selectbox)
full_state_number_input = make_state_save_widget(st.number_input, full_state=True)

with st.container():
    data = {}
    
    data['AGE'] = number_input('Возраст', min_value=16, max_value=100, step=1, key='age')
    data['SOCSTATUS_WORK_FL'] = checkbox('Работает', key='is_works')
    data['SOCSTATUS_PENS_FL'] = checkbox('Пенсионер', key='is_pens')
    data['GENDER'] = int(selectbox('Пол', options=('Жен', 'Муж'), key='gender') == 'Муж')
    data['CHILD_TOTAL'] = number_input('Количество детей', min_value=0, max_value=20, step=1, key='child_total')
    data['DEPENDANTS'] = number_input('Количество иждивенцев', min_value=0, max_value=20, step=1, key='dependants')
    data['PERSONAL_INCOME'] = number_input('Доход, &#8381;', min_value=1_000, max_value=250_000, step=100, key='personal_income')
    data['LOAN_NUM_TOTAL'] = number_input('Количество ссуд', min_value=0, max_value=20, step=1, key='total_loans')
    data['LOAN_NUM_CLOSED'] = full_state_number_input('Количество погашенных ссуд', min_value=0, max_value=st.session_state['total_loans'], step=1, key='closed_loans')
    data['WORK_AGE'] = full_state_number_input('Количество отработанных лет', min_value=0.0, max_value=st.session_state['age'] - 16.0, value=None, step=1.0, key='work_age')
    
    if st.button(label='Предсказать отклик на маркетинговую кампанию'):
        df = pd.Series(data).to_frame().transpose()
        if data['WORK_AGE'] is None:
            df['WORK_AGE'] = work_age_model().predict(df[['AGE', 'PERSONAL_INCOME', 'GENDER']])
        df['RELIABILITY'] = df['LOAN_NUM_CLOSED'] / df['LOAN_NUM_TOTAL'] if df['LOAN_NUM_TOTAL'].item() else [1.0]
        
        df = model.transform(df)
        positive = model.predict(df)
        do = 'откликнется' if positive else 'не откликнется'
        st.text(f'Клиент {do} с вероятностью {model.predict_proba(df)[0, positive].item()}')
