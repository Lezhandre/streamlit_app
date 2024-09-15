import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Model_Data import load_data, load_model, get_score, get_confusion_matrix


@st.fragment
def model_score():
    col1, col2 = st.columns([1, 2])
    with col1:
        text_input = st.selectbox(label='Enter score name', options=('recall', 'f1', 'precision', 'accuracy'))
    with col2:
        st.markdown(f'### {get_score(y_test, y_pred, text_input)}')


df = load_data()
model, y_test, y_pred = load_model(df)

st.title('Характеристики модели')

with st.container(border=True):
    fig, ax = plt.subplots()
    ax = sns.histplot(y=df.columns.values[1:], weights=model.classifier_.coef_[0], ax=ax)
    ax.set(xlabel='Вес фичи', title='Важность фичей')
    
    st.pyplot(fig)
    st.markdown('Как видно, наиболее важные фичи - это возраст и пол. Наименее - количество ссуд и доход: доход имеет действительно большой разброс, поэтому определение по нему является проблемой (если только не разделить его на категории).')

model_score()

st.markdown('### Матрица несоответствий:')
st.dataframe(pd.DataFrame(data=get_confusion_matrix(y_test, y_pred), index=['Predicted Negative', 'Predicted Positive'], columns=['True Negative', 'True Positive']))
