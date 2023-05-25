import streamlit as st
import pandas as pd

def get_recommendations(n=10, m=7):
    """
    Берём топ-`n` рекомендации от гибридной модели, выбираем из них позиции, которых не было
    в продаже за прошлые `m` дней
    """
    hybrid_recommendations = pd.read_csv('../data/results/hybrid_recommendations.csv')
    hybrid_recommendations = hybrid_recommendations[(hybrid_recommendations['client_id'].isin(df['client_id'])) &
                                                    (hybrid_recommendations['rank_ctb'] <= n)].reset_index(drop=True)
    hybrid_recommendations['product_id'] = hybrid_recommendations['product_id'].astype(int)

    recommendations = hybrid_recommendations.astype(str).groupby('product_id').agg(lambda x: ','.join(x.unique()))['client_id'].to_frame()
    recommendations['count'] = hybrid_recommendations.astype(str).groupby('product_id')['client_id'].count()
    recommendations.index = recommendations.index.astype(int, copy=False)

    products_last_week = df[df['transaction_datetime'] > (max(df['transaction_datetime']) - pd.Timedelta(m, 'd'))]['product_id']

    recom_id = set(recommendations.index)
    last_id = set(products_last_week)

    return recommendations.loc[list(recom_id.difference(last_id))]


# устанавливаем максимальную ширину вывода на экран
st.set_page_config(layout="wide")

# Создаём заголовки
st.title("Ларёк")
col1, col2 = st.columns([2, 1], gap="large")
col1.subheader("Прогноз")

col3, col4 = st.columns([2, 1], gap="large")
col3.subheader("Общие данные о продажах")
col4.subheader("Рекомендации")

# загружаем данные о продажах
df = pd.read_csv('../data/results/df_retail.csv')[['transaction_datetime', 'client_id', 'product_quantity', 'transaction_id', 'product_id']]
df_timeseries_preds = pd.read_csv('../data/results/df_timeseries_preds.csv', index_col='transaction_datetime')
df_historycal_data = pd.read_csv('../data/results/historycal_data.csv', index_col='transaction_datetime')

df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
df = df.sort_values(by='transaction_datetime')
df.reset_index(drop=True, inplace=True)

lists_of_products = df['product_id'].unique()
lists_of_users = df['client_id'].unique()

# Создаём sidebar
st.sidebar.header("Фильтры")

product_for_prediction = st.sidebar.selectbox("Продукт для прогноза", options=lists_of_products)

products_choosen = st.sidebar.multiselect("Фильтрация продаж по товарам", 
                                          options=lists_of_products)

# Выводим прогноз
if product_for_prediction:
    history = df_historycal_data[df_historycal_data.product_id == product_for_prediction]['y'].to_frame(name='Исторические данные')
    preds = df_timeseries_preds[df_timeseries_preds.product_id == product_for_prediction]['y'].to_frame(name='Прогноз')
    col1.line_chart(history.join(preds[:7], how='outer'))
else:
    col1.warning("Для вывода прогноза выберите товар", icon='🔎')


col2.metric("Спрос на следующей неделе", preds[:7].sum())

# Выводим рекомендации
rec = get_recommendations()
col4.dataframe(rec)

# выведем данные о продажах
if products_choosen:
    col3.dataframe(df[df['product_id'].isin(products_choosen)])
else:
    col3.dataframe(df)