from tools import time_series_preparation, train_model_multiple_ts, train_test_split_multiple
from darts.models import AutoARIMA
from darts import TimeSeries
from tqdm import tqdm
import pandas as pd
import numpy as np

df_retail = pd.read_csv('./data/df_retail.csv')

df_retail['product_id'] = df_retail['product_id'].astype(int)
df_retail['product_id'] = df_retail['product_id'].astype(str)
df_retail['transaction_datetime'] = pd.to_datetime(df_retail['transaction_datetime'])

t_s_p = time_series_preparation(df_retail)

total_purchases_by_day, total_orders_by_day, union_ts = t_s_p.create_datesets()

df_preds = None
df_train = None

n = 7

i = 0

for product_id in tqdm(total_purchases_by_day.columns):
    y = TimeSeries.from_series(total_purchases_by_day[product_id])
    
    train = y[:-n]
    
    model_aarima = AutoARIMA()
    model_aarima.fit(train)

    # представляем TimeSeries в виде DataFrame с дополнительной колонкой product_id
    prediction_aarima = model_aarima.predict(n).pd_series().astype(np.int64).to_frame(name='y')
    prediction_aarima['product_id'] = int(product_id)

    train_to_concat = train.pd_series().astype(np.int64).to_frame(name='y')
    train_to_concat['product_id'] = int(product_id)

    # склеиваем с общим датафреймом предсказания и исторические данные
    if i == 0:
        df_preds = prediction_aarima
        df_train = train_to_concat
    else:
        df_preds = pd.concat([df_preds, prediction_aarima])
        df_train = pd.concat([df_train, train_to_concat])

    i += 1

# сохраняем предсказание в формате csv
df_preds.to_csv('./data/results/df_timeseries_preds.csv')
df_train.to_csv('./data/results/historycal_data.csv')