import pandas as pd
from tqdm import tqdm

def prepare_clients(clients, lower_limit_age=0, upper_limit_age=100):
        """
        """
        clients = clients.copy()

        # {'U': 0, 'F': 1, 'M': 2}
        mean_age_male = round(clients[(clients['gender'] == 2) \
                                        & (clients['age'] >= lower_limit_age) \
                                        & (clients['age'] <= upper_limit_age)]['age'].mean())

        mean_age_female = round(clients[(clients['gender'] == 1) \
                                        & (clients['age'] >= lower_limit_age) \
                                        & (clients['age'] <= upper_limit_age)]['age'].mean())

        mean_age_unknown = round(clients[(clients['gender'] == 0) \
                                        & (clients['age'] >= lower_limit_age) \
                                        & (clients['age'] <= upper_limit_age)]['age'].mean())
        
        anomal_age_male = clients[(clients['gender'] == 2) \
                                    & ((clients['age'] < 0) | (clients['age'] > 100))].index

        anomal_age_female = clients[(clients['gender'] == 1) \
                                    & ((clients['age'] < 0) | (clients['age'] > 100))].index

        anomal_age_unknown = clients[(clients['gender'] == 0) \
                                        & ((clients['age'] < 0) | (clients['age'] > 100))].index

        clients.loc[anomal_age_male, 'age'] = mean_age_male
        clients.loc[anomal_age_female, 'age'] = mean_age_female
        clients.loc[anomal_age_unknown, 'age'] = mean_age_unknown

        return clients

def set_standart_prices(df, delete_zeros_price=True):
        """
        Убираем продукты, которые куплены в количестве ноль. Устанавливаем стандартизированные цены
        * `df` - исходные данные
        * `delete_zeros_price` - удаляем ли продукты с нулевой ценой
        """
        df = df.copy()
        df = df[df['product_quantity'] != 0] 
        df['trn_sum_from_iss'] = df['trn_sum_from_iss'] / df['product_quantity']
        prices = df.groupby('product_id').agg({'trn_sum_from_iss': lambda x: 
                                               x.max() if x.max() == 0 
                                               else x.max() if (x.mean() / x.max()) < 0.7 
                                               else round((x.max() + x.mean()) / 2, 1)})
        prices.rename(columns={'trn_sum_from_iss': 'price_per_one'}, inplace=True)

        df = df.merge(prices.reset_index(), on='product_id', how='left').drop(columns=('trn_sum_from_iss'))

        if delete_zeros_price:
               df = df[df['price_per_one'] != 0].reset_index(drop=True)

        return df

def merge_and_processing(purchases, products, clients, merge=True,
                         netto_limit=1, delete_zeros_price=True, 
                         delete_alcohol=True, alcohol_column_name='is_alcohol'):
        """
        Объединяем данные из таблиц и производим преобразования

        * `purchases` - таблица покупок
        * `products` - таблица с данными о продуктах
        * `clients` - таблица с данными о клиентах
        * `merge` - булево значение: объединяем всё в одну таблицу или нет
        * `netto_limit` - ограничение на вес товара в кг
        * `delete_zeros_price` - удаляем ли позиции с нулевой ценой
        * `delete_alcohol` - удаляем алкогольные позиции
        """

        if merge:
            df = purchases.merge(products, on='product_id', how='left')\
                                                .merge(clients, on='client_id', how='left')
        else:
            df = purchases.merge(products[['product_id', 'netto', 'is_alcohol']], on='product_id', how='left')
        
        
        print('Удаляем алкоголь...')

        # удаляем алкогольные позиции
        if delete_alcohol:
                df = df[df[alcohol_column_name] == 0]

        print('Выполняем ограничения на вес...')
        # выполняем ограничения по весу
        df = df[df['netto'] <= netto_limit]

        print('Устанавливаем стандартные цены...')
        # убираем варьирующиеся цены
        df = set_standart_prices(df)

        return df

def load_and_preparation(purchase_path='df_max_popular_clean.csv', 
                         products_path='./retail_hero_data/products.csv', 
                         clients_path='./retail_hero_data/clients.csv'):
        """
        Загружаем данные о покупках, ассортименте и клиентах. После этого
        производим преобразования над данными и возвращаем кортежем.
        """

        # читаем и обрабатываем клиентов
        clients = prepare_clients(pd.read_csv(clients_path))
        # clients = clients[['client_id', 'age', 'gender']]

        # читаем и обрабатываем список продуктов
        products = pd.read_csv(products_path)
        products = products[['product_id', 'segment_id', 'brand_id', 'vendor_id', 'netto', 'is_own_trademark', 'is_alcohol']]

        # читаем и обрабатываем таблицу покупок
        # purchases = pd.read_csv(purchase_path, index_col=0)
        purchases = pd.read_csv(purchase_path)
        
        # приводим время заказа к datetime и сортируем
        purchases['transaction_datetime'] = pd.to_datetime(purchases['transaction_datetime'])
        # purchases = purchases.sort_values(by='transaction_datetime')

        print('Начали merge_and_processing...')
        purchases = merge_and_processing(purchases, products, clients, merge=False)

        print('Закончили merge_and_processing...')
        # удаляем лишние столбцы

        products = products.merge(purchases[['product_id', 'price_per_one']].drop_duplicates(), how='right', on='product_id')
        products = products[[*set(products).difference(set(['is_alcohol']))]]

        purchases = purchases[[*set(purchases).difference(set(['is_alcohol', 'netto', 'price_per_one']))]]

        clients = clients[clients['client_id'].isin(purchases['client_id'])].reset_index(drop=True)

        return purchases, products, clients


purchases, products, clients = load_and_preparation(purchase_path='purchases_all.csv', 
                                                    products_path='products_new.csv', 
                                                    clients_path='clients_new.csv')


products.to_csv('./prepared_data/products_all.csv')
clients.to_csv('./prepared_data/clients_all.csv')
purchases.to_csv('./prepared_data/purchases_all.csv')