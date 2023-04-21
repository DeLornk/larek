import json
import pandas as pd
import numpy
from tqdm import tqdm

# функция, чтобы посмотреть размер датасета в памяти
def num_bytes_format(num_bytes, float_prec=4):
    """Useful pandas df memory consumption formatter
    Thanks to
    https://www.kaggle.com/sharthz23/pandas-scipy-for-recsys
    """

    units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Eb']
    for unit in units[:-1]:
        if abs(num_bytes) < 1000:
            return f'{num_bytes:.{float_prec}f} {unit}'
        num_bytes /= 1000
    return f'{num_bytes:.4f} {units[-1]}'

def change_clients(clients):
        num_bytes = clients.memory_usage(deep=True).sum()
        print(f'До: {num_bytes_format(num_bytes)}')

        clients_copy = clients.copy()
        clients_copy['first_issue_date'] = pd.to_datetime(clients_copy['first_issue_date'])
        clients_copy = clients_copy.sort_values(by='first_issue_date').reset_index(drop=True)
        clients_copy = clients_copy.reset_index(drop=False)[['index', 'client_id']]

        # Создаём словари для замены значений
        clients_dict = clients_copy.set_index('client_id').to_dict()['index']

        clients_gender_dict = {'U': 0, 'F': 1, 'M': 2}

        clients['client_id'] = clients['client_id'].map(clients_dict, na_action='ignore')
        clients['gender'] = clients['gender'].map(clients_gender_dict, na_action='ignore')

        num_bytes = clients.memory_usage(deep=True).sum()
        print(f'После: {num_bytes_format(num_bytes)}')

        return clients_dict, clients_gender_dict


def change_products(products):
        num_bytes = products.memory_usage(deep=True).sum()
        print(f'До: {num_bytes_format(num_bytes)}')

        products_copy = products.copy()


        product_id_dict = products_copy.reset_index(drop=False)[['index', 'product_id']]\
                                       .set_index('product_id').to_dict()['index']

        product_brand_dict = {j: i for i, j in enumerate(list(products_copy['brand_id'].dropna().unique()))}

        product_vendor_dict = {j: i for i, j in enumerate(list(products_copy['vendor_id'].dropna().unique()))}


        products['product_id'] = products['product_id'].map(product_id_dict, na_action='ignore')
        products['brand_id'] = products['brand_id'].map(product_brand_dict, na_action='ignore')
        products['vendor_id'] = products['vendor_id'].map(product_vendor_dict, na_action='ignore')

        num_bytes = products.memory_usage(deep=True).sum()
        print(f'После: {num_bytes_format(num_bytes)}')

        return product_id_dict, product_brand_dict, product_vendor_dict


def change_purchases(transaction_dict, store_dict, clients_dict, product_id_dict,
                     purchases_path='./retail_hero_data/purchases.csv', 
                     nrows=16_000_000, count_iter=3,
                     colums=['client_id', 'transaction_id', 'transaction_datetime',
                             'purchase_sum', 'store_id', 'product_id', 
                             'product_quantity', 'trn_sum_from_iss']):
        """
        Читаем по частям большой датасет, чтобы найти популярность магазинов
        
        - `purchases_path` - путь к датасету
        - `nrows` - количество строк, которые читаем за одну итерацию
        - `count_iter=9` - количество итераций. Указано 9, т.к. у нас 45_786_569 записей
        - `colums` - колонки, которые используем
        """
        for k in tqdm(range(count_iter)):
                purchases_k = pd.read_csv(purchases_path, skiprows=lambda x: x > 0 and x < k * nrows, 
                                             nrows=nrows, usecols=colums)
                
                num_bytes = purchases_k.memory_usage(deep=True).sum()
                print(f'До: {num_bytes_format(num_bytes)}')

                purchases_k['client_id'] = purchases_k['client_id'].map(clients_dict, na_action='ignore')
                purchases_k['transaction_id'] = purchases_k['transaction_id'].map(transaction_dict, na_action='ignore')
                purchases_k['store_id'] = purchases_k['store_id'].map(store_dict, na_action='ignore')
                purchases_k['product_id'] = purchases_k['product_id'].map(product_id_dict, na_action='ignore')
                purchases_k['transaction_datetime'] = pd.to_datetime(purchases_k['transaction_datetime'])

                num_bytes = purchases_k.memory_usage(deep=True).sum()
                print(f'После: {num_bytes_format(num_bytes)}')
                        
                if k == 0:
                    result_dataframe = purchases_k
                else:
                    result_dataframe = pd.concat([result_dataframe, purchases_k])
                
                del purchases_k
        
        return result_dataframe.sort_values(by='transaction_datetime')

def get_transaction_dict(purchases_path='./retail_hero_data/purchases.csv', 
                           nrows=16_000_000, count_iter=3, 
                           colums=['transaction_id']):
        """
        Читаем по частям большой датасет, чтобы найти популярность магазинов
        
        - `purchases_path` - путь к датасету
        - `nrows` - количество строк, которые читаем за одну итерацию
        - `count_iter=9` - количество итераций. Указано 9, т.к. у нас 45_786_569 записей
        - `colums` - колонки, которые используем
        """
        transactions = []
        for k in tqdm(range(count_iter)):
                purchases_k = pd.read_csv(purchases_path, skiprows=lambda x: x > 0 and x < k * nrows, 
                                             nrows=nrows, usecols=colums)
                transactions += list(purchases_k['transaction_id'].unique())

                del purchases_k
        
        ans = {j: i for i, j in enumerate(transactions)}
        return ans

def get_store_dict(purchases_path='./retail_hero_data/purchases.csv', 
                           nrows=16_000_000, count_iter=3, 
                           colums=['store_id']):
        """
        Читаем по частям большой датасет, чтобы найти популярность магазинов
        
        - `purchases_path` - путь к датасету
        - `nrows` - количество строк, которые читаем за одну итерацию
        - `count_iter=9` - количество итераций. Указано 9, т.к. у нас 45_786_569 записей
        - `colums` - колонки, которые используем
        """
        stores = []
        for k in tqdm(range(count_iter)):
                purchases_k = pd.read_csv(purchases_path, skiprows=lambda x: x > 0 and x < k * nrows, 
                                             nrows=nrows, usecols=colums)
                stores += list(purchases_k['store_id'].unique())

                del purchases_k
        
        ans = {j: i for i, j in enumerate(stores)}
        return ans

print('0/8 Загружаем products.csv и clients.csv...')
products=pd.read_csv('./retail_hero_data/products.csv')
clients=pd.read_csv('./retail_hero_data/clients.csv')

print('1/8 Меняем типы данных у клиентов...')
clients_dict, clients_gender_dict = change_clients(clients)
print('2/8 Меняем типы данных у продуктов...')
product_id_dict, product_brand_dict, product_vendor_dict = change_products(products)

print('3/8 Сохраняем products_new.csv и clients_new.csv...')
products.to_csv('products_new.csv', index=False)
clients.to_csv('clients_new.csv', index=False)


try:
    print('4/8 Загружаем store_dict.json...')
    with open("store_dict.json", "r") as fp:
        store_dict = json.load(fp)
except:
    print('store_dict.json отсутствует в директории - запускаем создание...')
    store_dict = get_store_dict()
    with open("store_dict.json", "w") as fp:
        json.dump(store_dict, fp)

try:
    print('5/8 Загружаем transaction_dict.json...')
    with open("transaction_dict.json", "r") as fp:
        transaction_dict = json.load(fp)
except:
    print('transaction_dict.json отсутствует в директории - запускаем создание...')
    transaction_dict = get_transaction_dict()
    with open("transaction_dict.json", "w") as fp:
        json.dump(transaction_dict, fp)

print('6/8 Меняем типы данных у покупок...')
purchases = change_purchases(transaction_dict, store_dict, clients_dict, product_id_dict)

# purchases.to_csv('purchases_all.csv', index=False)

try:
    print('7/8 Загружаем transaction_last.json...')
    with open("transaction_last.json", "r") as fp:
        transaction_last = json.load(fp)
except:
    print('transaction_last.json отсутствует в директории - запускаем создание...')
    index_transaction = purchases.reset_index(drop=True).reset_index()[['transaction_id', 'index']].set_index('transaction_id')
    transaction_last = index_transaction.reset_index().set_index('index').drop_duplicates().reset_index(drop=True).reset_index(drop=False).set_index('transaction_id').to_dict()['index']

print('8/8 Сохраняем purchases_all.csv...')
purchases.to_csv('purchases_all.csv', index=False)