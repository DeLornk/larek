import pickle
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from tools.tools_recommendation import train_valid_test_split, generate_lightfm_recs_mapper
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class recommender_class:
    def __init__(self, clients_all, products_all, purchases_all):
        """
        Инициализируем класс рекомендаций, загружая предобученные модели и данные
        """

        # Загружаем модели
        with open('./optimized_models/ctb_model.pickle', 'rb') as fle:
            self.ctb_model = pickle.load(fle)
        with open('./optimized_models/lfm_standart_10epoch.pickle', 'rb') as fle:
            self.lfm_model = pickle.load(fle)
        
        # Загружаем данные
        clients_all = pd.read_csv('./data/clients_all.csv', index_col=0)
        clients_all['first_issue_date'] = pd.to_datetime(clients_all['first_issue_date'])
        clients_all['first_redeem_date'] = pd.to_datetime(clients_all['first_redeem_date'])

        products_all = pd.read_csv('./data/products_all.csv', index_col=0)

        purchases_all = pd.read_csv('./data/purchases_all.csv', index_col=0)
        purchases_all['transaction_datetime'] = pd.to_datetime(purchases_all['transaction_datetime'])
        purchases_all = purchases_all[~purchases_all['product_id'].isna()]

        self.clients_all = clients_all
        self.products_all = products_all
        self.purchases_all = purchases_all

    def get_lfm_mapping(self):

        self.lfm_train, self.lfm_pred, self.test = train_valid_test_split(self.purchases_all)
        self.test = self.test[self.test['client_id'].isin(self.lfm_train['client_id'].unique())]

        dataset = Dataset()
        dataset.fit(self.lfm_train['client_id'].unique(), self.lfm_train['product_id'].unique())

        # matrix for training
        interactions_matrix, weights_matrix = dataset.build_interactions(
            zip(*self.lfm_train[['client_id', 'product_id', 'product_quantity']].values.T)
        )

        # weights_matrix_csr = weights_matrix.tocsr()

        # user / item mappings
        lightfm_mapping = dataset.mapping()
        lightfm_mapping = {
            'users_mapping': lightfm_mapping[0],
            'items_mapping': lightfm_mapping[2],
        }

        lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
        lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

        self.lightfm_mapping = lightfm_mapping

    def get_candidates_lfm(self):

        self.get_lfm_mapping()

        # ОТБИРАЕМ КАНДИДАТОВ С ПОМОЩЬЮ LIGHTFM

        lfm_prediction = pd.DataFrame({
            'client_id': self.test['client_id'].unique()
        })

        top_N = 30

        all_cols = list(self.lightfm_mapping['items_mapping'].values())

        lfm_prediction = pd.DataFrame({
            'client_id': self.test['client_id'].unique()
        })

        known_items = self.lfm_train.groupby('client_id')['product_id'].apply(list).to_dict()

        mapper = generate_lightfm_recs_mapper(
            self.lfm_model, 
            item_ids=all_cols, 
            known_items=known_items,
            N=top_N,
            user_features=None, 
            item_features=None, 
            user_mapping=self.lightfm_mapping['users_mapping'],
            item_inv_mapping=self.lightfm_mapping['items_inv_mapping']
        )
        
        lfm_prediction['product_id'] = lfm_prediction['client_id'].map(mapper)
        lfm_prediction = lfm_prediction.explode('product_id').reset_index(drop=True)
        lfm_prediction['rank'] = lfm_prediction.groupby('client_id').cumcount() + 1 

        self.lfm_prediction = lfm_prediction

        return lfm_prediction

    def get_hybrid_pred(self):

        if lfm_prediction:
            lfm_prediction = lfm_prediction
        else:
            lfm_prediction = self.get_candidates_lfm()
        
        # ПРОИЗВОДИМ ПЕРЕРАНЖИРОВАНИЕ С ПОМОЩЬЮ CatBoost
        lfm_ctb_prediction = lfm_prediction.copy()

        # формируем позитивные взаимодействия
        
        print('Формируем позитивные и негативные взаимодействия...')
        pos = lfm_ctb_prediction.merge(self.lfm_pred,
                                on=['client_id', 'product_id'],
                                how='inner')

        pos['target'] = 1

        # формируем негативные примеры
        neg = lfm_ctb_prediction.set_index(['client_id', 'product_id'])\
                .join(self.lfm_pred.set_index(['client_id', 'product_id']))

        neg = neg[neg['product_quantity'].isnull()].reset_index()

        neg['target'] = 0

        # ------------------------

        user_col = ['client_id', 'age', 'gender', 'first_issue_date', 'first_redeem_date']
        item_col = ['product_id', 'price_per_one', 'brand_id', 'netto', 'is_own_trademark', 'vendor_id']
        
        select_col = ['client_id', 'product_id', 'rank', 'target']

        ctb_train_users, ctb_test_users = train_test_split(self.lfm_pred['client_id'].unique(),
                                                          random_state=1,
                                                          test_size=0.2)

        ctb_train_users, ctb_eval_users = train_test_split(ctb_train_users,
                                                          random_state=1,
                                                          test_size=0.1)

        # формируем train
        ctb_train = shuffle(
            pd.concat([
                pos[pos['client_id'].isin(ctb_train_users)],
                neg[neg['client_id'].isin(ctb_train_users)]
        ])[select_col]
        )

        # формируем test
        ctb_test = shuffle(
            pd.concat([
                pos[pos['client_id'].isin(ctb_test_users)],
                neg[neg['client_id'].isin(ctb_test_users)]
        ])[select_col]
        )

        # for early stopping
        ctb_eval = shuffle(
            pd.concat([
                pos[pos['client_id'].isin(ctb_eval_users)],
                neg[neg['client_id'].isin(ctb_eval_users)]
        ])[select_col]
        )
        
        train_feat = ctb_train.merge(self.clients_all[user_col],
                           on=['client_id'],
                           how='left')\
                        .merge(self.products_all[item_col],
                                   on=['product_id'],
                                   how='left')

        eval_feat = ctb_eval.merge(self.clients_all[user_col],
                                   on=['client_id'],
                                   how='left')\
                                .merge(self.products_all[item_col],
                                           on=['product_id'],
                                           how='left')

        drop_col = ['client_id', 'product_id']
        target_col = ['target']
        cat_col = ['age', 'gender', 'brand_id', 'is_own_trademark', 'vendor_id']

        X_train, y_train = train_feat.drop(drop_col + target_col, axis=1), train_feat[target_col]
        X_val, y_val = eval_feat.drop(drop_col + target_col, axis=1), eval_feat[target_col]

        # фичи для теста
        score_feat = lfm_ctb_prediction.merge(self.clients_all[user_col],
                                        on=['client_id'],
                                        how='left')\
                                        .merge(self.products_all[item_col],
                                            on=['product_id'],
                                            how='left')

        # fillna for catboost with the most frequent value 
        score_feat = score_feat.fillna(X_train.mode().iloc[0])
        score_feat['brand_id'] = score_feat['brand_id'].astype(np.int)
        score_feat['vendor_id'] = score_feat['vendor_id'].astype(np.int)  
        
        # catboost predict_proba
        ctb_prediction = self.ctb_model.predict_proba(score_feat.drop(drop_col, axis=1, errors='ignore'))

        lfm_ctb_prediction['ctb_pred'] = ctb_prediction[:, 1]

        # сортируем по скору внутри одного пользователя и проставляем новый ранг
        lfm_ctb_prediction = lfm_ctb_prediction.sort_values(
            by=['client_id', 'ctb_pred'], ascending=[True, False])
        lfm_ctb_prediction['rank_ctb'] = lfm_ctb_prediction.groupby('client_id').cumcount() + 1

        lfm_ctb_prediction.to_csv('./data/results/hybrid_recommendations.csv', index=False)
        return lfm_ctb_prediction


rec_class = recommender_class()

ans = rec_class.get_hybrid_pred()
print(ans)