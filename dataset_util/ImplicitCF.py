import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from model.dataset_util.python_splitters import python_stratified_split
from collections import Counter
import torch
import pickle
import model.Configurations as config


def construct_dict(filename, user_set, item_set):
    dicts = []
    with open(filename) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                for iid in items:
                    dicts.append({'userID': uid, 'itemID': iid, 'rating': 1})
                    item_set.add(iid)
                user_set.add(uid)
    return pd.DataFrame.from_records(dicts), user_set, item_set


class ImplicitCF(object):
    """Data processing class for GCN models which use implicit feedback.

    Initialize train and test set, create normalized adjacency matrix and sample data for training epochs.

    """
    def __init__(
        self,
        base_model,
        dataset,
        adj_dir='tmp/'
    ):
        """Constructor

        Args:
            adj_dir (str): Directory to save / load adjacency matrices. If it is None, adjacency
                matrices will be created and will not be saved.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.

        """
        self.user_idx = None
        self.item_idx = None
        self.adj_dir = adj_dir
        self.col_user = 'userID'
        self.col_item = 'itemID'
        self.col_rating = 'rating'
        self.base_model = base_model
        self.dataset_type = dataset

        if dataset == 'ml-1m':
            filename = 'C:/Users/s4463160/Documents/data/ml-1m/ratings.dat'
            header = ['userID', 'itemID', 'rating', 'timestamp']
            dtypes = {h: np.int32 for h in header}
            df = pd.read_csv(
                filename, sep='::', header=0, names=header, engine='python', dtype=dtypes
            )

        elif dataset == 'yelp':
            train_file = '../data/yelp/train.txt'
            test_file = '../data/yelp/test.txt'

            train, user_set, item_set = construct_dict(train_file, set(), set())
            test, user_set, item_set = construct_dict(test_file, user_set, item_set)
            df = pd.concat([train, test])
        else:
            raise ValueError('Invalid dataset!')

        train, validation, test = python_stratified_split(df, ratio=[0.5, 0.25, 0.25])
        self.train, self.validation, self.test = self._data_processing(train, test, validation=validation)
        assert min(self.train['userID'].values) == 0 and min(self.train['itemID'].values) == 0

        self._init_train_data()

        self.positive_validation_pairs = set(
            (pair[0], pair[1]) for pair in self.validation[['userID', 'itemID']].values
        )

        self.positive_test_pairs = set(
            (pair[0], pair[1]) for pair in self.test[['userID', 'itemID']].values
        )

        self.train_size = len(self.train)
        self.validation_size = len(self.validation)
        self.test_size = len(self.test)

        self.user_freq = Counter(self.train['userID'])
        self.item_freq = Counter(self.train['itemID'])

        self.user_vocab = list(set(self.user_idx['userID_idx'].values))
        self.item_vocab = list(set(self.item_idx['itemID_idx'].values))
        if self.dataset_type == 'yelp':
            try:
                with open('y_true_{}_{}.pkl'.format(dataset, config.SEED), 'rb') as f:
                    self.y_true = pickle.load(f)
            except:
                with open('y_true_{}_{}.pkl'.format(dataset, config.SEED), 'wb') as f:
                    self.y_true = self.get_y_true(self.positive_validation_pairs)
                    pickle.dump(self.y_true, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.y_true = self.get_y_true(self.positive_validation_pairs)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _data_processing(self, train, test, validation=None):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            validation (pandas.DataFrame): Validation data
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                test can be None, if so, we only process the training data.

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.

        """
        if validation is None:
            df = pd.concat([train, test])
        else:
            df = pd.concat([train, validation, test])

        if self.user_idx is None:
            user_idx = df[[self.col_user]].drop_duplicates().reindex()
            user_idx[self.col_user + "_idx"] = np.arange(len(user_idx))
            self.n_users = len(user_idx)
            self.user_idx = user_idx

        if self.item_idx is None:
            item_idx = df[[self.col_item]].drop_duplicates()
            item_idx[self.col_item + "_idx"] = np.arange(len(item_idx))
            self.n_items = len(item_idx)
            self.item_idx = item_idx

        if validation is None:
            return self._reindex(train),  self._reindex(test)
        else:
            return self._reindex(train), self._reindex(validation), self._reindex(test)

    def get_freq(self, mode, entity):
        if mode == 'user':
            if entity in self.user_freq:
                return self.user_freq[entity]
            else:
                return 0
        else:
            assert mode == 'item'
            if entity in self.item_freq:
                return self.item_freq[entity]
            else:
                return 0

    # （num_users x num_items)
    def get_y_true(self, positive_pairs):
        y_true = np.zeros((len(self.user_vocab), len(self.item_vocab)), dtype=np.int32)
        X_ranking = [
            np.repeat(self.user_vocab, len(self.item_vocab)),
            np.tile(self.item_vocab, len(self.user_vocab))
        ]
        for user_id, item_id in zip(X_ranking[0], X_ranking[1]):
            if (user_id, item_id) in positive_pairs:
                user_pos = np.asarray(self.user_vocab == user_id).nonzero()
                assert len(user_pos[0]) == 1
                user_pos = user_pos[0][0]
                assert user_pos == user_id

                item_pos = np.asarray(self.item_vocab == item_id).nonzero()
                assert len(item_pos[0]) == 1
                item_pos = item_pos[0][0]
                assert item_pos == item_id

                y_true[user_id][item_id] = 1
        return y_true

    def _reindex(self, df):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating).

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.

        """
        if df is None:
            return None

        df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        df = df[df[self.col_rating] > 0]

        df_reindex = df[
            [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        ]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df_reindex

    def _init_train_data(self):
        """Record items interated with each user in a dataframe self.interact_status, and create adjacency
        matrix self.R.

        """
        self.interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R[self.train[self.col_user], self.train[self.col_item]] = 1.0

    def switch_to_test_mode(self):
        # combine validation and training sets
        self.train = pd.concat([self.train, self.validation])
        self._init_train_data()
        if self.dataset_type == 'yelp':
            try:
                with open('y_true_{}_{}_test.pkl'.format(self.dataset_type, config.SEED), 'rb') as f:
                    self.y_true = pickle.load(f)
            except:
                with open('y_true_{}_{}_test.pkl'.format(self.dataset_type, config.SEED), 'wb') as f:
                    self.y_true = self.get_y_true(self.positive_test_pairs)
                    pickle.dump(self.y_true, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.y_true = self.get_y_true(self.positive_test_pairs)
        self.R[self.train[self.col_user], self.train[self.col_item]] = 1.0
        self.R[self.validation[self.col_user], self.validation[self.col_item]] = 1.0

    def get_norm_adj_mat(self, retrain=False):
        """Load normalized adjacency matrix if it exists, otherwise create (and save) it.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        if retrain:
            print('Creating a new adjacent matrix for retraining...')
            return self.create_norm_adj_mat()
        try:
            if self.adj_dir is None:
                raise FileNotFoundError
            norm_adj_mat = sp.load_npz("tmp/norm_adj_mat_{}_{}_{}.npz".format(self.dataset_type, config.SEED, self.base_model))
            print("Already load norm adj matrix.")

        except FileNotFoundError:
            norm_adj_mat = self.create_norm_adj_mat()
            if self.adj_dir is not None:
                sp.save_npz("tmp/norm_adj_mat{}_{}_{}.npz".format(self.dataset_type, config.SEED, self.base_model), norm_adj_mat)
        return norm_adj_mat

    def create_norm_adj_mat(self):
        """Create normalized adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[: self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, : self.n_users] = R.T
        if self.base_model == 'ngcf':
            adj_mat = adj_mat.todok() + sp.eye(adj_mat.shape[0])
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat).tocoo()
            return norm_adj_mat.tocsr()
        elif self.base_model == 'lightgcn':
            adj_mat = adj_mat.todok()
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_mat.dot(d_mat_inv)
            return norm_adj_mat.tocsr()

    def train_loader(self, batch_size):
        """Sample train data every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size (int): Batch size of users.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray:
            - Sampled users.
            - Sampled positive items.
            - Sampled negative items.
        """
        def sample_neg(x):
            while True:
                neg_id = random.randint(0, self.n_items - 1)
                if neg_id not in x:
                    return neg_id

        indices = range(self.n_users)
        if self.n_users < batch_size:
            users = [random.choice(indices) for _ in range(batch_size)]
        else:
            users = random.sample(indices, batch_size)

        interact = self.interact_status.iloc[users]
        pos_items = interact[self.col_item + "_interacted"].apply(
            lambda x: random.choice(list(x))
        )
        neg_items = interact[self.col_item + "_interacted"].apply(
            lambda x: sample_neg(x)
        )

        return np.array(users), np.array(pos_items), np.array(neg_items)

