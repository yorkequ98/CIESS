import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Configurations as config


class NGCF(nn.Module):
    def __init__(self, dataset, user_sizes, item_sizes):
        print('Initialising NGCF...')
        super(NGCF, self).__init__()
        self.n_user = dataset.n_users
        self.n_item = dataset.n_items
        self.node_dropout = 0.1
        self.drop_flag = True
        self.mess_dropout = [0.1, 0.1, 0.1]

        self.norm_adj = dataset.get_norm_adj_mat()

        self.dataset = dataset

        self.emb_size = config.MAX_EMB_SIZE

        self.layers = [config.MAX_EMB_SIZE] * 3
        self.decay = 1e-3

        self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(config.device)

        self.update_sizes(user_sizes, item_sizes)

    def update_sizes(self, user_sizes, item_sizes):
        ##############
        # end = round(self.emb_size * 0.05 * 2)
        # user_sizes = np.random.randint(1, end, self.n_user, dtype=np.int32)
        # item_sizes = np.random.randint(1, end, self.n_item, dtype=np.int32)
        # print(np.mean(user_sizes), np.mean(item_sizes))
        ##############

        self.user_sizes = user_sizes
        self.item_sizes = item_sizes

        user_mask = np.zeros((self.dataset.n_users, self.emb_size), dtype=np.int32)
        for r in range(len(self.user_sizes)):
            user_mask[r][:self.user_sizes[r]] = 1
        self.user_mask = torch.tensor(user_mask, device=config.device)

        item_mask = np.zeros((self.dataset.n_items, self.emb_size), dtype=np.int32)
        for r in range(len(self.item_sizes)):
            item_mask[r][:self.item_sizes[r]] = 1
        self.item_mask = torch.tensor(item_mask, device=config.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        # initializer = nn.init.normal_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        self.weight_dict = weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = np.array([coo.row, coo.col])
        i = torch.LongTensor(i)
        # i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(config.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(config.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        user_embedding, pos_item_embedding, neg_item_embedding = self(users, pos_items, neg_items)

        pos_scores = torch.sum(torch.mul(user_embedding, pos_item_embedding), axis=1)
        neg_scores = torch.sum(torch.mul(user_embedding, neg_item_embedding), axis=1)

        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # mf_loss = torch.mean(nn.LogSigmoid()(neg_scores - pos_scores))

        regularizer = (user_embedding.norm(2).pow(2)
                       + pos_item_embedding.norm(2).pow(2)
                       + neg_item_embedding.norm(2).pow(2))
        reg_loss = self.decay * regularizer / float(len(users))

        total_loss = mf_loss + reg_loss

        return total_loss, mf_loss, reg_loss

    def get_users_rating(self, users, items):
        u_g_embeddings, i_g_embeddings = self.computer()
        u_g_embeddings = u_g_embeddings[users, :]
        i_g_embeddings = i_g_embeddings[items, :]
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def calc_sparsity(self):
        base_u = len(self.user_sizes) * config.MAX_EMB_SIZE
        base_i = len(self.item_sizes) * config.MAX_EMB_SIZE
        users_emb = self.embedding_dict['user_emb'] * self.user_mask
        items_emb = self.embedding_dict['item_emb'] * self.item_mask
        non_zero_u = torch.nonzero(users_emb).size(0)
        non_zero_i = torch.nonzero(items_emb).size(0)

        percentage = (non_zero_u + non_zero_i) / (base_u + base_i)

        return percentage, (non_zero_u + non_zero_i)

    def computer(self):
        A_hat = self.sparse_dropout(
            self.sparse_norm_adj,
            self.node_dropout,
            self.sparse_norm_adj._nnz()
        ) if self.drop_flag else self.sparse_norm_adj

        user_embs = self.embedding_dict['user_emb'] * self.user_mask
        item_embs = self.embedding_dict['item_emb'] * self.item_mask

        ego_embeddings = torch.cat([user_embs, item_embs], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]
        return u_g_embeddings, i_g_embeddings

    def forward(self, users, pos_items, neg_items):
        u_g_embeddings, i_g_embeddings = self.computer()

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
