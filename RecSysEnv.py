import sys
import numpy as np
from model.base_models.LightGCN import LightGCN
from model.base_models.NGCF import NGCF
from model.base_models.MLP import MLP
from model.base_models.NCF import NeuMF
import model.Configurations as config
import torch.optim as optim
import torch


def recall_at_k(zipped, total, k):
    top_k = zipped[:k]
    hits = np.sum([tup[0] for tup in top_k])
    return hits / np.maximum(total, 1)


def ndcg_at_k(zipped, ideal_rank, k):
    top_k = zipped[:k]
    ranked = np.array([tup[0] for tup in top_k])
    positions = np.log(np.arange(len(top_k)) + 2)
    dcg = np.round(ranked) / positions
    idcg = ideal_rank[:k] / positions
    return np.sum(dcg) / np.maximum(np.sum(idcg), 1)


def process_ranking_metrics(recalls_5, recalls_10, recalls_20, ndcgs_5, ndcgs_10, ndcgs_20):
    metrics_arr = np.array([
        recalls_5, recalls_10, recalls_20,
        ndcgs_5, ndcgs_10, ndcgs_20
    ])

    mean_recall_5, mean_recall_10, mean_recall_20, \
        mean_ndcg_5, mean_ndcg_10, mean_ndcg_20 = np.mean(metrics_arr, axis=1)

    mean_metrics_per_entity = np.mean(metrics_arr, axis=0)

    print('recall@5 = {:.4f}, NDCG@5 = {:.4f}'.format(mean_recall_5, mean_ndcg_5))
    print('recall@10 = {:.4f}, NDCG@10 = {:.4f}'.format(mean_recall_10, mean_ndcg_10))
    print('recall@20 = {:.4f}, NDCG@20 = {:.4f}'.format(mean_recall_20, mean_ndcg_20))
    print('AVG = {:.4f}'.format(np.mean(mean_metrics_per_entity)))
    print('-' * 50)

    return mean_metrics_per_entity


class RecSysEnv:
    def __init__(
            self,
            dataset,
            _lambda,
            base_model,
            user_peak_quality=None,
            item_peak_quality=None,
            user_sizes=None,
            item_sizes=None,
            retrain=False
    ):
        self.dataset = dataset

        if user_sizes is None:
            self.user_sizes = np.ones(dataset.n_users, dtype=np.int32) * config.MAX_EMB_SIZE
        else:
            self.user_sizes = user_sizes
        if item_sizes is None:
            self.item_sizes = np.ones(dataset.n_items, dtype=np.int32) * config.MAX_EMB_SIZE
        else:
            self.item_sizes = item_sizes

        wd = 0
        if base_model == 'ngcf':
            self.agent = NGCF(dataset, self.user_sizes, self.item_sizes).to(config.device)
            self.decay_batches = 100
            self.n_batches = 3000
            if self.dataset.dataset_type == 'yelp':
                self.n_batches = 5000
            self.min_lr = 1e-3
            self.max_lr = 0.03
        elif base_model == 'lightgcn':
            self.agent = LightGCN(dataset, self.user_sizes, self.item_sizes, retrain=retrain).to(config.device)
            self.decay_batches = 200
            self.n_batches = 2000
            if self.dataset.dataset_type == 'yelp':
                self.n_batches = 4000
            self.min_lr = 1e-3
            self.max_lr = 0.03
        elif base_model == 'mlp':
            self.agent = MLP(dataset, self.user_sizes, self.item_sizes).to(config.device)
            self.decay_batches = 100
            self.n_batches = 3000
            self.min_lr = 1e-3
            self.max_lr = 0.03
            wd = 1e-5
        elif base_model == 'ncf':
            self.agent = NeuMF(dataset, self.user_sizes, self.item_sizes).to(config.device)
            self.decay_batches = 50  # do not change!
            self.n_batches = 1500
            if self.dataset.dataset_type == 'yelp':
                self.n_batches = 3000
            self.min_lr = 1e-4  # do not change!
            self.max_lr = 0.03
        else:
            raise ValueError('Invalid choice of base model!')
        self.base_model = base_model

        self.user_qualities = np.zeros(len(dataset.user_vocab))
        self.item_qualities = np.zeros(len(dataset.item_vocab))

        if user_peak_quality is None or item_peak_quality is None:
            self.user_peak_qualities = np.zeros(len(dataset.user_vocab))
            self.item_peak_qualities = np.zeros(len(dataset.item_vocab))
        else:
            self.user_peak_qualities = user_peak_quality
            self.item_peak_qualities = item_peak_quality
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.max_lr, weight_decay=wd)
        self._lambda = _lambda

    def compute_metrics_user(self, y_pred, sampled_users, sampled_items):
        # convert matrix to array
        y_pred = np.asarray(y_pred)
        recalls_20, ndcgs_20, recalls_10, \
            ndcgs_10, recalls_5, ndcgs_5 = [], [], [], [], [], []
        for user_id in sampled_users:
            assert self.dataset.user_vocab[user_id] == user_id
            total = np.sum(self.dataset.y_true[user_id][sampled_items])
            user_pos = np.asarray(sampled_users == user_id).nonzero()
            assert len(user_pos[0]) == 1
            user_pos = user_pos[0][0]
            assert self.dataset.y_true[user_id][sampled_items].shape == y_pred[user_pos].shape
            zipped = sorted(
                list(zip(self.dataset.y_true[user_id][sampled_items], y_pred[user_pos])),
                key=lambda tup: tup[1],
                reverse=True
            )
            recalls_20.append(recall_at_k(zipped, total, k=20))
            recalls_10.append(recall_at_k(zipped, total, k=10))
            recalls_5.append(recall_at_k(zipped, total, k=5))

            ideal_rank = np.sort(self.dataset.y_true[user_id][sampled_items])[::-1]

            ndcgs_20.append(ndcg_at_k(zipped, ideal_rank, k=20))
            ndcgs_10.append(ndcg_at_k(zipped, ideal_rank, k=10))
            ndcgs_5.append(ndcg_at_k(zipped, ideal_rank, k=5))
        return process_ranking_metrics(recalls_5, recalls_10, recalls_20, ndcgs_5, ndcgs_10, ndcgs_20)

    def compute_metrics_item(self, y_pred, sampled_items, quality_u):
        y_pred = np.asarray(y_pred)
        quality_i = []
        for item_id in sampled_items:
            item_pos = np.asarray(sampled_items == item_id).nonzero()
            assert len(item_pos[0]) == 1
            item_pos = item_pos[0][0]
            paired_users = np.where(y_pred[:, item_pos] == -np.inf)[0]
            if len(paired_users) == 0:
                quality_i.append(0)
            else:
                quality_i.append(np.mean(quality_u[paired_users]))
        assert len(quality_i) == len(sampled_items)
        return np.array(quality_i)

    def compute_ranking_metrics(self, y_pred, mode, sampled_users=None, sampled_items=None, quality_u=None):
        if mode == 'item':
            assert len(quality_u) == len(sampled_users)
            return self.compute_metrics_item(y_pred, sampled_items, quality_u)
        else:
            return self.compute_metrics_user(y_pred, sampled_users, sampled_items)

    def step(self, actions_u, actions_i, sampled_users, sampled_items):
        self.renew_recommender(actions_u, actions_i)
        sparsity = self.agent.calc_sparsity()

        # adjust the embedding sizes and train the model
        quality_u, quality_i = self.train_n_batches(self.n_batches, sampled_users, sampled_items, verbose=1)

        self.user_qualities[sampled_users] = \
            (quality_u / np.maximum(self.user_peak_qualities[sampled_users], 1e-5)) * 2 - 1
        self.user_qualities[sampled_users] = np.clip(self.user_qualities[sampled_users], -1, 1)
        self.item_qualities[sampled_items] = \
            (quality_i / np.maximum(self.item_peak_qualities[sampled_items], 1e-5)) * 2 - 1
        self.item_qualities[sampled_items] = np.clip(self.item_qualities[sampled_items], -1, 1)

        # update the state
        new_state_u = self.get_state('user')
        reward_u = self.compute_reward(quality_u, actions_u[sampled_users], 'user')

        new_state_i = self.get_state('item')
        reward_i = self.compute_reward(quality_i, actions_i[sampled_items], 'item')

        if self.base_model == 'ncf':
            return new_state_u, new_state_i, \
                   reward_u, reward_i, \
                   quality_u, quality_i, \
                   sparsity
        else:
            return new_state_u, new_state_i, \
                   reward_u, reward_i, \
                   quality_u, quality_i

    def get_state(self, mode):
        if mode == 'user':
            entity_ids = np.array(self.dataset.user_vocab)
            max_freq = max(self.dataset.user_freq.values())
            min_freq = min(self.dataset.user_freq.values())
            entity_quality = self.user_qualities
            entity_sizes = self.agent.user_sizes
        else:
            entity_ids = np.array(self.dataset.item_vocab)
            max_freq = max(self.dataset.item_freq.values())
            min_freq = min(self.dataset.item_freq.values())
            entity_quality = self.item_qualities
            entity_sizes = self.agent.item_sizes

        freq_diff = max_freq - min_freq
        dim_diff = config.MAX_EMB_SIZE - config.MIN_EMB_SIZE

        return np.array(
            [
                [
                    (self.dataset.get_freq(mode, entity_ids[i]) - min_freq) / freq_diff * 2 - 1,
                    ((entity_sizes[entity_ids[i]] - config.MIN_EMB_SIZE) / dim_diff) * 2 - 1,
                    entity_quality[entity_ids[i]]
                ] for i in range(len(entity_ids))
            ], dtype=np.float32)

    def compute_reward(self, quality, actions, mode):
        # actual memory cost
        memory_cost = np.power((actions - config.MIN_EMB_SIZE) / config.MAX_EMB_SIZE, 2)
        reward = quality - self._lambda * memory_cost

        print('{} Q {:.4f} - {} * MC {:.4f} = R {:.4f}'.format(
            mode,
            np.nanmean(quality),
            self._lambda,
            np.mean(memory_cost),
            np.mean(reward))
        )
        # the reward returned here should be maximised
        return reward

    def train_one_batch(self):
        users, pos_items, neg_items = self.dataset.train_loader(config.BATCH_SIZE)
        users = torch.tensor(users).long().to(config.device)
        pos_items = torch.tensor(pos_items).long().to(config.device)
        neg_items = torch.tensor(neg_items).long().to(config.device)

        loss, mf_loss, emb_loss = self.agent.create_bpr_loss(
            users, pos_items, neg_items
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, mf_loss, emb_loss

    def train_n_batches(self, batches, sampled_users, sampled_items, verbose=0):
        for e in range(batches):
            loss, mf_loss, emb_loss = self.train_one_batch()
            if (e + 1) % self.decay_batches == 0:
                self.optimizer.param_groups[0]['lr'] = max(self.min_lr, 0.9 * self.optimizer.param_groups[0]['lr'])

            if (e + 1) % 2000 == 0 and verbose == 1:
                print('Epoch %d: [%.5f = %.5f + %.5f]' % (e, loss, mf_loss, emb_loss))
                #     print("Adjusting LR to ", self.optimizer.param_groups[0]['lr'])
        val = self.eval_rec(sampled_users, sampled_items)
        return val

    def eval_rec(self, sampled_users, sampled_items):
        y_pred = self.get_y_pred(sampled_users, sampled_items)
        user_metrics = self.compute_ranking_metrics(y_pred, 'user', sampled_users, sampled_items)
        item_metrics = self.compute_ranking_metrics(y_pred, 'item', sampled_users, sampled_items,
                                                    quality_u=user_metrics)
        return user_metrics, item_metrics

    def renew_recommender(self, user_sizes, item_sizes):
        wd = 0
        if self.base_model == 'lightgcn':
            self.agent = LightGCN(self.dataset, user_sizes, item_sizes).to(config.device)
        elif self.base_model == 'ngcf':
            self.agent = NGCF(self.dataset, user_sizes, item_sizes).to(config.device)
        elif self.base_model == 'ncf':
            self.agent = NeuMF(self.dataset, user_sizes, item_sizes).to(config.device)
        elif self.base_model == 'mlp':
            self.agent = MLP(self.dataset, user_sizes, item_sizes).to(config.device)
            wd = 1e-5
        else:
            raise ValueError('Invalid option for base model!')
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.max_lr, weight_decay=wd)

    def get_y_pred(self, sampled_users, sampled_items):
        """Score all items for test users.
        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """
        batch_size = 2000
        if self.base_model == 'mlp' or self.base_model == 'ncf':
            batch_size = 64
        with torch.no_grad():
            user_ids = sampled_users
            n_user_batchs = len(user_ids) // batch_size + 1
            test_scores = np.array([])

            for u_batch_id in range(n_user_batchs):
                start = u_batch_id * batch_size
                end = min((u_batch_id + 1) * batch_size, len(user_ids))
                user_batch = user_ids[start: end]
                batch_users_gpu = torch.Tensor(user_batch).long().to(config.device)
                batch_items_gpu = torch.Tensor(sampled_items).long().to(config.device)
                ratings = self.agent.get_users_rating(batch_users_gpu, batch_items_gpu).squeeze()
                if len(test_scores) == 0:
                    test_scores = ratings.cpu().numpy()
                else:
                    test_scores = np.concatenate((test_scores, ratings.cpu().numpy()), axis=0)

            test_scores = np.array(test_scores).reshape((len(sampled_users), len(sampled_items)))
            sampled_R = self.dataset.R.tocsr()[sampled_users][:, sampled_items]
            test_scores += sampled_R * -np.inf
            return test_scores
