import torch
import torch.nn as nn
import numpy as np
import model.Configurations as config


class MLP(nn.Module):
    def __init__(self, dataset, user_sizes, item_sizes):
        print('Initialising MLP...')
        super(MLP, self).__init__()
        self.dataset = dataset
        self.num_users = dataset.n_users
        self.num_items = dataset.n_items

        self.max_emb = max(max(user_sizes), max(item_sizes))

        self.update_sizes(user_sizes, item_sizes)

        self.__init_weight()

        self.fc_layers = torch.nn.ModuleList()
        self.fc_layers.append(torch.nn.Linear(2 * self.max_emb,  128))

        self.affine_output = nn.Linear(128, out_features=1)
        self.decay = 0

    def __init_weight(self):
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.max_emb
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.max_emb
        )

    def update_sizes(self, user_sizes, item_sizes):
        self.user_sizes = user_sizes
        self.item_sizes = item_sizes

        user_mask = np.zeros((self.dataset.n_users, self.max_emb), dtype=np.int32)
        for r in range(len(self.user_sizes)):
            size = self.user_sizes[r]
            user_mask[r][:max(size, config.LOWEST_EMB)] = 1
        self.user_mask = torch.tensor(user_mask, device=config.device)

        item_mask = np.zeros((self.dataset.n_items, self.max_emb), dtype=np.int32)
        for r in range(len(self.item_sizes)):
            size = self.item_sizes[r]
            item_mask[r][:max(size, config.LOWEST_EMB)] = 1
        self.item_mask = torch.tensor(item_mask, device=config.device)

    def forward(self, users, items):
        user_embedding = self.embedding_user(users) * self.user_mask[users]
        item_embedding = self.embedding_item(items) * self.item_mask[items]
        vector = torch.cat([user_embedding, item_embedding], dim=-1).float()
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        return self.affine_output(vector)

    def calc_sparsity(self):
        base_u = len(self.user_sizes) * 128
        base_i = len(self.item_sizes) * 128
        users_emb = self.embedding_user.weight * self.user_mask
        items_emb = self.embedding_item.weight * self.item_mask
        non_zero_u = torch.nonzero(users_emb).size(0)
        non_zero_i = torch.nonzero(items_emb).size(0)
        percentage = (non_zero_u + non_zero_i) / (base_u + base_i)
        return percentage, (non_zero_u + non_zero_i)

    def get_users_rating(self, users):
        users_size = len(users)
        all_items = torch.arange(self.dataset.n_items).long()
        items_size = len(all_items)
        all_items = all_items.to(config.device)
        all_items = all_items.repeat(users_size)
        users = users.repeat_interleave(items_size)
        return self(users, all_items)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self(users, pos_items)
        neg_scores = self(users, neg_items)

        mf_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        user_embedding = self.embedding_user(users) * self.user_mask[users]
        pos_item_embedding = self.embedding_item(pos_items) * self.item_mask[pos_items]
        neg_item_embedding = self.embedding_item(neg_items) * self.item_mask[neg_items]
        regularizer = self.decay * (user_embedding.norm(2).pow(2)
                       + pos_item_embedding.norm(2).pow(2)
                       + neg_item_embedding.norm(2).pow(2))
        reg_loss = regularizer.sum() / float(len(users))

        total_loss = mf_loss + reg_loss

        return total_loss, mf_loss, reg_loss

