import torch
import torch.nn as nn
import numpy as np
import model.Configurations as config


class NeuMF(nn.Module):
    def __init__(self, dataset, user_sizes, item_sizes):
        print('Initialising NeuMF...')
        super(NeuMF, self).__init__()
        self.dataset = dataset
        self.num_users = dataset.n_users
        self.num_items = dataset.n_items

        self.max_emb = max(max(user_sizes), max(item_sizes))
        print('max emb = {:.4f}'.format(self.max_emb))

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(torch.nn.Linear(2 * self.max_emb, self.max_emb))

        self.affine_output = nn.Linear(in_features=self.max_emb * 2, out_features=1)
        self.logistic = nn.Sigmoid()

        self.update_sizes(user_sizes, item_sizes)
        self.__init_weight()

        self.decay = 1e-2

    def update_sizes(self, user_sizes, item_sizes):
        self.user_sizes = user_sizes
        self.item_sizes = item_sizes

        user_mask = np.zeros((self.dataset.n_users, self.max_emb), dtype=np.int32)
        for r in range(len(self.user_sizes)):
            user_mask[r][:self.user_sizes[r]] = 1
        self.user_mask = torch.tensor(user_mask, device=config.device)

        item_mask = np.zeros((self.dataset.n_items, self.max_emb), dtype=np.int32)
        for r in range(len(self.item_sizes)):
            item_mask[r][:self.item_sizes[r]] = 1
        self.item_mask = torch.tensor(item_mask, device=config.device)

    def __init_weight(self):
        self.emb_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.max_emb)
        self.emb_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.max_emb)
        self.emb_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.max_emb)
        self.emb_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.max_emb)

        for emb in [self.emb_user_mlp, self.emb_item_mlp, self.emb_user_mf, self.emb_item_mf]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, users, items):
        user_embedding_mlp = self.emb_user_mlp(users) * self.user_mask[users]
        item_embedding_mlp = self.emb_item_mlp(items) * self.item_mask[items]

        user_embedding_mf = self.emb_user_mf(users)
        item_embedding_mf = self.emb_item_mf(items)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1).float()
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf).float()

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        return logits

    def create_bpr_loss(self, users, pos, neg):
        pos_scores = self(users, pos)
        neg_scores = self(users, neg)
        mf_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        user_embedding_mlp = self.emb_user_mlp(users) * self.user_mask[users]
        pos_embedding_mlp = self.emb_item_mlp(pos) * self.item_mask[pos]
        neg_embedding_mlp = self.emb_item_mlp(neg) * self.item_mask[neg]

        user_embedding_mf = self.emb_user_mf(users)
        pos_embedding_mf = self.emb_item_mf(pos)
        neg_embedding_mf = self.emb_item_mf(neg)

        reg_loss = self.decay * (user_embedding_mlp.norm(2).pow(2) +
                                pos_embedding_mlp.norm(2).pow(2) +
                                neg_embedding_mlp.norm(2).pow(2) +
                                user_embedding_mf.norm(2).pow(2) +
                                pos_embedding_mf.norm(2).pow(2) +
                                neg_embedding_mf.norm(2).pow(2)) / float(len(users))

        total_loss = mf_loss + reg_loss

        return total_loss, mf_loss, reg_loss

    def calc_sparsity(self):
        base_u = len(self.user_sizes) * config.MAX_EMB_SIZE
        base_i = len(self.item_sizes) * config.MAX_EMB_SIZE
        non_zero_u_mlp = torch.nonzero(self.user_mask).size(0)
        non_zero_i_mlp = torch.nonzero(self.item_mask).size(0)

        non_zero_u_mf = len(self.user_sizes) * self.max_emb
        non_zero_i_mf = len(self.item_sizes) * self.max_emb

        percentage = (non_zero_u_mlp + non_zero_i_mlp + non_zero_u_mf + non_zero_i_mf) / (2 * (base_u + base_i))

        return percentage, (non_zero_u_mlp + non_zero_i_mlp + non_zero_u_mf + non_zero_i_mf)

    def get_users_rating(self, users, items):
        users_size = len(users)
        all_items = items.long()
        items_size = len(all_items)
        all_items = all_items.to(config.device)
        all_items = all_items.repeat(users_size)
        users = users.repeat_interleave(items_size)
        return self(users, all_items)
