import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder,MLPLayers
from recbole.model.abstract_recommender import SequentialRecommender


class id_prompt(SequentialRecommender):
    def __init__(self, config, dataset, vqrec):
        super().__init__(config, dataset)
        self.pq_codes = dataset.pq_codes
        self.index_assignment_flag = False
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']
        self.reassigned_code_embedding = None
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.item_trm_encoder = TransformerEncoder(
            n_layers=2,
            n_heads=2,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.vqrec = vqrec
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.vqrec.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)
        extended_attention_mask = self.vqrec.get_attention_mask(item_seq)
        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.vqrec.forward(item_seq, item_seq_len)
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)
        return seq_output


    def calculate_item_emb(self):
        pq_code_emb = self.vqrec.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.calculate_item_emb()
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores



