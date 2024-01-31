import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender


class seq_prompt(SequentialRecommender):
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

        self.prompts = nn.Parameter(torch.randn(1024, self.hidden_size))
        nn.init.xavier_uniform_(self.prompts)

        self.vqrec = vqrec

        self.attn_layer = nn.MultiheadAttention(self.hidden_size, num_heads=4)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_seq, item_seq_len):
        return self.vqrec.forward(item_seq, item_seq_len)

    def calculate_item_emb(self):
        pq_code_emb = self.vqrec.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb

    def get_prompt(self, batch, seq_out):
        prompt = self.prompts.unsqueeze(1)
        prompt, _ = self.attn_layer(seq_out, prompt, prompt)
        return prompt

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.calculate_item_emb()
        prompt = self.get_prompt(batch=seq_output.shape[0], seq_out=seq_output.unsqueeze(1)).squeeze(1)
        prompt_out = seq_output + prompt
        if self.temperature > 0 :
            prompt_out = F.normalize(prompt_out, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)

        logits = torch.matmul(prompt_out, test_item_emb.transpose(0, 1))
        if self.temperature > 0:
            logits /= self.temperature

        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.calculate_item_emb()
        prompt = self.get_prompt(batch=seq_output.shape[0], seq_out=seq_output.unsqueeze(1)).squeeze(1)
        prompt_out = seq_output + prompt
        if self.temperature > 0:
            prompt_out = F.normalize(prompt_out, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)

        scores = torch.matmul(prompt_out, test_item_emb.transpose(0, 1))
        return scores




