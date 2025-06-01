import torch
from torch import nn 
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.dataset import Dataset
from recbole.model.layers import TransformerEncoder
from recbole.data.interaction import Interaction

class HybridRecommenderModel(SequentialRecommender):
    def __init__(self, config, dataset: Dataset):
        super().__init__(config, dataset)

        # Variable initialisation

        # --- SLM embeddings initialisation ---
        slm_pretrained_weights = dataset.get_preload_weight(self.item_id_field)
        if slm_pretrained_weights is None:
            raise ValueError("Preloaded SLM embeddings not found!")

        self.slm_embedding_dim = slm_pretrained_weights.shape[1]

        # freeze set to true to prevent fine-tuning of embeddings
        # this ensures pretrained SLM embeddings are not fine-tuned/updated during training
        # this preserves information obtained from SLM
        # any performance differential can be more clearly attributed to the impact of adding SLM embeddings
        self.slm_item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(slm_pretrained_weights), freeze=True)

        # --- Bert4Rec Transformer Encoder variables initialisation ---
        # from RecBole's BERT4Rec implementation
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config[
            "inner_size"
        ]  
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        try:
            assert self.loss_type in ["BPR", "CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # Projection layer to match SLM embeddings to hidden_size for fusion
        # This implies that eventhough SLM sizes are different, we are treating them as the same
        self.slm_projection = nn.Linear(self.slm_embedding_dim, self.hidden_size)
        
        # Concatenated embedding dimension
        # Size is double hidden size since we projected the embeddings into the shape of the hidden size
        self.fused_embedding_size = self.hidden_size * 2 

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last

        # Transformer initialisation
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        # --- Bert4Rec forward layer initialisation ---
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Input the fused embedding instead of just the sequential embedding into the linear layer
        self.fusion_layer = nn.Linear(self.fused_embedding_size, self.hidden_size)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # --- Initialise weights ---
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reconstruct_test_data(self, item_seq, item_seq_len):
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq
    
    def forward(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        item_emb = self.item_embedding(item_seq)
        
        # Get SLM contextual embeddings
        # Handle mask token in case SLM embeddings are not present for item
        item_seq_for_slm = torch.clamp(item_seq, 0, self.n_items - 1)
        slm_emb = self.slm_item_embedding(item_seq_for_slm) # [B, L, slm_dim]
        slm_emb = self.slm_projection(slm_emb) # [B, L, hidden_size]

        # Create mask for actual items (not padding or mask tokens)
        item_mask = (item_seq > 0).float().unsqueeze(-1) # [B, L, 1]

        # Apply mask to SLM embeddings to zero out padding/mask positions
        slm_emb = slm_emb * item_mask

        # Fuse embeddings -> sequential (item + position) + SLM contextual
        sequential_emb = item_emb + position_embedding # [B, L, hidden size]
        fused_emb = torch.cat([sequential_emb, slm_emb], dim=1) # [B, L, hidden size * 2]
        fused_emb = self.fusion_layer(fused_emb) # [B, L, hidden_size]
        
        # Apply layer norm and dropout
        input_emb = torch.concat(item_emb + position_embedding, slm_emb)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Pass through transformer with bidirectional attention (BERT-style)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        # Final output layers
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        
        return output  # [B, L, H]

    def multi_hot_embed(self, masked_index, max_length):
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        neg_items = interaction[self.NEG_ITEMS]
        masked_index = interaction[self.MASK_INDEX]

        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(
            masked_index, masked_item_seq.size(-1)
        )  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)  # [B mask_len H]
            neg_items_emb = self.item_embedding(neg_items)  # [B mask_len H]
            pos_score = (
                torch.sum(seq_output * pos_items_emb, dim=-1)
                + self.output_bias[pos_items]
            )  # [B mask_len]
            neg_score = (
                torch.sum(seq_output * neg_items_emb, dim=-1)
                + self.output_bias[neg_items]
            )  # [B mask_len]
            targets = (masked_index > 0).float()
            loss = -torch.sum(
                torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets
            ) / torch.sum(targets)
            return loss

        elif self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            test_item_emb = self.item_embedding.weight[: self.n_items]  # [item_num H]
            logits = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                + self.output_bias
            )  # [B mask_len item_num]
            targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
                * targets
            ) / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, H]

        test_item_emb = self.item_embedding(test_item)
        scores = (torch.mul(seq_output, test_item_emb)).sum(dim=1) + self.output_bias[
            test_item
        ]  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_items_emb = self.item_embedding.weight[
            : self.n_items
        ]  # delete masked token
        scores = (
            torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        )  # [B, item_num]
        return scores
    