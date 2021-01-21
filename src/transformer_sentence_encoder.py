# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

from typing import Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
import os 

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                514, #self.max_seq_len,
                self.embedding_dim,
                padding_idx=None, #(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_transformer_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.crossthought_layers = nn.ModuleList([])
        self.crossthought_layers.extend(
            [
                self.build_transformer_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                )
                for _ in range(2)
            ]
        )
        self.pre_train = False if 'CTPRETRAIN' not in os.environ else os.environ['CTPRETRAIN'] == 'True'
        self.short_seq_len = 64
        self.sent_emb_num = 5

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])


    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
    ):
        return TransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        if self.pre_train:
            batch_size_raw, seq_len_raw = tokens.size()
            assert batch_size_raw == 1
            if seq_len_raw % self.short_seq_len == 0:
                tokens = tokens.view(-1, self.short_seq_len)
            else:
                pad_num = (seq_len_raw//self.short_seq_len + 1) * self.short_seq_len - seq_len_raw
                tokens = F.pad(tokens, (0, pad_num), value=self.padding_idx).view(-1, self.short_seq_len)
            batch_size, seq_len = tokens.size()
            tokens = F.pad(tokens, (self.sent_emb_num, 0))
            tokens = tokens.detach()
        else:
            end_index = torch.nonzero(tokens==2, as_tuple=False)[:,1]+1
            tokens_new = []
            batch_size_raw, seq_len_raw = tokens.size()
            for j in range(batch_size_raw):
                tokens_new.append(tokens[j][:end_index[2*j]])
                tokens_new.append(tokens[j][end_index[2*j]: end_index[2*j+1]])
            seq_len = max(v.size(0) for v in tokens_new)
            tokens = tokens.new(len(tokens_new), seq_len).fill_(self.padding_idx)
            for j, v in enumerate(tokens_new):
                tokens[j][:v.size(0)] = v
            if self.sent_emb_num > 1:
                tokens = F.pad(tokens, (self.sent_emb_num-1, 0))


        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        
        if self.embed_positions is not None:
            mask_pos = tokens.ne(self.padding_idx).int()
            if self.pre_train:
                mask_pos[:,0] = 0
                mask_pos[:,self.sent_emb_num] = random.randint(1, 512-self.sent_emb_num-self.short_seq_len-1)
            positions = torch.cumsum(mask_pos, dim=1).type_as(mask_pos).long()
            x = x + self.embed_positions(tokens, positions=positions)


        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        if self.pre_train:
            for layer_id, layer in enumerate(self.crossthought_layers):
                if layer_id == 0:
                    x_sent = x[:self.sent_emb_num].clone().transpose(0, 1)
                    x_sent, _ = layer(x_sent)
                    x[:self.sent_emb_num] = x_sent.transpose(0, 1)
                else:
                    x, _ = layer(x, self_attn_padding_mask=padding_mask)

            x = x[self.sent_emb_num:].transpose(0, 1)
            x = x.contiguous().view(1, -1, x.size(-1))[:,:seq_len_raw].transpose(0, 1)
        else:
            x_sent = x[:self.sent_emb_num].view(self.sent_emb_num, int(x.size(1) / 2), 2, x.size(-1)).clone().transpose(0, 2)
            x_sent = x_sent.contiguous().view(2,-1, x.size(-1))
            x_sent, _ = self.crossthought_layers[0](x_sent)
            x_sent = x_sent.mean(0).view(-1, self.sent_emb_num, x.size(-1))
            x = x_sent.transpose(0, 1)
            x[0] = x[:self.sent_emb_num].mean(0)
            

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
