import torch
import torch.nn.functional as F
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionFusion, self).__init__()
        self.query_transform = nn.Linear(input_dim, attention_dim)
        self.key_transform = nn.Linear(input_dim, attention_dim)
        self.value_transform = nn.Linear(input_dim, attention_dim)

    def forward(self, query_input, key_value_input):

        attention_output = self.attention_operation(query_input, key_value_input)

        return attention_output

    def attention_operation(self, q_x, k_v_x):

        query = self.query_transform(q_x)  # [batch_size, seq_len, attention_dim]
        key = self.key_transform(k_v_x)  # [batch_size, seq_len, attention_dim]
        value = self.value_transform(k_v_x)  # [batch_size, seq_len, attention_dim]

        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(query.size(-1), dtype=torch.float32)
        )

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)

        return attention_output
