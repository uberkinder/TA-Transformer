import torch
from torch import nn
import torch.nn.functional as F

from tasa.modules import TransformerBlock
from tasa.utils import d


class TATransformer(nn.Module):
    """Time-Aware Transformer"""

    def __init__(self,
                 emb: int, 
                 heads: int, 
                 depth: int, 
                 trail_len: int, 
                 n_actions: int,
                 wide: bool = False):
        super().__init__()

        # actions embedding + temporal encoding
        self.n_actions = n_actions
        self.actions_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=n_actions)
        self.temporal_encoding = nn.Embedding(embedding_dim=emb, num_embeddings=n_actions)

        # transform
        self.tblocks = nn.Sequential(
            TransformerBlock(emb=emb, heads=heads, eq_length=trail_len, wide=wide, mask=True)
            for i in range(depth)
        )

        # softmax
        self.toprobs = nn.Linear(emb, n_actions)

    def forward(self, x):
        """
        Parameters:
        -----------
        x: A (batch, sequence length) integer tensor of token indices.

        Returns:
        --------
        x: predicted log-probability vectors for each token based on the preceding tokens.

        """
        # actions embedding
        actions = self.actions_embedding(x)
        b, t, e = actions.size()

        # temporal encoding
        timestamps = self.temporal_encoding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = actions + timestamps

        # transform
        x = self.tblocks(x)

        # softmax
        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)
        x = F.log_softmax(x, dim=2)

        return x
