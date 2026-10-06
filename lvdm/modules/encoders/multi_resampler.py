# Adapted from DynamiCrafter (https://github.com/Doubiiu/DynamiCrafter) and ViewCrafter
# (https://github.com/Drexubery/ViewCrafter), Apache License 2.0.
# Modified for UniVerse: Multi-input Query Transformer (a variable number of
# input images per sample, padded and masked).
# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math
import torch
import torch.nn as nn


class ImageProjModel(nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        #embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents, masks):
        """
        Args:
            x (list of torch.Tensor): image features
                shape (n1, D) 
            latent (torch.Tensor): latent features
                shape (b, n2, D)
            masks (torch.Tensor): masks
                shape (b,)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape
        
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight[masks] = -torch.inf
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)
    


def pad_and_mask(features, latents_len):
    """
    Pads a list of feature tensors to the same length and creates a mask.

    Args:
        features (list of torch.Tensor): List of feature tensors with shapes (n_i, dim).
        dim (int): The embedding dimension of the features.

    Returns:
        padded_features (torch.Tensor): Padded feature tensor with shape (batch_size, max_n, dim).
        mask (torch.Tensor): Mask tensor with shape (batch_size, max_n).
    """
    max_len = max(f.shape[0] for f in features)
    batch_size = len(features)
    dim = features[-1].shape[-1]

    padded_features = torch.zeros((batch_size, max_len, dim), device = features[-1].device)
    mask = torch.zeros((batch_size, max_len + latents_len), dtype=torch.bool, device= features[-1].device)

    for i, f in enumerate(features):
        padded_features[i, :f.shape[0], :] = f
        mask[i, f.shape[0]:-latents_len] = True
    return padded_features, mask

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        video_length=None, # using frame-wise version or not
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries 
        self.video_length = video_length
        self.heads=heads
        ## <num_queries> queries for each frame
        if video_length is not None: 
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        x, masks = pad_and_mask(x,self.latents.shape[1])
        latents = self.latents.repeat(x.size(0), 1, 1) ## B (T L) C
        masks = masks[:,None,None,:].repeat(1,self.heads,latents.shape[1],1)
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents, masks) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents) # B L C or B (T L) C

        return latents
