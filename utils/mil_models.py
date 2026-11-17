"""Multiple instance learning architectures for drinking-status classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def class_prototypes(text_embeddings):
    """Validate the mean-pooled CONCH prototype for each class."""
    prototypes = torch.tensor(text_embeddings, dtype=torch.float32)
    if prototypes.shape != (2, 512) or not torch.isfinite(prototypes).all():
        raise ValueError("Expected two finite, mean-pooled class prototypes of dimension 512.")
    return prototypes


class ABMIL(nn.Module):
    """
    Standard Attention-Based Multiple Instance Learning (ABMIL) aggregation.
    """
    def __init__(self, L=512, D=64, K=1):
        super().__init__()
        self.L, self.D, self.K = L, D, K
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.attention_V[0].weight)
        nn.init.zeros_(self.attention_V[0].bias)
        nn.init.xavier_uniform_(self.attention_U[0].weight)
        nn.init.zeros_(self.attention_U[0].bias)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        if self.attention_weights.bias is not None:
            nn.init.zeros_(self.attention_weights.bias)

    def forward(self, features):
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        raw_scores = self.attention_weights(A_V * A_U)
        
        w = torch.softmax(raw_scores / 0.33, dim=0)
        return torch.mm(w.t(), features), w

class MaxPoolModel(nn.Module):
    """Max-pooled CONCH features compared with fixed class text prototypes."""

    def __init__(self, text_embeddings):
        super().__init__()
        self.register_buffer("text_embeddings", class_prototypes(text_embeddings))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))

    def forward(self, features):
        embedding = torch.max(features, dim=0, keepdim=True)[0]
        N = features.shape[0]
        w = torch.ones((N, 1), device=features.device, dtype=features.dtype) / N
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        raw_sims = embedding_norm @ prototype_norm.t()
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        return raw_sims * clamped_scale, embedding_norm, w


class ABMILAdapterModel(nn.Module):
    def __init__(self, text_embeddings, reduction=4, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.reduction = reduction
        self.ratio = 0.2
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", class_prototypes(text_embeddings))
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
            
        self.adapter = nn.Sequential(
            nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
            nn.LayerNorm(self.c_in // self.reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
        )
        nn.init.zeros_(self.adapter[4].weight)

    def forward(self, features):
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
            
        res = self.adapter(embedding)
        adapted_emb = self.ratio * res + (1 - self.ratio) * embedding
        adapted_emb = F.normalize(adapted_emb, p=2, dim=-1)
        
        raw_sims = adapted_emb @ prototype_norm.t()
        
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = raw_sims * clamped_scale
        
        return logits, adapted_emb, w

class ZSMILModel(nn.Module):
    def __init__(self, text_embeddings, abmil_d=64):
        super().__init__()
        self.register_buffer("text_embeddings", class_prototypes(text_embeddings))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.aggregator = ABMIL(L=512, D=abmil_d)
        
    def forward(self, features):
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        raw_sims = embedding_norm @ prototype_norm.t()

        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = raw_sims * clamped_scale
        
        return logits, embedding_norm, w

class TaskResModel(nn.Module):
    def __init__(self, text_embeddings, alpha=0.25, abmil_d=64, dropout_p=0.1):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("text_embeddings", class_prototypes(text_embeddings))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        
        self.task_res = nn.Parameter(torch.empty_like(self.text_embeddings))
        nn.init.normal_(self.task_res, mean=0.0, std=0.02)

        self.task_res_drop = nn.Dropout(p=dropout_p)
        
        self.aggregator = ABMIL(L=512, D=abmil_d)

    def forward(self, features):
        dropped_res = self.task_res_drop(self.task_res)
        adapted_prototype = self.text_embeddings + self.alpha * dropped_res
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        raw_sims = embedding_norm @ prototype_norm.t()
        
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = raw_sims * clamped_scale
        
        return logits, embedding_norm, w
        


def build_model(args, text_embeddings):
    """Construct one of the four architectures evaluated in the paper."""
    if args.model == "max_pool":
        return MaxPoolModel(text_embeddings)
    if args.model == "abmil_adapter":
        return ABMILAdapterModel(text_embeddings, reduction=args.reduction, abmil_d=args.abmil_d)
    if args.model == "zsmil":
        return ZSMILModel(text_embeddings, abmil_d=args.abmil_d)
    if args.model == "taskres":
        return TaskResModel(text_embeddings, alpha=args.alpha, abmil_d=args.abmil_d,
                            dropout_p=args.taskres_drop)
    raise ValueError(f"Unknown model: {args.model}")
