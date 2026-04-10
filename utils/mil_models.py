"""
mil_models.py
Multiple Instance Learning (MIL) Model Architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, text_embeddings, head_type="none", reduction=4, alpha=0.25):
        super().__init__()
        self.c_in = 512
        self.head_type = head_type
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        
        if self.head_type == "adapter":
            self.reduction = reduction
            self.ratio = 0.2
            self.adapter = nn.Sequential(
                nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                nn.LayerNorm(self.c_in // self.reduction),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
            )
            nn.init.zeros_(self.adapter[4].weight)
        elif self.head_type == "taskres":
            self.alpha = alpha
            self.task_res = nn.Parameter(torch.zeros_like(self.text_embeddings))

    def forward(self, features):
        embedding = torch.max(features, dim=0, keepdim=True)[0]
        
        N = features.shape[0]
        w = torch.ones((N, 1), device=features.device, dtype=features.dtype) / N

        if self.head_type == "adapter":
            res = self.adapter(embedding)
            adapted_emb = self.ratio * res + (1 - self.ratio) * embedding
            prototype = self.text_embeddings
        elif self.head_type == "taskres":
            adapted_emb = embedding
            prototype = self.text_embeddings + self.alpha * self.task_res
        else:
            adapted_emb = embedding
            prototype = self.text_embeddings
            
        embedding_norm = F.normalize(adapted_emb, p=2, dim=-1)
        prototype_norm = F.normalize(prototype, p=2, dim=-1)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w

class ABMILAdapterModel(nn.Module):
    def __init__(self, text_embeddings, reduction=4, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.reduction = reduction
        self.ratio = 0.2
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        
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
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (adapted_emb @ prototype_norm.t()) * clamped_scale
        
        return logits, adapted_emb, w

class ZSMILModel(nn.Module):
    def __init__(self, text_embeddings, abmil_d=64):
        super().__init__()
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.aggregator = ABMIL(L=512, D=abmil_d)

    def forward(self, features):
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
            
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w

class TaskResModel(nn.Module):
    def __init__(self, text_embeddings, alpha=0.25, abmil_d=64):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.task_res = nn.Parameter(torch.zeros_like(self.text_embeddings))
        
        self.aggregator = ABMIL(L=512, D=abmil_d)

    def forward(self, features):
        adapted_prototype = self.text_embeddings + self.alpha * self.task_res
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
            
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w

class PreAggSemanticModel(nn.Module):
    """
    Pre-Aggregation Semantic Adaptation Model.
    Shifts and L2-normalizes every individual tissue patch into the semantic space 
    BEFORE passing them into the ABMIL gated attention mechanism.
    """
    def __init__(self, text_embeddings, reduction=8, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.reduction = reduction
        self.ratio = 0.2
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        
        # Patch-Level translation (bridging modality gap microscopically)
        self.patch_translator = nn.Sequential(
            nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
            nn.LayerNorm(self.c_in // self.reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
        )
        nn.init.zeros_(self.patch_translator[4].weight)
        
        # Standard ABMIL that now evaluates the semantically translated patches
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)

    def forward(self, features):
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        
        # 1. Translate and explicitly L2-normalize every single patch
        res = self.patch_translator(features)
        translated_patches = self.ratio * res + (1 - self.ratio) * features
        translated_patches_norm = F.normalize(translated_patches, p=2, dim=-1)
        
        # 2. Score and aggregate the adapted patches
        embedding, w = self.aggregator(translated_patches_norm)
            
        # 3. Final classification computation
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w
