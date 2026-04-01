"""
Multiple Instance Learning (MIL) Model Architectures.
Contains implementations for various aggregation and adaptation strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) aggregation module.
    Calculates attention weights for a bag of features to form a single representation.
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
        
        # Applying a fixed temperature scaling for more stable attention distributions
        fixed_temp = 0.33  
        w = torch.softmax(raw_scores / fixed_temp, dim=0)
        
        return torch.mm(w.t(), features), w


class ABMILAdapterModel(nn.Module):
    """
    ABMIL model with an adapter bottleneck layer for visual feature adaptation.
    """
    def __init__(self, text_embeddings, reduction=4, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.reduction = reduction
        self.ratio = 0.2
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        
        self.adapter = nn.Sequential(
            nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
            nn.LayerNorm(self.c_in // self.reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
        )
        nn.init.zeros_(self.adapter[4].weight)

    def forward(self, features):
        embedding, w = self.aggregator(features)
        res = self.adapter(embedding)
        
        adapted_emb = self.ratio * res + (1 - self.ratio) * embedding
        adapted_emb = F.normalize(adapted_emb, p=2, dim=-1)
        
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (adapted_emb @ prototype_norm.t()) * clamped_scale
        
        return logits, adapted_emb, w


class ZSMILModel(nn.Module):
    """
    Zero-Shot Multiple Instance Learning Model.
    Compares ABMIL aggregated visual features against frozen text prototypes.
    """
    def __init__(self, text_embeddings, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))

    def forward(self, features):
        embedding, w = self.aggregator(features)
        
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w


class TaskResModel(nn.Module):
    """
    Task-Residual Multiple Instance Learning Model.
    Learns a task-specific residual applied directly to the text prototypes.
    """
    def __init__(self, text_embeddings, alpha=0.25, abmil_d=64):
        super().__init__()
        self.c_in = 512
        self.alpha = alpha
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        
        self.task_res = nn.Parameter(torch.zeros_like(self.text_embeddings))

    def forward(self, features):
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        adapted_prototype = self.text_embeddings + self.alpha * self.task_res
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = (embedding_norm @ prototype_norm.t()) * clamped_scale
        
        return logits, embedding_norm, w


class MaxPoolModel(nn.Module):
    """
    Baseline MIL Model using standard max-pooling for feature aggregation.
    Supports optional projection heads (adapter or task-residual).
    """
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
        
        # Uniform weighting for structural compatibility with standard xAI pipelines
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


class CoOpPromptLearner(nn.Module):
    """
    Learnable Prompt Context for Vision-Language Models (Context Optimization).
    Generates dynamic text embeddings by appending learnable tokens to templates.
    """
    def __init__(self, concepts_dict, templates, clip_model, tokenizer, n_ctx=16):
        super().__init__()
        self.classes = list(concepts_dict.keys())
        self.n_cls = len(self.classes)
        self.n_ctx = n_ctx
        
        self.text_tower = clip_model.text if hasattr(clip_model, 'text') else clip_model
        
        # Dynamically determine the maximum sequence length supported by the text tower
        if self.text_tower.positional_embedding.dim() == 2:
            self.max_seq_len = self.text_tower.positional_embedding.shape[0]
        else:
            self.max_seq_len = self.text_tower.positional_embedding.shape[1]
        
        if hasattr(self.text_tower, 'text_projection') and self.text_tower.text_projection is not None:
            ctx_dim = self.text_tower.text_projection.shape[0]
        else:
            ctx_dim = self.text_tower.token_embedding.weight.shape[1]
            
        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim))
        nn.init.normal_(self.ctx, std=0.02)
        
        all_prompts = []
        self.class_indices = []
        
        current_idx = 0
        for cls_name in self.classes:
            concepts = concepts_dict[cls_name]
            start_idx = current_idx
            for concept in concepts:
                for t in templates:
                    all_prompts.append(t.format(concept))
                    current_idx += 1
            self.class_indices.append((start_idx, current_idx))
            
        tokens_dict = tokenizer(
            all_prompts, 
            padding="max_length", 
            max_length=self.max_seq_len, 
            truncation=True, 
            return_tensors="pt"
        )
        
        self.register_buffer("tokenized_prompts", tokens_dict["input_ids"])
        self.register_buffer("attention_mask", tokens_dict["attention_mask"])
        
        with torch.no_grad():
            embedding = self.text_tower.token_embedding(
                self.tokenized_prompts.to(self.text_tower.token_embedding.weight.device)
            )
            
        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1 : self.max_seq_len - n_ctx, :])  

    def forward(self):
        total_prompts = self.token_prefix.shape[0]
        ctx_expanded = self.ctx.unsqueeze(0).expand(total_prompts, -1, -1)
        
        prompts = torch.cat([self.token_prefix, ctx_expanded, self.token_suffix], dim=1)
        return prompts, self.class_indices


class PromptLearningMILModel(nn.Module):
    """
    MIL Model utilizing Context Optimization (CoOp) for text prompt tuning.
    Supports both latent and logit-based ensembling of learned concepts.
    """
    def __init__(self, clip_model, tokenizer, concepts_dict, templates, n_ctx=16, abmil_d=64, ensemble_mode="latent"):
        super().__init__()
        self.prompt_learner = CoOpPromptLearner(concepts_dict, templates, clip_model, tokenizer, n_ctx)
        self.text_tower = clip_model.text if hasattr(clip_model, 'text') else clip_model
        
        self.ensemble_mode = ensemble_mode 
        self.c_in = 512
        self.aggregator = ABMIL(L=self.c_in, D=abmil_d)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))

    def forward(self, features):
        # Generate text representations
        prompts, class_indices = self.prompt_learner()
        
        x = prompts + self.text_tower.positional_embedding
        x = x.permute(1, 0, 2)  
        x = self.text_tower.transformer(x, attn_mask=self.text_tower.attn_mask)
        x = x.permute(1, 0, 2)  
        x = self.text_tower.ln_final(x)
        
        eos_indices = self.prompt_learner.attention_mask.sum(dim=-1) - 1 + self.prompt_learner.n_ctx
        text_features = x[torch.arange(x.shape[0]), eos_indices]
        
        if hasattr(self.text_tower, 'text_projection') and self.text_tower.text_projection is not None:
            text_features = text_features @ self.text_tower.text_projection
            
        if self.ensemble_mode == "latent":
            class_prototypes = []
            for (start_idx, end_idx) in class_indices:
                cls_feats = text_features[start_idx:end_idx]
                cls_proto = torch.mean(cls_feats, dim=0) 
                class_prototypes.append(cls_proto)
            
            class_prototypes = torch.stack(class_prototypes) 
            text_features_norm = F.normalize(class_prototypes, p=2, dim=-1)
        else:
            text_features_norm = F.normalize(text_features, p=2, dim=-1) 
        
        # Generate visual representations
        embedding, w = self.aggregator(features) 
        embedding_norm = F.normalize(embedding, p=2, dim=-1) 
        
        # Aggregate similarity scores
        if self.ensemble_mode == "latent":
            logits = embedding_norm @ text_features_norm.t() 
        else:
            raw_sims = embedding_norm @ text_features_norm.t() 
            class_logits = []
            for (start_idx, end_idx) in class_indices:
                cls_score = torch.mean(raw_sims[:, start_idx:end_idx], dim=1, keepdim=True)
                class_logits.append(cls_score)
                
            logits = torch.cat(class_logits, dim=1) 
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = logits * clamped_scale
        
        return logits, embedding_norm, w
