"""
mil_models.py
Multiple Instance Learning (MIL) Model Architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_text_pooling(logits, text_pool, num_healthy):
    """Pools text similarities to 2 classes if LogSumExp is enabled."""
    if text_pool == "mean":
        if logits.shape[1] != 2:
            raise ValueError("Mean pooling requires exactly two class prototypes.")
        return logits # Shape is already (N, 2)
    if text_pool != "lse":
        raise ValueError(f"Unknown text pooling mode: {text_pool}")
    num_ald = logits.shape[1] - num_healthy
    if num_healthy < 1 or num_ald < 1:
        raise ValueError("LSE pooling requires at least one prompt in each class.")
    
    logit_healthy = torch.logsumexp(logits[:, :num_healthy], dim=1) - np.log(num_healthy)
    logit_ald = torch.logsumexp(logits[:, num_healthy:], dim=1) - np.log(num_ald)
    
    return torch.stack([logit_healthy, logit_ald], dim=-1)
    
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
    def __init__(self, text_embeddings, head_type="none", reduction=4, alpha=0.25, num_healthy_prompts=10, text_pool="mean"):
        super().__init__()
        self.c_in = 512
        self.head_type = head_type
        self.text_pool = text_pool
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.num_healthy_prompts = num_healthy_prompts
        
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
        
        raw_sims = embedding_norm @ prototype_norm.t()
        
        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, embedding_norm, w

class ABMILAdapterModel(nn.Module):
    def __init__(self, text_embeddings, reduction=4, abmil_d=64, text_pool="mean", num_healthy_prompts=10):
        super().__init__()
        self.c_in = 512
        self.reduction = reduction
        self.ratio = 0.2
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.text_pool = text_pool
        self.num_healthy_prompts = num_healthy_prompts
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
        
        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, adapted_emb, w

class ZSMILModel(nn.Module):
    def __init__(self, text_embeddings, abmil_d=64, text_pool="mean", num_healthy_prompts=10):
        super().__init__()
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.aggregator = ABMIL(L=512, D=abmil_d)
        self.text_pool = text_pool
        self.num_healthy_prompts = num_healthy_prompts # Store it
        
    def forward(self, features):
        prototype_norm = F.normalize(self.text_embeddings, p=2, dim=-1)
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        raw_sims = embedding_norm @ prototype_norm.t()

        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, embedding_norm, w

class TaskResModel(nn.Module):
    def __init__(self, text_embeddings, alpha=0.25, abmil_d=64, text_pool="mean", dropout_p=0.1, num_healthy_prompts=10):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        
        #self.task_res = nn.Parameter(torch.zeros_like(self.text_embeddings))
        self.task_res = nn.Parameter(torch.empty_like(self.text_embeddings))
        nn.init.normal_(self.task_res, mean=0.0, std=0.02)

        self.task_res_drop = nn.Dropout(p=dropout_p)
        
        self.text_pool = text_pool
        self.aggregator = ABMIL(L=512, D=abmil_d)
        self.num_healthy_prompts = num_healthy_prompts # Store it

    def forward(self, features):
        dropped_res = self.task_res_drop(self.task_res)
        adapted_prototype = self.text_embeddings + self.alpha * dropped_res
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        raw_sims = embedding_norm @ prototype_norm.t()
        
        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, embedding_norm, w
        
class DualAdaptModel(nn.Module):
    """
    Combines both Visual Adapter (MLP) and Semantic Adapter (TaskRes)
    for simultaneous dual-modality alignment.
    """
    def __init__(self, text_embeddings, reduction=4, alpha=0.25, abmil_d=64, text_pool="mean", num_healthy_prompts=10):
        super().__init__()
        self.c_in = 512
        self.reduction = reduction
        self.alpha = alpha
        self.ratio = 0.2
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.task_res = nn.Parameter(torch.zeros_like(self.text_embeddings))
        self.text_pool = text_pool
        self.num_healthy_prompts = num_healthy_prompts
        
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
        adapted_prototype = self.text_embeddings + self.alpha * self.task_res
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
            
        res = self.adapter(embedding)
        adapted_emb = self.ratio * res + (1 - self.ratio) * embedding
        adapted_emb_norm = F.normalize(adapted_emb, p=2, dim=-1)
        
        raw_sims = adapted_emb_norm @ prototype_norm.t()
        
        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, adapted_emb_norm, w  
        
        
class NLTaskResModel(nn.Module):
    """
    Non-Linear Semantic Adapter (NL-TaskRes).
    Passes the base textual prototypes through a bottleneck MLP.
    """
    def __init__(self, text_embeddings, alpha=0.25, reduction=8, abmil_d=64, text_pool="mean", num_healthy_prompts=10):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.text_pool = text_pool
        self.num_healthy_prompts = num_healthy_prompts
        self.aggregator = ABMIL(L=512, D=abmil_d)
        
        self.semantic_adapter = nn.Sequential(
            nn.Linear(512, 512 // self.reduction, bias=False),
            nn.LayerNorm(512 // self.reduction),
            nn.GELU(),
            nn.Linear(512 // self.reduction, 512, bias=False)
        )
        nn.init.zeros_(self.semantic_adapter[3].weight)

    def forward(self, features):
        residual = self.semantic_adapter(self.text_embeddings)
        adapted_prototype = self.text_embeddings + self.alpha * residual
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        raw_sims = embedding_norm @ prototype_norm.t()
        
        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, embedding_norm, w
        
class AffineTaskResModel(nn.Module):
    """
    Scale and Shift Semantic Adapter.
    Applies an affine transformation (Feature-wise Linear Modulation) to the 
    base text embeddings, allowing the model to stretch/mute features (Scale) 
    before translating them (Shift).
    """
    def __init__(self, text_embeddings, alpha=0.25, abmil_d=64, text_pool="mean", num_healthy_prompts=10):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.text_pool = text_pool
        self.num_healthy_prompts = num_healthy_prompts
        self.aggregator = ABMIL(L=512, D=abmil_d)

        self.task_scale = nn.Parameter(torch.ones_like(self.text_embeddings))
        self.task_shift = nn.Parameter(torch.zeros_like(self.text_embeddings))

    def forward(self, features):
        # Apply Scale (Element-wise multiplication) and Shift (Addition)
        adapted_prototype = (self.task_scale * self.text_embeddings) + (self.alpha * self.task_shift)
        prototype_norm = F.normalize(adapted_prototype, p=2, dim=-1)
        
        embedding, w = self.aggregator(features)
        embedding_norm = F.normalize(embedding, p=2, dim=-1)
        
        raw_sims = embedding_norm @ prototype_norm.t()
        
        pooled_sims = apply_text_pooling(raw_sims, self.text_pool, self.num_healthy_prompts)
        
        clamped_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        logits = pooled_sims * clamped_scale
        
        return logits, embedding_norm, w


BASELINE_MODELS = ("mean_logreg", "score_calibration")


class MeanLogisticModel(nn.Module):
    """Frozen mean slide features + an L2-regularized binary linear classifier."""

    def __init__(self, input_dim=512):
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.classifier = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def classifier_features(self, features):
        return F.normalize(features.mean(dim=0, keepdim=True), dim=-1)

    def forward(self, features):
        representation = self.classifier_features(features)
        decision = self.classifier((representation - self.feature_center) / self.feature_scale)
        logits = torch.cat([torch.zeros_like(decision), decision], dim=1)
        weights = features.new_full((len(features), 1), 1.0 / len(features))
        return logits, representation, weights


class ScoreCalibrationModel(MeanLogisticModel):
    """Learn just weight and bias on the frozen Max Pool text-score difference.

    A negative weight may be learned from TRAINING subjects. Neither test AUC
    nor test labels choose the direction. No attention or adapter is trained.
    """

    def __init__(self, text_embeddings, text_pool="mean", num_healthy_prompts=10):
        super().__init__(input_dim=1)
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.text_pool, self.num_healthy_prompts = text_pool, num_healthy_prompts

    def classifier_features(self, features):
        embedding = F.normalize(features.max(dim=0, keepdim=True).values, dim=-1)
        similarities = embedding @ F.normalize(self.text_embeddings, dim=-1).t()
        classes = apply_text_pooling(similarities, self.text_pool, self.num_healthy_prompts)
        return classes[:, 1:2] - classes[:, :1]


def build_model(args, text_embeddings):
    """Use the identical architecture for training, parameter counts, and xAI."""
    if args.model == "mean_logreg":
        return MeanLogisticModel()
    if args.model == "score_calibration":
        return ScoreCalibrationModel(text_embeddings, args.text_pool, args.num_healthy_prompts)
    common = dict(text_embeddings=text_embeddings, text_pool=args.text_pool,
                  num_healthy_prompts=args.num_healthy_prompts)
    if args.model == "max_pool":
        return MaxPoolModel(**common, head_type=args.max_pool_head,
                            reduction=args.reduction, alpha=args.alpha)
    common["abmil_d"] = args.abmil_d
    if args.model == "abmil_adapter":
        return ABMILAdapterModel(**common, reduction=args.reduction)
    if args.model == "zsmil":
        return ZSMILModel(**common)
    common["alpha"] = args.alpha
    if args.model == "taskres":
        return TaskResModel(**common, dropout_p=args.taskres_drop)
    if args.model == "dual_adapt":
        return DualAdaptModel(**common, reduction=args.reduction)
    if args.model == "nl_taskres":
        return NLTaskResModel(**common, reduction=args.reduction)
    if args.model == "affine_taskres":
        return AffineTaskResModel(**common)
    raise ValueError(f"Unknown model: {args.model}")
