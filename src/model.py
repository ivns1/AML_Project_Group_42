#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    LATENT_DIM,
    LABEL_SMOOTHING,
    ATTR_W_INIT,
    ATTR_W_FINAL,
    CONS_WEIGHT,
    TOTAL_EPOCHS,
)


class ConvNeXtMultiTask(nn.Module):
    def __init__(self, class_attr_matrix):
        super().__init__()

        self.num_classes = NUM_CLASSES
        self.label_smoothing = LABEL_SMOOTHING
        self.attr_w_init = ATTR_W_INIT
        self.attr_w_final = ATTR_W_FINAL
        self.cons_weight = CONS_WEIGHT
        self.total_epochs = TOTAL_EPOCHS

        config = AutoConfig.from_pretrained(MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, config=config)

        hidden_size = (
            config.hidden_sizes[-1]
            if hasattr(config, "hidden_sizes")
            else config.hidden_size
        )

        self.latent = nn.Linear(hidden_size, LATENT_DIM)
        self.class_head = nn.Linear(LATENT_DIM, NUM_CLASSES)
        self.attr_head = nn.Linear(LATENT_DIM, class_attr_matrix.shape[1])

        self.register_buffer("class_attr", class_attr_matrix.float())

    def _lambda_attr(self, epoch):
        alpha = min(epoch / max(1, self.total_epochs - 1), 1.0)
        return self.attr_w_init + alpha * (self.attr_w_final - self.attr_w_init)

    def forward(self, pixel_values, labels=None, attributes=None, epoch=None):
        out = self.backbone(pixel_values=pixel_values)

        if getattr(out, "pooler_output", None) is not None:
            feats = out.pooler_output
        else:
            feats = out.last_hidden_state.mean(dim=[2, 3])

        z = self.latent(feats)
        class_logits = self.class_head(z)
        attr_logits = self.attr_head(z)

        # INFERENCE
        if labels is None or attributes is None:
            return class_logits

        # CLASSIFICATION LOSS (label smoothing)
        C = class_logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(class_logits)
            true_dist.fill_(self.label_smoothing / (C - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(class_logits, dim=-1)
        L_class = (-true_dist * log_probs).sum(dim=-1).mean()

        # ATTRIBUTE LOSS
        L_attr = F.binary_cross_entropy_with_logits(attr_logits, attributes)

        # CONSISTENCY LOSS
        p_class = F.softmax(class_logits, dim=-1)
        attr_probs = torch.sigmoid(attr_logits)
        logits_attr_based = attr_probs @ self.class_attr.t()
        p_attr = F.softmax(logits_attr_based, dim=-1)
        L_cons = F.kl_div(p_class.log(), p_attr, reduction="batchmean")

        lam_attr = self._lambda_attr(epoch)
        loss = L_class + lam_attr * L_attr + self.cons_weight * L_cons

        return loss, class_logits

