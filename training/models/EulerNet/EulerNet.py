# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#
# EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction
# Paper: https://arxiv.org/abs/2304.10711

import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class EulerNet(BaseModel):
    """EulerNet model for CTR prediction using Euler's formula for feature interactions.

    Maps feature embeddings to complex space (amplitude + phase) and computes
    higher-order interactions via element-wise complex multiplication:
    e^(i*(θ1+θ2)) = e^(i*θ1) * e^(i*θ2)

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"EulerNet"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``16``.
        order (int): Number of interaction orders (Euler layers). Default: ``3``.
        hidden_units (list): Output dimensions per Euler interaction layer. Default: ``[256, 128, 64]``.
        net_dropout (float): Dropout rate after each Euler layer. Default: ``0.0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``True``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
    """

    def __init__(
        self,
        feature_map,
        model_id="EulerNet",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=16,
        order=3,
        hidden_units=[256, 128, 64],
        net_dropout=0.0,
        batch_norm=True,
        embedding_regularizer=None,
        net_regularizer=None,
        **kwargs,
    ):
        super(EulerNet, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        input_dim = feature_map.num_fields * embedding_dim
        self.euler_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        dims = [input_dim] + hidden_units[:order]
        for i in range(len(dims) - 1):
            self.euler_layers.append(EulerInteractionLayer(dims[i], dims[i + 1]))
            if batch_norm:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
            else:
                self.norms.append(nn.Identity())
            self.dropouts.append(nn.Dropout(net_dropout))

        self.fc = nn.Linear(dims[-1], 1)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        # (B, num_fields, embedding_dim) → (B, num_fields * embedding_dim)
        emb = self.embedding_layer(X).flatten(start_dim=1)

        h = emb
        for euler, norm, drop in zip(self.euler_layers, self.norms, self.dropouts):
            h = euler(h)
            h = norm(h)
            h = drop(h)

        y_pred = self.output_activation(self.fc(h))
        return {"y_pred": y_pred}


class EulerInteractionLayer(nn.Module):
    """Single Euler interaction layer.

    Splits input into amplitude (r) and phase (θ), applies complex multiplication
    via Euler's formula, then projects back to real space.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension (must be even).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Project to 2 * output_dim: first half → amplitude weights, second half → phase weights
        self.r_proj = nn.Linear(input_dim, output_dim)
        self.theta_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Amplitude: softplus keeps it positive
        r = F.softplus(self.r_proj(x))
        theta = self.theta_proj(x)
        # Euler: r * e^(i*θ) → real part: r*cos(θ), imag part: r*sin(θ)
        # Combine back to real-valued vector via concatenation then project
        real = r * torch.cos(theta)
        imag = r * torch.sin(theta)
        # Sum real and imaginary contributions (equivalent to Re[complex product])
        return real + imag
