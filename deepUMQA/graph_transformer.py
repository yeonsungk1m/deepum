"""Graph-attention-based model that consumes DeepUMQA features.

This module repurposes the graph-biased Transformer LossNet (see the user-provided
reference) to work with the featurization pipeline already implemented in this
repository. The original ResNet model operated on 3D voxel grids and 2D maps;
here we directly treat residues as graph nodes and pairwise signals as edges so
that the rich geometric descriptors extracted in :mod:`deepUMQA.featurize` can
be used without going through convolutional backbones.

Node features (per-residue)
---------------------------
- Backbone torsions (``phi``/``psi`` encoded as sine/cosine) capture local
  conformation and remain smooth across angular wrap-around.
- One-body terms from :func:`extractOneBodyTerms` mix bond geometry, backbone
  energy terms (``p_aa_pp``, ``rama_prepro``, ``omega``, ``fa_dun``), and DSSP
  secondary structure one-hots. These summarize how well each residue is
  individually packed.
- Sequence/physicochemical descriptors from
  :func:`extract_AAs_properties_ver1` (20 AA one-hot + BLOSUM + position +
  Meiler features) plus USR shape signatures give per-residue evolutionary and
  shape context. These were already concatenated into ``prop`` during
  featurization.
- Optionally, a pooled 3D voxel descriptor (mirroring the original 3D CNN
  stem) is concatenated so voxelized atomic neighborhoods can inform the
  graph encoder.

Edge features (pairwise)
------------------------
- ``tbt`` from :func:`extract_EnergyDistM` contains CB–CB distances, pairwise
  Rosetta energies, and hydrogen-bond indicators; the first channel is the raw
  distance matrix, which we use to define adjacency.
- ``maps`` from :func:`extract_multi_distance_map` supplies multiple atom-pair
  distances (CB–CB, tip–tip, CA–tip, tip–CA) that enrich geometric context.
- ``euler`` and ``omega6d``/``theta6d``/``phi6d`` give relative orientation
  information between local frames; these are sinusoidal-expanded before being
  concatenated in the dataset and are preserved here for edge decoding.
- ``seqsep`` expresses sequence separation; we keep it as an edge feature so
  the model can disambiguate short-range versus long-range contacts even when
  geometric distances are similar.

Design choices
--------------
- Adjacency is derived from the (inverse-transformed) CB–CB distance map so the
  attention bias follows the actual spatial neighborhood instead of grid
  connectivity. A configurable cutoff (``neighbor_cutoff``) controls sparsity.
- The bias builder mirrors the original LossNet logic (degree bias, optional
  soft masks, shortest-path distance scaling, and learnable CLS couplings) but
  consumes a per-graph adjacency matrix so variable-length proteins work.
- Pairwise predictions are produced by combining contextualized node embeddings
  with projected edge features. Outputs match the original training targets:
  deviation logits (15 bins), a binary mask logit, and per-residue lDDT.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepUMQA.mymodel import calculate_LDDT


# Size of the pooled voxel descriptor produced by :class:`VoxelEncoder`.
VOXEL_FEATURE_DIM = 640


def invert_dist_transform(x: torch.Tensor, *, cutoff: float = 4.0, scaling: float = 3.0) -> torch.Tensor:
    """Invert ``dist_transform``/``transform`` used during featurization.

    The datasets store distances as ``arcsinh(max(d, cutoff) - cutoff) / scaling``.
    Recovering approximate raw distances lets us threshold edges in ångströms.
    """

    return scaling * torch.sinh(x) + cutoff


def floyd_warshall(adj: torch.Tensor) -> torch.Tensor:
    """Compute shortest-path hop distances for a batch of unweighted graphs.

    Args:
        adj: (B, N, N) adjacency with non-negative entries.
    Returns:
        dist: (B, N, N) hop counts where dist[b,i,j] is the minimum number of
            hops from i to j in graph b; large value if disconnected.
    """

    assert adj.dim() == 3 and adj.size(1) == adj.size(2), "adj must be (B,N,N)"
    B, N, _ = adj.shape
    device = adj.device
    INF = 1e6

    dist = torch.full((B, N, N), INF, device=device, dtype=torch.float32)
    dist = dist.masked_fill(adj > 0, 1.0)
    idx = torch.arange(N, device=device)
    dist[:, idx, idx] = 0.0

    for k in range(N):
        dist = torch.minimum(dist, dist[:, :, k:k+1] + dist[:, k:k+1, :])
    return dist


def _scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape) -> torch.Tensor:
    """Scatter sparse voxel values into a dense tensor.

    This mirrors the helper used by the original 3D CNN backbone so that we can
    reuse the same voxel featurization path for the graph model. ``indices`` is
    expected to contain ``(residue, x, y, z, channel)`` columns, and ``updates``
    stores the corresponding trilinear interpolation weights.
    """

    device = updates.device

    out = torch.zeros(int(torch.prod(torch.tensor(shape))), device=device)
    multipliers = torch.as_tensor([int(torch.prod(torch.tensor(shape[i + 1 :]))) for i in range(len(shape))], device=device)

    flat_idx = (indices.long() * multipliers.unsqueeze(0)).sum(dim=1)
    out = out.scatter_add(0, flat_idx, updates)
    return out.view(shape)


class VoxelEncoder(nn.Module):
    """Convert residue-centric voxel grids into fixed-length descriptors.

    The encoder mirrors the early 3D-convolutional stem of the original
    DeepUMQA model (three Conv3d layers followed by average pooling), producing
    a 640-dimensional vector per residue. These descriptors can be concatenated
    with the existing 1D node features before projection.
    """

    def __init__(self, *, num_restype: int = 20, grid_size: int = 24, out_dim: int = VOXEL_FEATURE_DIM):
        super().__init__()
        self.num_restype = num_restype
        self.grid_size = grid_size
        self.out_dim = out_dim

        self.retype = nn.Conv3d(num_restype, 20, 1, padding=0, bias=False)
        self.conv3d_1 = nn.Conv3d(20, 20, 3, padding=0, bias=True)
        self.conv3d_2 = nn.Conv3d(20, 30, 4, padding=0, bias=True)
        self.conv3d_3 = nn.Conv3d(30, 10, 4, padding=0, bias=True)
        self.pool3d_1 = nn.AvgPool3d(kernel_size=4, stride=4, padding=0)

    def forward(self, idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        """Encode sparse voxel inputs into (B, N, ``out_dim``) descriptors."""

        if idx.dim() == 2:
            idx = idx.unsqueeze(0)
            val = val.unsqueeze(0)

        batch_voxels = []
        for b in range(idx.size(0)):
            # Residue count is encoded in the first column of ``idx``.
            nres = int(idx[b, :, 0].max().item()) + 1
            grid = _scatter_nd(
                idx[b],
                val[b],
                (nres, self.grid_size, self.grid_size, self.grid_size, self.num_restype),
            )
            grid = grid.permute(0, 4, 1, 2, 3)

            out_retype = self.retype(grid)
            out_conv3d_1 = F.elu(self.conv3d_1(out_retype))
            out_conv3d_2 = F.elu(self.conv3d_2(out_conv3d_1))
            out_conv3d_3 = F.elu(self.conv3d_3(out_conv3d_2))
            out_pool3d_1 = self.pool3d_1(out_conv3d_3)

            flattened = torch.flatten(out_pool3d_1.permute(0, 2, 3, 4, 1), start_dim=1)
            batch_voxels.append(flattened)

        return torch.stack(batch_voxels, dim=0)


class DynamicAttnBiasBuilder(nn.Module):
    """Build attention bias tensors conditioned on per-graph adjacency.

    Learnable parameters mirror the user-supplied LossNet snippet but we accept
    ``adj``/``edge_type_oh``/``dist`` tensors at runtime so graphs of different
    sizes can be processed in a single model.
    """

    def __init__(
        self,
        *,
        hard_mask: bool = False,
        use_degree: bool = True,
        vnode_spd_mode: str = "replace",
        edge_type_dim: int = 0,
    ):
        super().__init__()
        self.hard_mask = hard_mask
        self.use_degree = use_degree
        self.vnode_spd_mode = vnode_spd_mode
        self.edge_type_weight = nn.Parameter(torch.zeros(max(edge_type_dim, 1)))
        self.noedge_bias = nn.Parameter(torch.tensor(-2.0))
        if use_degree:
            self.deg_scale = nn.Parameter(torch.tensor(0.2))
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.vnode_to_node = nn.Parameter(torch.zeros(1))
        self.node_to_vnode = nn.Parameter(torch.zeros(1))
        self.vnode_self = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        adj: torch.Tensor,
        *,
        dist: Optional[torch.Tensor] = None,
        edge_type_oh: Optional[torch.Tensor] = None,
        H: int = 1,
    ) -> torch.Tensor:
        """Create (B, H, N+1, N+1) attention bias given a batch of graphs."""

        assert adj.dim() == 3 and adj.size(1) == adj.size(2), "adj must be (B,N,N)"
        B, N, _ = adj.shape
        device = adj.device
        T = N + 1
        bias = torch.zeros(B, H, T, T, device=device, dtype=torch.float32)

        # Edge-type term
        if edge_type_oh is not None:
            e = (edge_type_oh * self.edge_type_weight.view(1, 1, 1, -1)).sum(-1)
        else:
            e = torch.zeros_like(adj)

        # Distance term for node-node block
        if (dist is not None) and (float(self.gamma.item()) != 0.0):
            e = e + (-self.gamma) * dist

        # Neighbor / non-neighbor handling
        if self.hard_mask:
            base = torch.full_like(adj, float("-inf"))
            e = torch.where(adj > 0, e, base)
        else:
            e = torch.where(adj > 0, e, e + self.noedge_bias)

        # Degree bias
        if self.use_degree:
            deg = adj.sum(-1)  # (B,N)
            deg_norm = (deg.unsqueeze(2) + deg.unsqueeze(1)) / (deg.max(dim=1, keepdim=True).values.unsqueeze(2) + 1e-6)
            e = e + self.deg_scale * deg_norm

        bias[:, :, 1:, 1:] = e.unsqueeze(1).expand(B, H, N, N)

        # CLS/VNode couplings
        if self.vnode_spd_mode != "off":
            v_spd = torch.tensor(0.0, device=device)
            if (dist is not None) and (float(self.gamma.item()) != 0.0):
                v_spd = (-self.gamma).detach()
            if self.vnode_spd_mode == "replace":
                bias[:, :, 0, 1:] = self.vnode_to_node
                bias[:, :, 1:, 0] = self.node_to_vnode
            elif self.vnode_spd_mode == "add":
                bias[:, :, 0, 1:] = v_spd + self.vnode_to_node
                bias[:, :, 1:, 0] = v_spd + self.node_to_vnode

            if self.vnode_spd_mode == "replace":
                bias[:, :, 0, 0] = self.vnode_self
            elif self.vnode_spd_mode == "add":
                bias[:, :, 0, 0] = v_spd + self.vnode_self

        return bias


class BiasedMHSA(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.dk = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H = self.h
        dk = self.dk
        q = self.q(x).view(B, T, H, dk).transpose(1, 2)
        k = self.k(x).view(B, T, H, dk).transpose(1, 2)
        v = self.v(x).view(B, T, H, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, mlp_ratio: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BiasedMHSA(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_bias))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GraphFeatureNet(nn.Module):
    """Graph-biased Transformer that consumes DeepUMQA node/pair features."""

    def __init__(
        self,
        node_in_dim: int,
        pair_in_dim: int,
        *,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
        num_bins: int = 15,
        neighbor_cutoff: float = 15.0,
        distance_cutoff: float = 4.0,
        distance_scaling: float = 3.0,
        use_distance_bias: bool = True,
        distance_gamma_init: float = 0.8,
        vnode_spd_mode: str = "replace",
        use_voxel: bool = True,
        voxel_restype: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.neighbor_cutoff = neighbor_cutoff
        self.distance_cutoff = distance_cutoff
        self.distance_scaling = distance_scaling

        self.use_voxel = use_voxel
        voxel_dim = VOXEL_FEATURE_DIM if use_voxel else 0
        self.node_proj = nn.Linear(node_in_dim + voxel_dim, d_model)
        self.pair_proj = nn.Linear(pair_in_dim, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if use_voxel:
            self.voxel_encoder = VoxelEncoder(num_restype=voxel_restype)

        self.bias_builder = DynamicAttnBiasBuilder(
            hard_mask=False,
            use_degree=True,
            vnode_spd_mode=vnode_spd_mode,
        )
        if use_distance_bias:
            with torch.no_grad():
                self.bias_builder.gamma.copy_(torch.tensor(distance_gamma_init))

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        self.pair_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.deviation_head = nn.Linear(d_model, num_bins)
        self.mask_head = nn.Linear(d_model, 1)

    def _symmetrize(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x + x.transpose(-1, -2))

    def forward(self, node_feats: torch.Tensor, pair_feats: torch.Tensor, *, idx: Optional[torch.Tensor] = None, val: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            node_feats: (B, N, F1) node features from the dataset (angles + one-body
                terms + properties). When ``use_voxel`` is True, this tensor will be
                concatenated with voxel descriptors derived from ``idx``/``val``.
            pair_feats: (B, C, N, N) pairwise features from the dataset; channel 0
                is assumed to be the distance map (after ``dist_transform``).
            idx / val: Sparse voxel indices/values encoding residue-local grids;
                required when voxel features are enabled.
        Returns:
            deviation_prediction: (B, num_bins, N, N)
            mask_prediction: (B, N, N)
            lddt_prediction: (B, N)
            (deviation_logits, mask_logits): raw logits before activation
        """

        if pair_feats.dim() != 4:
            raise ValueError("pair_feats must be (B,C,N,N)")

        B, C, N, _ = pair_feats.shape
        device = node_feats.device

        if self.use_voxel:
            if idx is None or val is None:
                raise ValueError("Voxel features requested but idx/val tensors were not provided")

            voxel_feats = self.voxel_encoder(idx.to(device), val.to(device))
            if voxel_feats.shape[1] != N:
                raise ValueError(
                    f"Voxel descriptor residue count {voxel_feats.shape[1]} does not match node features ({N})"
                )
            node_feats = torch.cat([node_feats, voxel_feats], dim=-1)

        # Prepare adjacency & distance bias from the first channel (CB–CB distance)
        dist_feat = pair_feats[:, 0]
        dist_raw = invert_dist_transform(dist_feat, cutoff=self.distance_cutoff, scaling=self.distance_scaling)
        adj = (dist_raw < self.neighbor_cutoff).float()
        eye = torch.eye(N, device=device).unsqueeze(0)
        adj = adj * (1 - eye)

        dist_bias = floyd_warshall(adj) if float(self.bias_builder.gamma.item()) != 0.0 else None
        attn_bias = self.bias_builder(adj, dist=dist_bias, H=self.blocks[0].attn.h)

        # Node encoder + CLS
        z = self.node_proj(node_feats)  # (B,N,D)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)

        for blk in self.blocks:
            z = blk(z, attn_bias=attn_bias)
        z = self.ln_f(z)
        nodes = z[:, 1:, :]  # (B,N,D)

        # Pair decoder: combine contextual nodes with projected pair signals
        pair_feats_t = pair_feats.permute(0, 2, 3, 1)  # (B,N,N,C)
        pair_context = self.pair_proj(pair_feats_t)
        node_pair = nodes.unsqueeze(2) + nodes.unsqueeze(1)
        pair_hidden = self.pair_head(pair_context + 0.5 * node_pair)

        deviation_logits = self._symmetrize(self.deviation_head(pair_hidden).permute(0, 3, 1, 2))
        deviation_prediction = F.softmax(deviation_logits, dim=1)

        mask_logits = self._symmetrize(self.mask_head(pair_hidden).squeeze(-1))
        mask_prediction = torch.sigmoid(mask_logits)

        lddt_list = []
        for b in range(B):
            lddt_list.append(calculate_LDDT(deviation_prediction[b], mask_prediction[b]))
        lddt_prediction = torch.stack(lddt_list, dim=0)

        return deviation_prediction, mask_prediction, lddt_prediction, (deviation_logits, mask_logits)

