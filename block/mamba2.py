# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modified by Wangzhenbin in 2024

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

# By Wangzhenbin
# @torch.compile
def merge_permutation(xz_main, _perm):
    return xz_main[:, _perm, :].contiguous()  # [B,C,T]

def scan_permutation(o_main, _perm_rev):
    return o_main[:, _perm_rev, :].contiguous()  # out is [B,T,C]

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, order: list, order_reversal: list, origina: list, origina_reversal): 
        B, C, D = x.shape
        ctx.order = order
        ctx.order_reversal = order_reversal
        ctx.origina = origina
        ctx.origina_reversal = origina_reversal
        ctx.shape = (B, C, D)
        xs = x.new_empty((B, 3, C, D))
        xs[:, 0] = x
        xs[:, 1] = scan_permutation(x, order)
        xs[:, 2] = scan_permutation(x, order_reversal)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        order = ctx.order
        order_reversal = ctx.order_reversal
        origina = ctx.origina
        origina_reversal = ctx.origina_reversal
        y = ys[:,0]
        y = y + scan_permutation(ys[:, 1], origina)
        y = y + scan_permutation(ys[:, 2], origina_reversal)
        return y, None, None, None, None

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, order: list, order_reversal: list, origina: list, origina_reversal: list):
        ctx.order = order
        ctx.order_reversal = order_reversal
        ctx.origina = origina
        ctx.origina_reversal = origina_reversal
        y = ys[:,0]
        y = y + merge_permutation(ys[:, 1], origina)
        y = y + merge_permutation(ys[:, 2], origina_reversal)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        order = ctx.order
        order_reversal = ctx.order_reversal
        origina = ctx.origina
        origina_reversal = ctx.origina_reversal
        B, C, D = x.shape
        xs = x.new_empty((B, 3, C, D))
        xs[:, 0] = x
        xs[:, 1] = merge_permutation(x, order)
        xs[:, 2] = merge_permutation(x, order_reversal)
        return xs, None, None, None, None
    
############################################
class Zig_CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, order: list, origina: list): 
        B, C, D = x.shape
        ctx.order = order
        ctx.origina = origina
        ctx.shape = (B, C, D)
        xs = scan_permutation(x, order)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        order = ctx.order
        origina = ctx.origina
        y = scan_permutation(ys, origina)
        return y, None, None

class Zig_CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, order: list, origina: list):
        ctx.order = order
        ctx.origina = origina
        y = merge_permutation(ys, origina)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        order = ctx.order
        origina = ctx.origina
        B, C, D = x.shape
        xs = merge_permutation(x, order)
        return xs, None, None

#######################################
class CrossScan_vmamba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, order: list, origina: list): 
        B, C, D = x.shape
        ctx.order = order
        ctx.origina = origina
        ctx.shape = (B, C, D)
        xs = x.new_empty((B, 4, C, D))
        xs[:, 0] = scan_permutation(x, order[0])
        xs[:, 1] = scan_permutation(x, order[1])
        xs[:, 2] = scan_permutation(x, order[2])
        xs[:, 3] = scan_permutation(x, order[3])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        order = ctx.order
        origina = ctx.origina
        y = scan_permutation(ys[:, 0], origina[0])
        y = y + scan_permutation(ys[:, 1], origina[1])
        y = y + scan_permutation(ys[:, 2], origina[2])
        y = y + scan_permutation(ys[:, 3], origina[3])
        return y, None, None

class CrossMerge_vmamba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, order: list, origina: list):
        ctx.order = order
        ctx.origina = origina
        y = merge_permutation(ys[:, 0], origina[0])
        y = y + merge_permutation(ys[:, 1], origina[1])
        y = y + merge_permutation(ys[:, 2], origina[2])
        y = y + merge_permutation(ys[:, 3], origina[3])
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        order = ctx.order
        origina = ctx.origina
        B, C, D = x.shape
        xs = x.new_empty((B, 4, C, D))
        xs[:, 0] = merge_permutation(x, order[0])
        xs[:, 1] = merge_permutation(x, order[1])
        xs[:, 2] = merge_permutation(x, order[2])
        xs[:, 3] = merge_permutation(x, order[3])
        return xs, None, None  


#######################################################
class CrossScan_EfficientVMamba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor): 
        B, C, D = x.shape
        ctx.shape = (B, C, D)
        ctx.sqrt_c = int(C ** 0.5)
        reshaped_x = x.view(B, ctx.sqrt_c, ctx.sqrt_c, D)
        HW = ctx.sqrt_c // 2
        xs = x.new_empty((B, 4, HW * HW, D))

        xs[:, 0] = reshaped_x[:, ::2, ::2, :].contiguous().view(B, -1, D)
        xs[:, 1] = reshaped_x.transpose(dim0=1, dim1=2)[:, ::2, 1::2, :].contiguous().view(B, -1, D)
        xs[:, 2] = reshaped_x[:, ::2, 1::2, :].contiguous().view(B, -1, D)
        xs[:, 3] = reshaped_x.transpose(dim0=1, dim1=2)[:, 1::2, 1::2, :].contiguous().view(B, -1, D)

        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        HW = int(math.sqrt(C))
        newH, newW = math.ceil(HW / 2), math.ceil(HW / 2)
        y = ys.new_empty(B, HW, HW, D)

        y[:, ::2, ::2, :] = ys[:, 0].reshape(B, newH, newW, D)
        y[:, 1::2, ::2, :] = ys[:, 1].reshape(B, newW, newH, D).transpose(dim0=1, dim1=2)
        y[:, ::2, 1::2, :] = ys[:, 2].reshape(B, newH, newW, D)
        y[:, 1::2, 1::2, :] = ys[:, 3].reshape(B, newW, newH, D).transpose(dim0=1, dim1=2)
        y = y.reshape(B, -1, D)

        return y


class CrossMerge_EfficientVMamba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor): 
        B, _, C, D = ys.shape
        C = C*4
        ctx.shape = (B, C, D)
        HW = int(math.sqrt(C))
        y = ys.new_empty(B, HW, HW, D)

        y[:, ::2, ::2, :] = ys[:, 0].reshape(B, int(HW / 2), int(HW / 2), D)
        y[:, 1::2, ::2, :] = ys[:, 1].reshape(B, int(HW / 2), int(HW / 2), D).transpose(dim0=1, dim1=2)
        y[:, ::2, 1::2, :] = ys[:, 2].reshape(B, int(HW / 2), int(HW / 2), D)
        y[:, 1::2, 1::2, :] = ys[:, 3].reshape(B, int(HW / 2), int(HW / 2), D).transpose(dim0=1, dim1=2)
        y = y.reshape(B, -1, D)

        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, D = ctx.shape
        HW = int(C ** 0.5)
        reshaped_x = x.view(B, HW, HW, D)
        new_HW = int(HW / 2)
        xs = x.new_empty((B, 4, new_HW * new_HW, D))

        xs[:, 0] = reshaped_x[:, ::2, ::2, :].contiguous().view(B, -1, D)
        xs[:, 1] = reshaped_x.transpose(dim0=1, dim1=2)[:, ::2, 1::2, :].contiguous().view(B, -1, D)
        xs[:, 2] = reshaped_x[:, ::2, 1::2, :].contiguous().view(B, -1, D)
        xs[:, 3] = reshaped_x.transpose(dim0=1, dim1=2)[:, 1::2, 1::2, :].contiguous().view(B, -1, D)

        return xs

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        token_list=[],
        token_list_reversal=[],
        origina_list=[],
        origina_list_reversal=[],
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.token_list = token_list
        self.token_list_reversal = token_list_reversal
        self.origina_list = origina_list
        self.origina_list_reversal = origina_list_reversal

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, scan_type, seqlen=None, seq_idx=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # By Wangzhenbin
        if scan_type == 'spiral':
            xz_list = CrossScan.apply(zxbcdt, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)

            out1 = mamba_split_conv1d_scan_combined(
            xz_list[:,0],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out2 = mamba_split_conv1d_scan_combined(
            xz_list[:,1],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out3 = mamba_split_conv1d_scan_combined(
            xz_list[:,2],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            B_, L_, D_ = out1.shape
            out_m = out1.new_empty((B_, 3, L_, D_))
            out_m[:,0] = out1
            out_m[:,1] = out2
            out_m[:,2] = out3
            out = CrossMerge.apply(out_m, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)  

        elif scan_type == 'zigma':
            xz_ = Zig_CrossScan.apply(zxbcdt, self.token_list, self.origina_list)
            out1 = mamba_split_conv1d_scan_combined(
            xz_,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )
            out = Zig_CrossMerge.apply(out1, self.token_list, self.origina_list)  

        elif scan_type == 'vim':
            out1 = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )
            back_xz =  torch.flip(zxbcdt, [1])
            out2 = mamba_split_conv1d_scan_combined(
            back_xz,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )
            out2 = torch.flip(out2, [1])
            out = (out1 + out2) / 2    

        elif scan_type == 'vmamba':
            xz_list = CrossScan_vmamba.apply(zxbcdt, self.token_list, self.origina_list)

            out1 = mamba_split_conv1d_scan_combined(
            xz_list[:,0],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out2 = mamba_split_conv1d_scan_combined(
            xz_list[:,1],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out3 = mamba_split_conv1d_scan_combined(
            xz_list[:,2],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out4 = mamba_split_conv1d_scan_combined(
            xz_list[:,3],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            B_, L_, D_ = out1.shape
            out_m = out1.new_empty((B_, 4, L_, D_))
            out_m[:,0] = out1
            out_m[:,1] = out2
            out_m[:,2] = out3
            out_m[:,3] = out4
            out = CrossMerge_vmamba.apply(out_m, self.token_list, self.origina_list)  
        
        elif scan_type == 'eff':
            xz_list = CrossScan_EfficientVMamba.apply(zxbcdt)
            out1 = mamba_split_conv1d_scan_combined(
            xz_list[:,0],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out2 = mamba_split_conv1d_scan_combined(
            xz_list[:,1],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out3 = mamba_split_conv1d_scan_combined(
            xz_list[:,2],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            out4 = mamba_split_conv1d_scan_combined(
            xz_list[:,3],
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
            )

            B_, L_, D_ = out1.shape
            out_m = out1.new_empty((B_, 4, L_, D_))
            out_m[:,0] = out1
            out_m[:,1] = out2
            out_m[:,2] = out3
            out_m[:,3] = out4
            out = CrossMerge_EfficientVMamba.apply(out_m, self.token_list, self.origina_list)  

        if seqlen_og is not None:
            out = rearrange(out, "b l d -> (b l) d")
        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)
        
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state