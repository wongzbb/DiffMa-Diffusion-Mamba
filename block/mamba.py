# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified by Wangzhenbin in 2024

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# @torch.compile
def scan_permutation(xz_main, _perm):
    return xz_main[:, :, _perm].contiguous()  # [B,C,T]

def merge_permutation(o_main, _perm_rev):
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

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
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
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.token_list = token_list
        self.token_list_reversal = token_list_reversal
        self.origina_list = origina_list
        self.origina_list_reversal = origina_list_reversal

        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, scan_type, inference_params=None):
        """
        hidden_states: (B, (F H W), D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # by Wangzhenbin
        if scan_type == 'spiral':
            xz_list = CrossScan.apply(xz, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)

            out1 = mamba_inner_fn(xz_list[:,0], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out2 = mamba_inner_fn(xz_list[:,1], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out3 = mamba_inner_fn(xz_list[:,2], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)

            B_, L_, D_ = out1.shape
            out_m = out1.new_empty((B_, 3, L_, D_))
            out_m[:,0] = out1
            out_m[:,1] = out2
            out_m[:,2] = out3
            out = CrossMerge.apply(out_m, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)  

        elif scan_type == 'zigma':
            xz_ = Zig_CrossScan.apply(xz, self.token_list, self.origina_list)
            out1 = mamba_inner_fn(xz_, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out = Zig_CrossMerge.apply(out1, self.token_list, self.origina_list)  

        elif scan_type == 'vim':
            out1 = mamba_inner_fn(xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            back_xz =  torch.flip(xz, [2])
            out2 = mamba_inner_fn(back_xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out2 = torch.flip(out2, [2])
            out = (out1 + out2) / 2    

        elif scan_type == 'vmamba':
            xz_list = CrossScan_vmamba.apply(xz, self.token_list, self.origina_list)

            out1 = mamba_inner_fn(xz_list[:,0], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out2 = mamba_inner_fn(xz_list[:,1], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out3 = mamba_inner_fn(xz_list[:,2], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out4 = mamba_inner_fn(xz_list[:,3], self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)

            B_, L_, D_ = out1.shape
            out_m = out1.new_empty((B_, 4, L_, D_))
            out_m[:,0] = out1
            out_m[:,1] = out2
            out_m[:,2] = out3
            out_m[:,3] = out4
            out = CrossMerge_vmamba.apply(out_m, self.token_list, self.origina_list)  
        
        elif scan_type == 'eff':
            B, D, L = xz.shape
            xz = rearrange(xz, 'B D L -> B L D', B=B, D=D, L=L)
            xz_list = CrossScan_EfficientVMamba.apply(xz)
            bb, ll, dd = xz_list[:,0].shape
            out1 = mamba_inner_fn( rearrange(xz_list[:,0], 'b l d -> b d l', b=bb, l=ll, d=dd), self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out2 = mamba_inner_fn( rearrange(xz_list[:,1], 'b l d -> b d l', b=bb, l=ll, d=dd), self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out3 = mamba_inner_fn( rearrange(xz_list[:,2], 'b l d -> b d l', b=bb, l=ll, d=dd), self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)
            out4 = mamba_inner_fn( rearrange(xz_list[:,3], 'b l d -> b d l', b=bb, l=ll, d=dd), self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias, A, None, None, self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,)

            B__, D__, L__ = out1.shape
            out_m = out1.new_empty((B__, 4, D__, L__))
            out_m[:,0] = out1 
            out_m[:,1] = out2 
            out_m[:,2] = out3 
            out_m[:,3] = out4 
            out = CrossMerge_EfficientVMamba.apply(out_m)  

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
