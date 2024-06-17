

import torch
from einops import rearrange
from torch import nn
from zeta.nn import SSM
from timm.models.vision_transformer import Attention, Mlp
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                 spiral mamba block                            #
#################################################################################
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
        xs[:, 1] = x[:, order]
        xs[:, 2] = x[:, order_reversal]
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        order = ctx.order
        order_reversal = ctx.order_reversal
        origina = ctx.origina
        origina_reversal = ctx.origina_reversal
        y = ys[:, 0]
        y = y + ys[:, 1][:, origina]
        y = y + ys[:, 2][:, origina_reversal]
        return y, None, None, None, None

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, order: list, order_reversal: list, origina: list, origina_reversal: list):
        ctx.order = order
        ctx.order_reversal = order_reversal
        ctx.origina = origina
        ctx.origina_reversal = origina_reversal
        y = ys[:, 0]
        y = y + ys[:, 1][:, origina]
        y = y + ys[:, 2][:, origina_reversal]
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
        xs[:, 1] = x[:, order]
        xs[:, 2] = x[:, order_reversal]
        return xs, None, None, None, None


class Spiral_MambaBlock(nn.Module):  #ours
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        token_list: list,
        token_list_reversal: list,
        origina_list: list,
        origina_list_reversal: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.token_list = token_list
        self.token_list_reversal = token_list_reversal
        self.origina_list = origina_list
        self.origina_list_reversal = origina_list_reversal

        self.forward_conv1d = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )
        self.w_forward_conv1d = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(D_dim)
        self.norm2 = nn.LayerNorm(D_dim)
        self.norm3 = nn.LayerNorm(D_dim)

        self.activation = nn.SiLU()

        self.forward_ssm_1 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_2 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_3 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_4 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_5 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_6 = SSM(E_dim, dt_rank, dim_inner, d_state)

        self.proj_x = nn.Linear(D_dim, E_dim)
        self.proj_z = nn.Linear(D_dim, E_dim)
        self.proj_x_w = nn.Linear(D_dim, E_dim)
        self.proj_z_w = nn.Linear(D_dim, E_dim)
        self.proj_final = nn.Linear(E_dim, D_dim)
        self.proj_final_w = nn.Linear(E_dim, D_dim)

        self.adaLN_modulation = nn.Sequential(
            # nn.LayerNorm(D_dim*2),
            nn.SiLU(),
            nn.Linear(D_dim*2, D_dim*3, bias=True),
        )


        self.attention_network = nn.Sequential(
            nn.LayerNorm(2*D_dim),
            nn.Linear(2*D_dim, D_dim, bias=True),
            nn.SiLU(),
            nn.Linear(D_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)

        # Inner mamba
        skip = x_ssm
        w_ssm = x_ssm * w
        w_skip = w_ssm

        x_ssm = self.norm2(x_ssm)
        w_ssm = self.norm3(w_ssm)

        # Branch one
        mamba_z = self.proj_z(x_ssm)
        mamba_x = self.proj_x(x_ssm) 
        mamba_z = rearrange(mamba_z, "b s d -> b d s")  
        mamba_z = self.forward_conv1d(mamba_z)
        mamba_z = rearrange(mamba_z, "b d s -> b s d") 
        mamba_z_list = CrossScan.apply(mamba_z, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)
        mamba_z_list[:, 0] = self.forward_ssm_1(mamba_z_list[:, 0].clone())
        mamba_z_list[:, 1] = self.forward_ssm_2(mamba_z_list[:, 1].clone())  
        mamba_z_list[:, 2] = self.forward_ssm_3(mamba_z_list[:, 2].clone())   
        mamba_z = CrossMerge.apply(mamba_z_list, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)
        x_ssm = self.activation(mamba_x)  * mamba_z
        x_ssm = self.proj_final(x_ssm) + skip   # residual

        # Branch two
        w_mamba_z = self.proj_z_w(w_ssm)
        w_mamba_x = self.proj_x_w(w_ssm) 
        w_mamba_z = rearrange(w_mamba_z, "b s d -> b d s")  
        w_mamba_z = self.w_forward_conv1d(w_mamba_z)
        w_mamba_z = rearrange(w_mamba_z, "b d s -> b s d") 
        w_mamba_z_list = CrossScan.apply(w_mamba_z, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)
        w_mamba_z_list[:, 0] = self.forward_ssm_4(w_mamba_z_list[:, 0].clone())
        w_mamba_z_list[:, 1] = self.forward_ssm_5(w_mamba_z_list[:, 1].clone())  
        w_mamba_z_list[:, 2] = self.forward_ssm_6(w_mamba_z_list[:, 2].clone())   
        w_mamba_z = CrossMerge.apply(w_mamba_z_list, self.token_list, self.token_list_reversal, self.origina_list, self.origina_list_reversal)
        w_ssm = self.activation(w_mamba_x) * w_mamba_z
        w_ssm = self.proj_final_w(w_ssm) + w_skip   # residual

        # Spatial attention
        combined_ssm = torch.cat([x_ssm, w_ssm], dim=-1)
        attention_weights = self.attention_network(combined_ssm) 
        x_ssm = attention_weights * x_ssm + (1 - attention_weights) * w_ssm

        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x
    
    
    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize like nn.Linear:
        nn.init.constant_(self.attention_network[1].weight, 0)
        nn.init.constant_(self.attention_network[1].bias, 0)
        nn.init.constant_(self.attention_network[3].weight, 0)
        nn.init.constant_(self.attention_network[3].bias, 0)

        # nn.init.kaiming_normal_(self.attention_network[1].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.attention_network[3].weight, mode='fan_in', nonlinearity='relu')

        nn.init.constant_(self.proj_x.weight, 0)
        nn.init.constant_(self.proj_x.bias, 0)
        nn.init.constant_(self.proj_z.weight, 0)
        nn.init.constant_(self.proj_z.bias, 0)
        nn.init.constant_(self.proj_x_w.weight, 0)
        nn.init.constant_(self.proj_x_w.bias, 0)
        nn.init.constant_(self.proj_z_w.weight, 0)
        nn.init.constant_(self.proj_z_w.bias, 0)
        nn.init.constant_(self.proj_final.weight, 0)
        nn.init.constant_(self.proj_final.bias, 0)
        nn.init.constant_(self.proj_final_w.weight, 0)
        nn.init.constant_(self.proj_final_w.bias, 0)



#################################################################################
#                                 ZigMa mamba block                             #
#################################################################################
# code reproduction for paper 
# 'ZigMa: Zigzag Mamba Diffusion Model'. 
class Zig_CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, order: list, origina: list): 
        B, C, D = x.shape
        ctx.order = order
        ctx.origina = origina
        ctx.shape = (B, C, D)
        xs = x[:, order]
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        order = ctx.order
        origina = ctx.origina
        y = ys[:, origina]
        return y, None, None

class Zig_CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, order: list, origina: list):
        ctx.order = order
        ctx.origina = origina
        y = ys[:, origina]
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        order = ctx.order
        origina = ctx.origina
        B, C, D = x.shape
        xs = x[:, order]
        return xs, None, None
    
class Zig_MambaBlock(nn.Module): 
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        token_list: list,
        origina_list: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.token_list = token_list
        self.origina_list = origina_list

        self.forward_conv1d = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(D_dim)
        self.norm2 = nn.LayerNorm(D_dim)

        self.activation = nn.SiLU() 

        self.forward_ssm = SSM(E_dim, dt_rank, dim_inner, d_state)

        self.proj_x = nn.Linear(D_dim, E_dim)
        self.proj_z = nn.Linear(D_dim, E_dim)
        self.proj_final = nn.Linear(E_dim, D_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)

        # Inner mamba
        skip = x_ssm

        x_ssm = self.norm2(x_ssm)

        # Branch one
        mamba_z = self.proj_z(x_ssm)
        mamba_x = self.proj_x(x_ssm) 
        mamba_z = rearrange(mamba_z, "b s d -> b d s")  
        mamba_z = self.forward_conv1d(mamba_z)
        mamba_z = rearrange(mamba_z, "b d s -> b s d") 
        re_mamba_z = Zig_CrossScan.apply(mamba_z, self.token_list, self.origina_list)
        re_mamba_z = self.forward_ssm(re_mamba_z)
        mamba_z = Zig_CrossMerge.apply(re_mamba_z, self.token_list, self.origina_list)
        x_ssm = self.activation(mamba_x)  * mamba_z     # it's add in the paper, that seems like a typo. We use Hadamard product instead.
        x_ssm = self.proj_final(x_ssm) + skip           # residual

        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x
    
    
    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize like nn.Linear:
        nn.init.constant_(self.proj_x.weight, 0)
        nn.init.constant_(self.proj_x.bias, 0)
        nn.init.constant_(self.proj_z.weight, 0)
        nn.init.constant_(self.proj_z.bias, 0)
        nn.init.constant_(self.proj_final.weight, 0)
        nn.init.constant_(self.proj_final.bias, 0)


#################################################################################
#                                 ViM mamba block                               #
#################################################################################
# code reproduction for paper 
# 'Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model'.   
class ViM_MambaBlock(nn.Module): 
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d_1 = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )
        self.forward_conv1d_2 = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(D_dim)
        self.norm2 = nn.LayerNorm(D_dim)

        self.activation = nn.SiLU()

        self.forward_ssm_1 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_2 = SSM(E_dim, dt_rank, dim_inner, d_state)

        self.proj_x = nn.Linear(D_dim, E_dim)
        self.proj_z = nn.Linear(D_dim, E_dim)

        self.proj_final = nn.Linear(E_dim, D_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)

        # Inner mamba
        skip = x_ssm
        x_ssm = self.norm2(x_ssm)
        mamba_z = self.proj_z(x_ssm)
        mamba_x = self.proj_x(x_ssm)

        backward_mamba_z = mamba_z

        # Branch one
        mamba_z = rearrange(mamba_z, "b s d -> b d s")  
        mamba_z = self.forward_conv1d_1(mamba_z)
        mamba_z = rearrange(mamba_z, "b d s -> b s d") 
        mamba_z = self.forward_ssm_1(mamba_z)       
        x_ssm_1 = self.activation(mamba_x)  * mamba_z

        # Branch two 
        backward_mamba_z = rearrange(backward_mamba_z, "b s d -> b d s")  
        backward_mamba_z = self.forward_conv1d_2(backward_mamba_z)
        backward_mamba_z = rearrange(backward_mamba_z, "b d s -> b s d") 
        backward_mamba_z = torch.flip(backward_mamba_z, [1])
        backward_mamba_z = self.forward_ssm_2(backward_mamba_z)
        backward_mamba_z = torch.flip(backward_mamba_z, [1])
        x_ssm_2 = self.activation(mamba_x)  * backward_mamba_z

        x_ssm = x_ssm_1 + x_ssm_2
        x_ssm = self.proj_final(x_ssm) + skip   # residual

        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x
    
    
    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize like nn.Linear:
        nn.init.constant_(self.proj_x.weight, 0)
        nn.init.constant_(self.proj_x.bias, 0)
        nn.init.constant_(self.proj_z.weight, 0)
        nn.init.constant_(self.proj_z.bias, 0)
        nn.init.constant_(self.proj_final.weight, 0)
        nn.init.constant_(self.proj_final.bias, 0)



#################################################################################
#                                 VMamba mamba block                            #
#################################################################################
# code reproduction for paper 
# 'VMamba: Visual State Space Model'.   
class CrossScan_vmamba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, order: list, origina: list): 
        B, C, D = x.shape
        ctx.order = order
        ctx.origina = origina
        ctx.shape = (B, C, D)
        xs = x.new_empty((B, 4, C, D))
        xs[:, 0] = x[:, order[0]]
        xs[:, 1] = x[:, order[1]]
        xs[:, 2] = x[:, order[2]]
        xs[:, 3] = x[:, order[3]]
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, D = ctx.shape
        order = ctx.order
        origina = ctx.origina
        y = ys[:, 0][:, origina[0]]
        y = y + ys[:, 1][:, origina[1]]
        y = y + ys[:, 2][:, origina[2]]
        y = y + ys[:, 3][:, origina[3]]
        return y/4.0, None, None

class CrossMerge_vmamba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, order: list, origina: list):
        ctx.order = order
        ctx.origina = origina
        y = ys[:, 0][:, origina[0]]
        y = y + ys[:, 1][:, origina[1]]
        y = y + ys[:, 2][:, origina[2]]
        y = y + ys[:, 3][:, origina[3]]
        return y/4.0
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        order = ctx.order
        origina = ctx.origina
        B, C, D = x.shape
        xs = x.new_empty((B, 4, C, D))
        xs[:, 0] = x[:, order[0]]
        xs[:, 1] = x[:, order[1]]
        xs[:, 2] = x[:, order[2]]
        xs[:, 3] = x[:, order[3]]
        return xs, None, None
    
class VMamba_MambaBlock(nn.Module): 
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        token_list: list,
        origina_list: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.token_list = token_list
        self.origina_list = origina_list

        self.forward_conv1d = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(D_dim)
        self.norm2 = nn.LayerNorm(D_dim)

        self.activation = nn.SiLU()

        self.forward_ssm_1 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_2 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_3 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_4 = SSM(E_dim, dt_rank, dim_inner, d_state)

        self.proj_x = nn.Linear(D_dim, E_dim)
        self.proj_z = nn.Linear(D_dim, E_dim)
        
        self.proj_final = nn.Linear(E_dim, D_dim)


        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)

        # Inner mamba
        skip = x_ssm
        x_ssm = self.norm2(x_ssm)

        # Branch one
        mamba_z = self.proj_z(x_ssm)
        mamba_x = self.proj_x(x_ssm) 
        mamba_z = rearrange(mamba_z, "b s d -> b d s")  
        mamba_z = self.forward_conv1d(mamba_z)
        mamba_z = rearrange(mamba_z, "b d s -> b s d") 

        mamba_z_list = CrossScan_vmamba.apply(mamba_z, self.token_list, self.origina_list)

        mamba_z_list[:, 0] = self.forward_ssm_1(mamba_z_list[:, 0].clone())
        mamba_z_list[:, 1] = self.forward_ssm_2(mamba_z_list[:, 1].clone())  
        mamba_z_list[:, 2] = self.forward_ssm_3(mamba_z_list[:, 2].clone())   
        mamba_z_list[:, 3] = self.forward_ssm_4(mamba_z_list[:, 3].clone())   

        mamba_z = CrossMerge_vmamba.apply(mamba_z_list, self.token_list, self.origina_list)

        x_ssm = self.activation(mamba_x) * mamba_z  
        x_ssm = self.proj_final(x_ssm) + skip   # residual

        x = x + gate_msa.unsqueeze(1) * x_ssm

        return x
    
    
    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize like nn.Linear:
        nn.init.constant_(self.proj_x.weight, 0)
        nn.init.constant_(self.proj_x.bias, 0)
        nn.init.constant_(self.proj_z.weight, 0)
        nn.init.constant_(self.proj_z.bias, 0)
        nn.init.constant_(self.proj_final.weight, 0)
        nn.init.constant_(self.proj_final.bias, 0)



#################################################################################
#                              EfficientVMamba block                            #
#################################################################################
# code reproduction for paper 
# 'EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba'. 
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


class EfficientVMamba_MambaBlock(nn.Module): 
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=E_dim, out_channels=E_dim, kernel_size=1
        )

        self.norm1 = nn.LayerNorm(D_dim)
        self.norm2 = nn.LayerNorm(D_dim)

        self.activation = nn.SiLU()

        self.forward_ssm_1 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_2 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_3 = SSM(E_dim, dt_rank, dim_inner, d_state)
        self.forward_ssm_4 = SSM(E_dim, dt_rank, dim_inner, d_state)

        self.proj_x = nn.Linear(D_dim, E_dim)
        self.proj_z = nn.Linear(D_dim, E_dim)
        
        self.proj_final = nn.Linear(E_dim, D_dim)


        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)

        # Inner mamba
        skip = x_ssm
        x_ssm = self.norm2(x_ssm)

        # Branch one
        mamba_z = self.proj_z(x_ssm)
        mamba_x = self.proj_x(x_ssm) 
        mamba_z = rearrange(mamba_z, "b s d -> b d s")  
        mamba_z = self.forward_conv1d(mamba_z)
        mamba_z = rearrange(mamba_z, "b d s -> b s d") 

        mamba_z_list = CrossScan_EfficientVMamba.apply(mamba_z)

        mamba_z_list[:, 0] = self.forward_ssm_1(mamba_z_list[:, 0].clone())
        mamba_z_list[:, 1] = self.forward_ssm_2(mamba_z_list[:, 1].clone())  
        mamba_z_list[:, 2] = self.forward_ssm_3(mamba_z_list[:, 2].clone())   
        mamba_z_list[:, 3] = self.forward_ssm_4(mamba_z_list[:, 3].clone())   

        mamba_z = CrossMerge_EfficientVMamba.apply(mamba_z_list)

        x_ssm = self.activation(mamba_x) * mamba_z  
        x_ssm = self.proj_final(x_ssm) + skip   # residual

        x = x + gate_msa.unsqueeze(1) * x_ssm

        return x
    
    
    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize like nn.Linear:
        nn.init.constant_(self.proj_x.weight, 0)
        nn.init.constant_(self.proj_x.bias, 0)
        nn.init.constant_(self.proj_z.weight, 0)
        nn.init.constant_(self.proj_z.bias, 0)
        nn.init.constant_(self.proj_final.weight, 0)
        nn.init.constant_(self.proj_final.bias, 0)
