import torch
from einops import rearrange
from torch import nn
from timm.models.vision_transformer import Attention, Mlp
from block.mamba import Mamba as ssm
from block.mamba2 import Mamba2 as ssm2

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
#################################################################################
#                                 spiral mamba block                            #
#################################################################################
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
        use_mamba2: bool,
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

        self.norm1 = nn.LayerNorm(D_dim)

        if use_mamba2:
            self.mamba1 = ssm2(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    token_list_reversal=self.token_list_reversal,
                    origina_list = self.origina_list,
                    origina_list_reversal = self.origina_list_reversal,
                    )
            self.mamba2 = ssm2(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    token_list_reversal=self.token_list_reversal,
                    origina_list = self.origina_list,
                    origina_list_reversal = self.origina_list_reversal,
                    )
        else:
            self.mamba1 = ssm(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    token_list_reversal=self.token_list_reversal,
                    origina_list = self.origina_list,
                    origina_list_reversal = self.origina_list_reversal,
                    )
            self.mamba2 = ssm(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    token_list_reversal=self.token_list_reversal,
                    origina_list = self.origina_list,
                    origina_list_reversal = self.origina_list_reversal,
                    )

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
        w_ssm = x_ssm * w

        x_ssm = self.mamba1(x_ssm, 'spiral')
        w_ssm = self.mamba2(w_ssm, 'spiral')

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

#################################################################################
#                                 ZigMa mamba block                             #
#################################################################################
# code reproduction for paper 
# 'ZigMa: Zigzag Mamba Diffusion Model'.   
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
        use_mamba2: bool,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.token_list = token_list
        self.origina_list = origina_list
        self.norm1 = nn.LayerNorm(D_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )

        if use_mamba2:
            self.mamba = ssm2(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    origina_list = self.origina_list,
                    ) 
        else:
            self.mamba = ssm(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    origina_list = self.origina_list,
                    )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)
        x_ssm = self.mamba(x_ssm, 'zigma')
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
        use_mamba2: bool,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        if use_mamba2:
            self.mamba = ssm2(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    )
        else:
            self.mamba = ssm(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    )
        self.norm1 = nn.LayerNorm(D_dim)
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
        x_ssm = self.mamba(x_ssm, 'vim')
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
        
#################################################################################
#                                 VMamba mamba block                            #
#################################################################################
# code reproduction for paper 
# 'VMamba: Visual State Space Model'.   

    
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
        use_mamba2: bool,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.token_list = token_list
        self.origina_list = origina_list

        if use_mamba2:
            self.mamba = ssm2(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    origina_list = self.origina_list,
                    )
        else:
            self.mamba = ssm(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    token_list=self.token_list, 
                    origina_list = self.origina_list,
                    )

        self.norm1 = nn.LayerNorm(D_dim)
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
        x_ssm = self.mamba(x_ssm, 'vmamba')
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

#################################################################################
#                              EfficientVMamba block                            #
#################################################################################
# code reproduction for paper 
# 'EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba'. 
class EfficientVMamba_MambaBlock(nn.Module): 
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        use_mamba2: bool,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.norm1 = nn.LayerNorm(D_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )
        if use_mamba2:
            self.mamba = ssm2(
                d_model=D_dim, 
                d_state=d_state, 
                d_conv=4, 
                expand=2, 
                )
        else:
            self.mamba = ssm(
                    d_model=D_dim, 
                    d_state=d_state, 
                    d_conv=4, 
                    expand=2, 
                    )
        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, w: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape 
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)
        x_ssm = self.mamba(x_ssm, 'eff')
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

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size*2, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, w):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
