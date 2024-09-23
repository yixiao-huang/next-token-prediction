import torch
import torch.nn as nn

class MLayerAttn(nn.Module):
    def __init__(self, input_size, n_layer = 2, identity_W = True, factorize_W = True, identity_V = True, shared_v = True, skip = False, output_mode = 'single', layer_norm = False, avg_norm = True, init = 'random', regress = False, dim_m= None, dim_t = None):
        super(MLayerAttn, self).__init__() 
        self.skip = skip
        self.n_layer = n_layer
        if dim_m is None:
            dim_m = input_size
        
        if identity_W:
            self.qlist = nn.ModuleList([nn.Identity() for _ in range(n_layer)])
            self.klist = nn.ModuleList([nn.Identity() for _ in range(n_layer)])
        elif factorize_W:
            self.qlist = nn.ModuleList([nn.Linear(input_size, dim_m, bias=False) for _ in range(n_layer)])
            self.klist = nn.ModuleList([nn.Linear(input_size, dim_m, bias=False) for _ in range(n_layer)])
        else:
            self.qklist = nn.ModuleList([nn.Linear(input_size, input_size, bias=False) for _ in range(n_layer)])
        
        self.factorize_W = factorize_W
        self.vlist = nn.ModuleList([nn.Identity() for _ in range(n_layer)])
        self.shared_v = shared_v

        if identity_V:
            self.wlist = nn.ParameterList([nn.Parameter(torch.eye(input_size)) for _ in range(n_layer - 1)])
            # self.w1 = nn.Parameter(torch.eye(input_size) * 0.01)
        else:
            # self.w1 = nn.Parameter(torch.randn(input_size, input_size) * 0.01)
            self.wlist = nn.ParameterList([nn.Parameter(torch.randn(input_size, input_size) * 0.01) for _ in range(n_layer - 1)])
        
        # if self.shared_v:
        # Final layer
        self.regress = regress
        if not regress:
            self.wn = nn.Parameter(torch.randn(input_size) * 0.01)
        # else:
        #     self.w2 = nn.Parameter(torch.randn(input_size, input_size) * 0.01)

        self.attn = [] # attention of each layer: [n, T, T] if n_layer < n else [n, 1, T]
        self.out = [] # output of each layer
        self.prev_out = [] # output of each layer

        # Only for self-attention
        if output_mode == 'single':
            self.output1 = True
        elif output_mode == 'sum':
            self.output1 = False
        else:
            raise NotImplementedError("Output mode {} not implemented".format(output_mode))
        
        for i in range(n_layer):
            self.attn.append([])
            self.out.append([]) 
            self.prev_out.append([])
        
        if layer_norm:
            if avg_norm:
                self.ln = None
            else:
                self.ln = nn.LayerNorm([dim_t, input_size], elementwise_affine = True)
            self.norm = True
        else:
            self.norm = False
        if not init == 'default':
            for qki in self.qklist:
                self._init_weights(qki, init)
            
    def _init_weights(self, module, choice = 'zero'):
        if choice == 'zero':
            if isinstance(module, nn.Linear):
                module.weight.data.zero_()
                if module.bias is not None:
                    module.bias.data.zero_()
        elif choice == 'random':
            if isinstance(module, nn.Linear):
                stdv = 1. / math.sqrt(module.weight.size(1)) * torch.rand(1).item() * 10
                module.weight.data.uniform_(-stdv, stdv)
                if module.bias is not None:
                    module.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_seq, cross_attn = None, disp = False): # [n, T, d], [m, d] -> []

        # Layer 1: Z = \mathbb{S}(XQK^TX^T)XV 
        prev_out = input_seq


        for i in range(self.n_layer):
            self.prev_out[i] = prev_out
            # SA block 
            if self.factorize_W:
                # if i == self.n_layer - 1:
                #     Qi = self.qlist[i](prev_out[:, -1].unsqueeze(1)) # [n, T, d] -> [n, T, m]
                #     raise NotImplementedError("Under construction")
                #     # if cross_attn is not None:
                #     #     Qi = self.qlist[i](cross_attn) # [m, d] -> [m, d]
                #     # else:
                #     #     Qi = self.qlist[i](prev_out) # [n, T, d] -> [n, T, m]
                # else:
                Qi = self.qlist[i](prev_out)
                Ki = self.klist[i](prev_out) # [n, T, d] -> [n, T, m]
                Ai = torch.softmax(Qi @ Ki.transpose(-2, -1), dim=-1) # -> [n, T, T]
            else:
                # if i == self.n_layer - 1:
                #     QKi = self.qklist[i](prev_out[:, -1].unsqueeze(1))
                #     raise NotImplementedError("Under construction")
                # else:
                if i == self.n_layer - 1 and cross_attn is not None:
                    QKi = self.qklist[i](cross_attn)
                else:
                    QKi = self.qklist[i](prev_out)
                    
                Ai = torch.softmax(QKi @ prev_out.transpose(-2, -1), dim=-1) # -> [n, T, T]
                # if cross_attn is not None:
                #     print(Ai.shape)
                #     assert Ai.shape == (input_seq.shape[0], 1, input_seq.shape[1])
            Vi = self.vlist[i](prev_out) # [n, T, d] -> [n, T, d]
            self.attn[i] = Ai
            
            out = Ai @ Vi  # [n, T, T] \times [n, T, d] -> [n, T, d]
            self.out[i] = out

            if self.skip:
                out = out + prev_out
            if self.norm:
                if self.ln is not None:
                    out = self.ln(out)
                else:
                    out = out / 2
            if disp:
                print("Layer ", i)
                print(out[0])
                print(input_seq[0] + (2**(i+1) - 1)*input_seq[0, 0])
            # MLP/FFN Block
            if i < self.n_layer - 1: 
                prev_out = out @ self.wlist[i] 
                # if self.skip:
                #     prev_out = prev_out + out 
                # if self.norm is not None:
                #     # prev_out = self.norm(prev_out)
                #     prev_out = prev_out / 2 

            else:
                if cross_attn is not None:
                    if self.regress:
                        return out 
                    else:
                        assert (out @ self.wn).shape == (input_seq.shape[0], 1)
                        # return out @ self.wn # this may lead to errors
                        return out @ self.wn
                else:
                    if self.output1:
                        return (out @ self.wn)[:,0]
                    else:
                        return torch.sum(out @ self.wn, dim = -1) # [n, T, d] \times [d] -> [n, T]
