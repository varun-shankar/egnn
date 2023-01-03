import torch
from e3nn import o3, nn
from torch.nn import ReLU, Linear

## Helpers ##
class o3GatedLinear(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=ReLU(), **kwargs):
        super(o3GatedLinear, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify()
        self.gate = nn.Gate(irreps_scalars, [act for _, ir in irreps_scalars],
                            irreps_gates, [act for _, ir in irreps_gates], irreps_gated)
        self.lin = o3.Linear(self.irreps_input, self.gate.irreps_in)

    def forward(self, x):

        return self.gate(self.lin(x))

class LinNet(torch.nn.Module):
    def __init__(self, irreps_input, irreps_hidden, irreps_output, equivariant, act=ReLU(), **kwargs):
        super(LinNet, self).__init__()

        if equivariant:
            self.net = torch.nn.Sequential(
                o3GatedLinear(irreps_input, (irreps_hidden).sort().irreps.simplify(), act=act, **kwargs),
                o3.Linear((irreps_hidden).sort().irreps.simplify(), irreps_output)
            )
        else:
            self.net = torch.nn.Sequential(
                Linear(irreps_input.dim, irreps_hidden.dim), act,
                Linear(irreps_hidden.dim, irreps_output.dim)
            )

    def forward(self, x):

        return self.net(x)

class NormLayer(torch.nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
    
    def normalize(self,data):
        if torch.is_tensor(data):
            data = data - data.mean(0, keepdim=True)
            data = data * (data.pow(2).mean(1, keepdim=True) + 1e-12).pow(-0.5)
        else:
            data.hn = data.hn - data.hn.mean(0, keepdim=True)
            data.hn = data.hn * (data.hn.pow(2).mean(1, keepdim=True) + 1e-12).pow(-0.5)
            data.he = data.he - data.he.mean(0, keepdim=True)
            data.he = data.he * (data.he.pow(2).mean(1, keepdim=True) + 1e-12).pow(-0.5)
        return data

    def forward(self, data):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self.normalize(data[i])

            return data
        else:
            return self.normalize(data)
