from torch2mindspore import *
from module_test import add_module_test

class _MsCat(ms.nn.Cell):
    def __init__(self, dim=0):
        super(_MsCat, self).__init__()
        self.cat = ms.ops.Concat(dim)
    def construct(self, *inputs):
        return self.cat(inputs)

@mindspore_converter('torch.cat')
def convert_cat(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=0)

    output = ctx.method_return
    input_mss = add_missing_ms_tensors(ctx.network, inputs)
    # input_mss = broadcast_ms_tensors(ctx.network, input_mss, len(output.shape) - 1)

    input_tensors = [ctx.network.nodes[_] for _ in input_mss]


    ms_cell = _MsCat(dim)

    out = ms_cell(*input_tensors)

    op_key = ctx.network.add_ops(ms_cell)

    out_ms_tensor = ctx.network.add_node(out)

    output._ms_tensor = out_ms_tensor
    ctx.network.add_pre(op_key, input_mss)
    ctx.network.add_out(op_key, [out_ms_tensor])


class Cat(torch.nn.Module):
    def __init__(self, dim):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 4, 4), (1, 4, 4)])
def test_Cat_basic():
    return Cat(0)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4), (1, 4, 4), (1, 17, 4)])
def test_Cat_dim1():
    return Cat(1)
