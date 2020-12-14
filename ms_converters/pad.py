from torch2mindspore import *
from module_test import add_module_test


@mindspore_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    input = ctx.method_args[0]
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_ms_tensor = ctx.network.nodes[input_ms]

    pad = ctx.method_args[1]
    ms_pad = [(0, 0) for i in range(len(input_ms_tensor.shape))]

    if len(pad) == 2:
        ms_pad[-1] = (pad[0], pad[1])
    elif len(pad) == 4:
        ms_pad[-2] = (pad[2], pad[3])
        ms_pad[-1] = (pad[0], pad[1])
    elif len(pad) == 6:
        ms_pad[-3] = (pad[4], pad[5])
        ms_pad[-2] = (pad[2], pad[3])
        ms_pad[-1] = (pad[0], pad[1])

    ms_pad = tuple(ms_pad)


    mode = ctx.method_args[2] if len(ctx.method_args) > 2 else 'constant'

    if mode == 'reflect':
        mode = 'REFLECT'
    elif mode == 'circular':
        mode = 'SYMMETRIC'
    else:
        mode = 'CONSTANT'

    ms_cell = ms.nn.Pad(paddings=ms_pad, mode=mode)

    out = ms_cell(input_ms_tensor)
    out_ms = ctx.network.add_node(out)
    output._ms_tensor = out_ms

    op_key = ctx.network.add_ops(ms_cell)
    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])



class Pad(torch.nn.Module):

    def __init__(self, pad):
        super(Pad, self).__init__()
        self.pad = pad

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_pad_basic():
    return Pad((1, 2, 3, 4))
