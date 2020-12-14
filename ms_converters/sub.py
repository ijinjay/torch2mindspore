from torch2mindspore import *
from module_test import add_module_test


class _MsSub(ms.nn.Cell):
    def __init__(self):
        super(_MsSub, self).__init__()
    def construct(self, x, y):
        return x - y


@mindspore_converter('torch.sub')
@mindspore_converter('torch.Tensor.__isub__')
@mindspore_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_ms, input_b_ms = add_missing_ms_tensors(ctx.network, [input_a, input_b])

    ms_cell = _MsSub()
    op_key = ctx.network.add_ops(ms_cell)
    out = ms_cell(ctx.network.nodes[input_a_ms], ctx.network.nodes[input_a_ms])

    out_ms = ctx.network.add_node(out)
    output._ms_tensor = out_ms

    ctx.network.add_pre(op_key, [input_a_ms, input_b_ms])
    ctx.network.add_out(op_key, [out_ms])


@mindspore_converter('torch.Tensor.__rsub__')
def convert_sub(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    output = ctx.method_return

    input_a_ms, input_b_ms = add_missing_ms_tensors(ctx.network, [input_a, input_b])

    ms_cell = _MsSub()
    op_key = ctx.network.add_ops(ms_cell)
    out = ms_cell(ctx.network.nodes[input_a_ms], ctx.network.nodes[input_a_ms])

    out_ms = ctx.network.add_node(out)
    output._ms_tensor = out_ms

    ctx.network.add_pre(op_key, [input_a_ms, input_b_ms])
    ctx.network.add_out(op_key, [out_ms])


class Sub(torch.nn.Module):
    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x, y):
        return x - y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_basic():
    return Sub()


class ISub(torch.nn.Module):
    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_isub():
    return ISub()


class TorchSub(torch.nn.Module):
    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_sub():
    return TorchSub()


class RSubInt(torch.nn.Module):
    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_int():
    return RSubInt()


class RSubFloat(torch.nn.Module):
    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_float():
    return RSubFloat()

class SubConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(SubConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x - self.y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_sub_constant_nobatch():
    return SubConstantNoBatch()


class SubConstantBatch(torch.nn.Module):
    def __init__(self):
        super(SubConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x - self.y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_sub_constant_batch():
    return SubConstantBatch()
