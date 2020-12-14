from torch2mindspore import *
from module_test import add_module_test


class _MsAdd(ms.nn.Cell):
    def __init__(self):
        super(_MsAdd, self).__init__()
    def construct(self, x, y):
        z = x + y
        return z



@mindspore_converter('torch.add')
@mindspore_converter('torch.Tensor.__iadd__')
@mindspore_converter('torch.Tensor.__add__')
@mindspore_converter('torch.Tensor.__radd__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    input_a_ms, input_b_ms = add_missing_ms_tensors(ctx.network, [input_a, input_b])
    ms_cell = _MsAdd()
    op_key = ctx.network.add_ops(ms_cell)

    # print('--------------', ctx.network.nodes[input_a_ms].dtype)
    ms_output = ms_cell(ctx.network.nodes[input_a_ms], ctx.network.nodes[input_b_ms])

    out_ms_tensor = ctx.network.add_node(ms_output)

    output._ms_tensor = out_ms_tensor
    ctx.network.add_pre(op_key, [input_a_ms, input_b_ms])
    ctx.network.add_out(op_key, [out_ms_tensor])



class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_basic():
    return Add()


class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_iadd():
    return IAdd()


class TorchAdd(torch.nn.Module):
    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torchadd():
    return TorchAdd()


class RAddInt(torch.nn.Module):
    def __init__(self):
        super(RAddInt, self).__init__()

    def forward(self, x):
        return 1 + x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_int():
    return RAddInt()


class RAddFloat(torch.nn.Module):
    def __init__(self):
        super(RAddFloat, self).__init__()

    def forward(self, x):
        y = 1.0 + x
        y = y + y + 1
        y = y + y + 1
        x = y + x
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_radd_float():
    return RAddFloat()


class AddConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(AddConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x + self.y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_add_constant_nobatch():
    return AddConstantNoBatch()


class AddConstantBatch(torch.nn.Module):
    def __init__(self):
        super(AddConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x + self.y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_add_constant_batch():
    return AddConstantBatch()
