from torch2mindspore import *
from module_test import add_module_test


class _MsMul(ms.nn.Cell):
    def __init__(self):
        super(_MsMul, self).__init__()
    def construct(self, x, y):
        return x * y

@mindspore_converter('torch.mul')
@mindspore_converter('torch.Tensor.mul')
@mindspore_converter('torch.Tensor.__imul__')
@mindspore_converter('torch.Tensor.__mul__')
@mindspore_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    input_a_ms, input_b_ms = add_missing_ms_tensors(ctx.network, [input_a, input_b])

    # input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)

    input_a_tensor = ctx.network.nodes[input_a_ms]
    input_b_tensor = ctx.network.nodes[input_b_ms]

    ms_cell = _MsMul()
    out = ms_cell(input_a_tensor, input_b_tensor)

    op_key = ctx.network.add_ops(ms_cell)
    out_ms_tensor = ctx.network.add_node(out)

    output._ms_tensor = out_ms_tensor
    ctx.network.add_pre(op_key, [input_a_ms, input_b_ms])
    ctx.network.add_out(op_key, [out_ms_tensor])

    # layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    # output._trt = layer.get_output(0)

class Mul(torch.nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_basic():
    return Mul()


class IMul(torch.nn.Module):
    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_imul():
    return IMul()


class TorchMul(torch.nn.Module):
    def __init__(self):
        super(TorchMul, self).__init__()

    def forward(self, x, y):
        return torch.mul(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_torchmul():
    return TorchMul()


class RMulInt(torch.nn.Module):
    def __init__(self):
        super(RMulInt, self).__init__()

    def forward(self, x):
        return 10 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_int():
    return RMulInt()


class RMulFloat(torch.nn.Module):
    def __init__(self):
        super(RMulFloat, self).__init__()

    def forward(self, x):
        return 10.0 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_float():
    return RMulFloat()


class MulConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x * self.y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_mul_constant_nobatch():
    return MulConstantNoBatch()


class MulConstantBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x * self.y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_mul_constant_batch():
    return MulConstantBatch()
