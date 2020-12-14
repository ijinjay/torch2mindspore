from torch2mindspore import *
from module_test import add_module_test


@mindspore_converter('torch.nn.Conv1d.forward')
def convert_Conv1d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    kernel_size = module.kernel_size[0]
    stride = module.stride[0]
    padding = module.padding[0]
    dilation = module.dilation[0]

    kernel = module.weight.detach().cpu().numpy()[..., None]

    has_bias = False
    # bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        has_bias = True
        bias = module.bias.detach().cpu().numpy()

    ms_cell = ms.nn.Conv1d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        pad_mode='pad',
        padding=padding,
        dilation=dilation,
        has_bias=has_bias
        )

    kernel = kernel.reshape((kernel.shape[0], kernel.shape[1], kernel.shape[3], kernel.shape[2]))

    ms_cell.weight = ms.Parameter(ms.Tensor(kernel), name='weight')
    if has_bias:
        ms_cell.bias = ms.Parameter(ms.Tensor(bias), name='bias')

    input_ms_tensor = ctx.network.nodes[input_ms]


    out = ms_cell(input_ms_tensor)

    out_ms = ctx.network.add_node(out)

    output._ms_tensor = out_ms

    op_key = ctx.network.add_ops(ms_cell)
    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)])
def test_Conv1d_basic():
    return torch.nn.Conv1d(10, 5, kernel_size=1, stride=1, padding=0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)])
def test_Conv1d_stride2():
    return torch.nn.Conv1d(10, 5, kernel_size=1, stride=2, padding=0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)])
def test_Conv1d_kernel3():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=2, padding=1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)])
def test_Conv1d_dilation2():
    return torch.nn.Conv1d(10, 5, kernel_size=3, stride=1, padding=1, dilation=2)

