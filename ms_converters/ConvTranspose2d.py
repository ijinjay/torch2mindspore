from torch2mindspore import *
from module_test import add_module_test

@mindspore_converter("torch.nn.ConvTranspose2d.forward")
def convert_ConvTranspose2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size,) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride,) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding,) * 4

    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    kernel = module.weight.detach().cpu().numpy()

    has_bias = False
    if module.bias is not None:
        has_bias = True
        bias = module.bias.detach().cpu().numpy()

    ms_cell = ms.nn.Conv2dTranspose(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=padding,
        pad_mode='pad',
        dilation=module.dilation,
        group=module.groups,
        has_bias=has_bias)

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



@add_module_test(torch.float32, torch.device("cuda"), [(1,3,224,224)])
def test_square_kernel_equal_stride_mode():
    return torch.nn.ConvTranspose2d(3,3,3,stride=2)

@add_module_test(torch.float32, torch.device("cuda"), [(1,3,224,224)])
def test_square_kernel_equal_stride_mode_unequal_op_size():
    return torch.nn.ConvTranspose2d(3,6,3,stride=2)

@add_module_test(torch.float32, torch.device("cuda"), [(1,3,224,224)])
def test_unequal_stride_mode():
    return torch.nn.ConvTranspose2d(3,3,3, stride=(2,1), padding=(4,2))

@add_module_test(torch.float32, torch.device("cuda"), [(1,3,112,112)])
@add_module_test(torch.float32, torch.device("cuda"), [(1,3,7,7)])
def test_kernelsize_4():
    return torch.nn.ConvTranspose2d(3,3,4, stride=2, padding=1)

