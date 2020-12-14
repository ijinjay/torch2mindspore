from torch2mindspore import *
from module_test import add_module_test


@mindspore_converter("torch.nn.Conv2d.forward")
def convert_Conv2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    # get_input_name
    input_ms = add_missing_ms_tensors(ctx.network,[input])[0]

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size,) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride,) * 2

    padding = module.padding
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    # if not isinstance(padding, tuple):
        # padding = (padding,) * 2

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation,) * 2

    kernel = module.weight.detach().cpu().numpy()

    # bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    bias = False
    if module.bias is not None:
        bias = True

    ms_cell = ms.nn.Conv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size = module.kernel_size,
        stride=module.stride,
        padding=padding,
        pad_mode='pad',
        dilation=module.dilation,
        group=module.groups,
        has_bias=bias)
    ms_cell.weight = ms.Parameter(ms.Tensor(kernel), name='weight')
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()
        ms_cell.bias = ms.Parameter(ms.Tensor(bias), name='bias')

    # get tensor
    input_ms_tensor = ctx.network.nodes[input_ms]
    # input_ms.set_dtype(ms.float32)
    out = ms_cell(input_ms_tensor)

    # add output tensor
    out_ms_tensor = ctx.network.add_node(out)

    output._ms_tensor = out_ms_tensor

    # add ops
    op_key = ctx.network.add_ops(ms_cell)
    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms_tensor])



@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=True)
def test_Conv2d_basic():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=1, padding=0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=True)
def test_Conv2d_stride2():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=2, padding=0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=True)
def test_Conv2d_kernel3():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=1)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=True)
def test_Conv2d_dilation2():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1, dilation=2)
