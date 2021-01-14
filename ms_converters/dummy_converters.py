from torch2mindspore import *


def is_private(method):
    method = method.split('.')[-1]  # remove prefix
    return method[0] == '_' and method[1] is not '_'

def is_function_type(method):
    fntype =  eval(method + '.__class__.__name__')
    return fntype == 'function' or fntype == 'builtin_function_or_method' or fntype == 'method_descriptor'

def get_methods(namespace):
    methods = []
    for method in dir(eval(namespace)):
        full_method = namespace + '.' + method
        if not is_private(full_method) and is_function_type(full_method):
            methods.append(full_method)
    return methods


TORCH_METHODS = []
TORCH_METHODS += get_methods('torch')
TORCH_METHODS += get_methods('torch.Tensor')
TORCH_METHODS += get_methods('torch.nn.functional')


for method in TORCH_METHODS:

    @mindspore_converter(method, is_real=False)
    def warn_method(ctx):
        print('Warning: Encountered known unsupported method %s' % ctx.method_str)


# @mindspore_converter('torch.Tensor.size', is_real=False)
# @mindspore_converter('torch.Tensor.dim', is_real=False)
# def dont_warn(ctx):
    # pass


class _MsGetSize(ms.nn.Cell):
    def __init__(self):
        super(_MsGetSize, self).__init__()
    def construct(self, x):
        return x.shape

@mindspore_converter('torch.Tensor.size', is_real=True)
def convert_size(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    input_ms_tensor = ctx.network.nodes[input_ms]

    ms_cell = _MsGetSize()
    out = ms_cell(input_ms_tensor)

    op_key = ctx.network.add_ops(ms_cell)
    out_ms = ctx.network.add_node(out)

    # print("-----")
    # print(type(output), type(out))
    # output._ms_tensor = out_ms
    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])


class _MsGetDim(ms.nn.Cell):
    def __init__(self):
        super(_MsGetDim, self).__init__()
    def construct(self, x):
        return x.ndim

@mindspore_converter('torch.Tensor.dim', is_real=True)
def convert_dim(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    input_ms_tensor = ctx.network.nodes[input_ms]

    ms_cell = _MsGetDim()
    out = ms_cell(input_ms_tensor)

    op_key = ctx.network.add_ops(ms_cell)
    out_ms = ctx.network.add_node(out)

    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])

