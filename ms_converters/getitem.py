from torch2mindspore import *
from module_test import add_module_test

class _MsGetItem(ms.nn.Cell):
    def __init__(self, a_slice):
        super(_MsGetItem, self).__init__()
        self.a_slice = a_slice
    def construct(self, x):
        return x[self.a_slice]


class _MsExpand(ms.nn.Cell):
    def __init__(self, pos):
        super(_MsExpand, self).__init__()
        self.pos = pos
    def construct(self, x):
        return ms.ops.expand_dims(x, self.pos)


def slice_to_ms(dim_size, dim_slice):

    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step

    size = (stop - start - 1) // stride + 1

    return start, size, stride


def num_slice_types(slices):
    num_slice = 0
    for s in slices:
        if isinstance(s, slice) or isinstance(s, int):
            num_slice += 1
    return num_slice


@mindspore_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx):
    input = ctx.method_args[0]
    slices = ctx.method_args[1]
    output = ctx.method_return

    input_ms = input._ms_tensor
    input_ms_tensor = ctx.network.nodes[input_ms]

    # Step 1 - Replace ellipsis with expanded slices
    num_ellipsis = len(input_ms_tensor.shape) - num_slice_types(slices)

    new_slices = []
    for s in slices:

        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
        elif s is None:
            new_slices.append(None)
        elif isinstance(s, int):
            new_slices.append(s)

    # fill missing slices at end
    while num_slice_types(new_slices) < len(input.shape):
        new_slices.append(slice(None, None, None))

    # print("------------------------------")
    # print(new_slices, type(new_slices))

    if len(new_slices) == len(input_ms_tensor.shape):
        a_slice = tuple(new_slices)
        ms_cell = _MsGetItem(a_slice)
        out = ms_cell(input_ms_tensor)

        op_key = ctx.network.add_ops(ms_cell)
        out_ms = ctx.network.add_node(out)

        output._ms_tensor = out_ms
        ctx.network.add_pre(op_key, [input_ms])
        ctx.network.add_out(op_key, [out_ms])
        return


    # Step 2 - Remove batch from slices (MS from this point)

    slices = tuple(new_slices[1:]) # remove batch


    # Step 4 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices

    num_non_slice = len([s for s in slices if not isinstance(s, slice)])
    if num_non_slice > 0:
        # print("++++++++++++++++++++++++++++++")
        # print(new_slices)
        for i in range(len(new_slices)):
            if new_slices[i] is None:
                ms_cell = _MsExpand(i)
                a_input_tensor = ctx.network.nodes[input_ms]

                out = ms_cell(a_input_tensor)

                op_key = ctx.network.add_ops(ms_cell)
                out_ms = ctx.network.add_node(out)
                ctx.network.add_pre(op_key, [input_ms])
                ctx.network.add_out(op_key, [out_ms])

                input_ms = out_ms
                new_slices[i] = slice(None, None, None)

    a_slice = tuple(new_slices)
    ms_cell = _MsGetItem(a_slice)
    input_ms_tensor = ctx.network.nodes[input_ms]
    out = ms_cell(input_ms_tensor)
    # print("000000000000000000000")
    # print(out.shape)

    op_key = ctx.network.add_ops(ms_cell)
    out_ms = ctx.network.add_node(out)

    output._ms_tensor = out_ms
    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])


class LambdaModule(torch.nn.Module):
    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_tensor_getitem_1d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided():
    return LambdaModule(lambda x: x[:, ::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_offset():
    return LambdaModule(lambda x: x[:, 1::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_range():
    return LambdaModule(lambda x: x[:, 1:3:2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim():
    return LambdaModule(lambda x: x[:, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim_ellipsis():
    return LambdaModule(lambda x: x[:, None, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_dim():
    return LambdaModule(lambda x: x[:, ..., None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_2dim():
    return LambdaModule(lambda x: x[:, ..., None, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_weird_combo():
    return LambdaModule(lambda x: x[:, 0:3:4, None, None, 1, ...])
