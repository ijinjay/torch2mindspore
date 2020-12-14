from torch2mindspore import *

@mindspore_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    # print('+++++++++++', input)
    output = ctx.method_return

    input_ms_tensor = add_missing_ms_tensors(ctx.network, [input])[0]

    relu = ms.nn.ReLU()
    op_key = ctx.network.add_ops(relu)

    ms_out = relu(ctx.network.nodes[input_ms_tensor])

    out_ms_tensor = ctx.network.add_node(ms_out)
    output._ms_tensor = out_ms_tensor

    ctx.network.add_pre(op_key, [input_ms_tensor])
    ctx.network.add_out(op_key, [out_ms_tensor])

