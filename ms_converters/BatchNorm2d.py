from torch2mindspore import *
from module_test import add_module_test


@mindspore_converter("torch.nn.BatchNorm2d.forward")
def convert_BatchNorm2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    num_features = module.num_features
    eps = module.eps
    momentum = module.momentum
    affine = module.affine

    gamma = module.weight.detach().cpu().numpy()
    beta = module.bias.detach().cpu().numpy()
    running_mean = module.running_mean.detach().cpu().numpy()
    running_var = module.running_var.detach().cpu().numpy()

    ms_cell = ms.nn.BatchNorm2d(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        gamma_init=ms.Tensor(gamma),
        beta_init=ms.Tensor(beta),
        moving_mean_init=ms.Tensor(running_mean),
        moving_var_init=ms.Tensor(running_var)
    )

    op_key = ctx.network.add_ops(ms_cell)

    input_ms_tensor = ctx.network.nodes[input_ms]
    out = ms_cell(input_ms_tensor)

    out_ms = ctx.network.add_node(out)
    output._ms_tensor = out_ms

    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_batch_norm_2d_trt7():
    return torch.nn.BatchNorm2d(10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_batch_norm_2d_Random():
    x = torch.nn.BatchNorm2d(10)
    x.weight = torch.nn.Parameter(torch.rand(10))
    return x


