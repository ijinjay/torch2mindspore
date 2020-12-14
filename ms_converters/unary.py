from torch2mindspore import *
from module_test import add_module_test
        
def __convert_unary(ctx, op):
    input = get_arg(ctx, 'input', pos=0, default=None)
    input_ms = add_missing_ms_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    ms_op = op()

    # get tensor
    input_ms_tensor = ctx.network.nodes[input_ms]

    out = ms_op(input_ms_tensor)
    # add output tensor
    out_ms = ctx.network.add_node(out)

    output._ms_tensor = out_ms

    # add op
    op_key = ctx.network.add_ops(ms_op)
    ctx.network.add_pre(op_key, [input_ms])
    ctx.network.add_out(op_key, [out_ms])

    

class UnaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)
    
# EXP : Exponentiation


@mindspore_converter('torch.exp')
@mindspore_converter('torch.exp_')
@mindspore_converter('torch.Tensor.exp')
@mindspore_converter('torch.Tensor.exp_')
def convert_exp(ctx):
    __convert_unary(ctx, ms.ops.Exp)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_exp():
    return UnaryModule(lambda x: torch.exp(x))


#  LOG : Log (base e)


@mindspore_converter('torch.log')
@mindspore_converter('torch.log_')
@mindspore_converter('torch.Tensor.log')
@mindspore_converter('torch.Tensor.log_')
def convert_log(ctx):
    __convert_unary(ctx, ms.ops.Log)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_log():
    return UnaryModule(lambda x: torch.log(x))


# SQRT : Square root


@mindspore_converter('torch.sqrt')
@mindspore_converter('torch.sqrt_')
@mindspore_converter('torch.Tensor.sqrt')
@mindspore_converter('torch.Tensor.sqrt_')
def convert_sqrt(ctx):
    __convert_unary(ctx, ms.ops.Sqrt)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_sqrt():
    return UnaryModule(lambda x: torch.sqrt(x))


# RECIP : Reciprocal


@mindspore_converter('torch.reciprocal')
@mindspore_converter('torch.reciprocal_')
@mindspore_converter('torch.Tensor.reciprocal')
@mindspore_converter('torch.Tensor.reciprocal_')
def convert_reciprocal(ctx):
    __convert_unary(ctx, ms.ops.Reciprocal)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_reciprocal():
    return UnaryModule(lambda x: torch.reciprocal(x))


# ABS : Absolute value


@mindspore_converter('torch.abs')
@mindspore_converter('torch.abs_')
@mindspore_converter('torch.Tensor.abs')
@mindspore_converter('torch.Tensor.abs_')
def convert_abs(ctx):
    __convert_unary(ctx, ms.ops.Abs)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_abs():
    return UnaryModule(lambda x: torch.abs(x))


#  NEG : Negation

@mindspore_converter('torch.neg')
@mindspore_converter('torch.neg_')
@mindspore_converter('torch.Tensor.neg')
@mindspore_converter('torch.Tensor.__neg__')
@mindspore_converter('torch.Tensor.neg_')
def convert_neg(ctx):
    __convert_unary(ctx, ms.ops.Neg)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_neg():
    return UnaryModule(lambda x: torch.neg(x))


#  SIN : Sine


@mindspore_converter('torch.sin')
@mindspore_converter('torch.sin_')
@mindspore_converter('torch.Tensor.sin')
@mindspore_converter('torch.Tensor.sin_')
def convert_sin(ctx):
    __convert_unary(ctx, ms.ops.Sin)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_sin():
    return UnaryModule(lambda x: torch.sin(x))


#  COS : Cosine


@mindspore_converter('torch.cos')
@mindspore_converter('torch.cos_')
@mindspore_converter('torch.Tensor.cos')
@mindspore_converter('torch.Tensor.cos_')
def convert_cos(ctx):
    __convert_unary(ctx, ms.ops.Cos)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_cos():
    return UnaryModule(lambda x: torch.cos(x))


#  |    TAN : Tangent


# @mindspore_converter('torch.tan')
# @mindspore_converter('torch.tan_')
# @mindspore_converter('torch.Tensor.tan')
# @mindspore_converter('torch.Tensor.tan_')
# def convert_cos(ctx):
#     __convert_unary(ctx, ms.ops.Tan)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_tan():
#     return UnaryModule(lambda x: torch.tan(x))


#  |    SINH : Hyperbolic sine


# @mindspore_converter('torch.sinh')
# @mindspore_converter('torch.sinh_')
# @mindspore_converter('torch.Tensor.sinh')
# @mindspore_converter('torch.Tensor.sinh_')
# def convert_sinh(ctx):
#     __convert_unary(ctx, ms.ops.Sinh)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_sinh():
#     return UnaryModule(lambda x: torch.sinh(x))


#  |    COSH : Hyperbolic cosine


# @mindspore_converter('torch.cosh')
# @mindspore_converter('torch.cosh_')
# @mindspore_converter('torch.Tensor.cosh')
# @mindspore_converter('torch.Tensor.cosh_')
# def convert_cosh(ctx):
#     __convert_unary(ctx, ms.ops.Cosh)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_cosh():
#     return UnaryModule(lambda x: torch.cosh(x))


#  |    ASIN : Inverse sine


# @mindspore_converter('torch.asin')
# @mindspore_converter('torch.asin_')
# @mindspore_converter('torch.Tensor.asin')
# @mindspore_converter('torch.Tensor.asin_')
# def convert_asin(ctx):
#     __convert_unary(ctx, ms.ops.Asin)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_asin():
#     return UnaryModule(lambda x: torch.asin(x))


#  |    ACOS : Inverse cosine


# @mindspore_converter('torch.acos')
# @mindspore_converter('torch.acos_')
# @mindspore_converter('torch.Tensor.acos')
# @mindspore_converter('torch.Tensor.acos_')
# def convert_acos(ctx):
#     __convert_unary(ctx, ms.ops.Acos)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_acos():
#     return UnaryModule(lambda x: torch.acos(x))


#  |    ATAN : Inverse tangent


# @mindspore_converter('torch.atan')
# @mindspore_converter('torch.atan_')
# @mindspore_converter('torch.Tensor.atan')
# @mindspore_converter('torch.Tensor.atan_')
# def convert_atan(ctx):
#     __convert_unary(ctx, ms.ops.Atan)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_atan():
#     return UnaryModule(lambda x: torch.atan(x))


#  |    ASINH : Inverse hyperbolic sine
#  |  
#  |    ACOSH : Inverse hyperbolic cosine
#  |  
#  |    ATANH : Inverse hyperbolic tangent
#  |  

#  CEIL : Ceiling


# @mindspore_converter('torch.ceil')
# @mindspore_converter('torch.ceil_')
# @mindspore_converter('torch.Tensor.ceil')
# @mindspore_converter('torch.Tensor.ceil_')
# def convert_ceil(ctx):
#     __convert_unary(ctx, ms.ops.Ceil)


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
# def test_ceil():
#     return UnaryModule(lambda x: torch.ceil(x))


#  FLOOR : Floor
        

@mindspore_converter('torch.floor')
@mindspore_converter('torch.floor_')
@mindspore_converter('torch.Tensor.floor')
@mindspore_converter('torch.Tensor.floor_')
def convert_floor(ctx):
    __convert_unary(ctx, ms.ops.Floor)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_floor():
    return UnaryModule(lambda x: torch.floor(x))

