# torch2mindspore

参考 [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt),将PyTorch模型转为华为mindspore模型.

## 工作原理

通过装饰器模式，将转换函数(比如convert_Conv2d)绑定到PyTorch的函数调用上(如torch.nn.Conv2d.forward)。通过输入示例数据(input),网络执行过程中，注册函数(torch.nn.Conv2d.forward)执行的同时，相应的转换函数也会执行。转换函数会传递PyTorch函数的参数，此时可以构建mindspore网络。PyTorch函数的输入tensor会添加一个`_ms_tensor`属性，记录Mindspore下的张量。转换器执行过程中，将操作符和操作数均记录下来，转换为mindspore支持的格式。一旦整个网络都执行完成，神经网络操作符按照顺序构建的神经网络会构建出来，最后对应的输出也会返回。

## 目前支持的算子

```python
torch.nn.BatchNorm2d
torch.nn.cat
torch.nn.Conv1d
torch.nn.Conv2d
torch.nn.ConvTranspose2d
torch.nn.functional.pad
torch.nn.ReLU
```

## TODO





