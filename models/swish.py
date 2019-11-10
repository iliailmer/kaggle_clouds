import torch
@torch.jit.script
def swish(x):
    return x*x.sigmoid()


@torch.jit.script
def swish_back(x, grad_output):
    sigmoid = x.sigmoid()
    g = sigmoid*(1. + x*(1.-sigmoid))
    return grad_output*g  # chain rule


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish(x)

    @staticmethod
    def backward(ctx, grad_output):
        return swish_back(x=ctx.saved_variables[0],
                          grad_output=grad_output)


class Swish(torch.nn.Module):
    """Swish Activation Function - PyTorch CUDA Version"""

    def forward(self, inp): return SwishFunction.apply(inp)
