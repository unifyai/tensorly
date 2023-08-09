try:
    import ivy
except ImportError as error:
    message = (
        "Impossible to import Ivy.\n"
        "To use TensorLy with the Ivy backend, "
        "you must first install Ivy!"
    )
    raise ImportError(message) from error



from .core import (
    Backend,
    backend_types,
    backend_array,
)



class ivyBackend(Backend, backend_name="ivy"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
        return ivy.array(data, dtype=dtype)

    @staticmethod
    def lstsq(a, b, rcond="warn"):
        solution = ivy.matmul(
            ivy.pinv(a, rtol=1e-15).astype(ivy.float64), b.astype(ivy.float64)
        )
        residuals = ivy.sum((b - ivy.matmul(a, solution)) ** 2).astype(ivy.float64)
        return (solution, residuals)

    # Array Manipulation
    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return ivy.clip(tensor, a_min, a_max)

    @staticmethod
    def concatenate(tensors, axis=0, out=None):
        return ivy.concat(tensors, axis=axis, out=out)

    @staticmethod
    def copy(tensor, order="k", subok=False):
        return ivy.copy_array(tensor, to_ivy_array=False)





    @staticmethod
    def to_numpy(tensor):
        return ivy.to_numpy(tensor)

    @staticmethod
    def transpose(tensor, axes=None):
        axes = axes or list(range(ivy.get_num_dims(tensor)))[::-1]
        return ivy.permute_dims(tensor, axes)

    @staticmethod
    def sum(tensor, axis=None, dtype=None, keepdims=False, out=None):
        return ivy.sum(tensor, axis=axis, dtype=dtype, keepdims=keepdims, out=out)



    @staticmethod
    def stack(arrays, axis=0):
        return ivy.stack(arrays, axis=axis)

    @staticmethod
    def sort(tensor, axis=-1):
        if axis is None:
            tensor = tensor.flatten()
            axis = -1

        return ivy.sort(tensor, axis=axis, descending=False, stable=True)



    @staticmethod
    def sign(tensor, out=None):
        return ivy.sign(tensor, out=out)

    @staticmethod
    def abs(tensor, out=None):
        return ivy.abs(tensor, out=out)

    @staticmethod
    def mean(tensor, axis=None, keepdims=False):
        if axis is None:
            return ivy.mean(tensor, keepdims=keepdims)
        else:
            return ivy.mean(tensor, axis=axis, keepdims=keepdims)



    @staticmethod
    def argmin(tensor, axis=None, keepdims=False, out=None):
        return ivy.argmin(tensor, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def argmax(tensor, axis=None, keepdims=False, out=None):
        return ivy.argmax(tensor, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def max(tensor, axis=None, keepdims=False, out=None):
        return ivy.max(tensor, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def min(tensor, axis=None, keepdims=False, out=None):
        return ivy.min(tensor, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def conj(tensor, out=None):
        return ivy.conj(tensor, out=out)

    @staticmethod
    def arange(start=0, stop=None, step=1.0, dtype=None):
        return ivy.arange(start, stop=stop, step=step, dtype=dtype)

    @staticmethod
    def moveaxis(array, source, destination):
        return ivy.moveaxis(array, source, destination)

    @staticmethod
    def shape(tensor):
        return tuple(ivy.shape(tensor, as_array=False))

    @staticmethod
    def ndim(tensor):
        return ivy.get_num_dims(tensor)



    @staticmethod
    def dot(a, b):
        if a.ndim > 2 and b.ndim > 2:
            return ivy.tensordot(a, b, axes=([-1], [-2]))
        if not a.ndim or not b.ndim:
            return a * b
        return ivy.matmul(
            a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False
        )

    @staticmethod
    def matmul(a, b):
        return ivy.matmul(
            a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False
        )

    @staticmethod
    def kron(a, b):
        return ivy.kron(a, b)

    @staticmethod
    def solve(a, b):
        return ivy.solve(a, b)

    @staticmethod
    def qr(a, mode="reduced"):
        return ivy.qr(a, mode=mode)

    @staticmethod
    def diag(v, k=0):
        return ivy.diag(v, k=k)

    @staticmethod
    def eigh(tensor):
        return ivy.eigh(tensor)

    @staticmethod
    def is_tensor(tensor):
        return ivy.is_array(tensor)

    @staticmethod
    def argsort(input, axis=None):
        return ivy.argsort(input, axis=axis, descending=False, stable=True)

    @staticmethod
    def log(x):
        return ivy.log(x)

    @staticmethod
    def log2(x):
        return ivy.log2(x)

    @staticmethod
    def finfo(x):
        return ivy.finfo(x)

    @staticmethod
    def log2(x):
        return ivy.log2(x)

    @staticmethod
    def tensordot(a, b, axes=2):
        return ivy.tensordot(a, b, axes=axes)

    @staticmethod
    def logsumexp(input, dim, keepdim=False, *, out=None):
        c = ivy.max(input, axis=dim, keepdims=True)
        if ivy.get_num_dims(c) > 0:
            c = ivy.where(ivy.isinf(c), ivy.zeros_like(c), c)
        elif not ivy.isinf(c):
            c = 0
        exponential = ivy.exp(input - c)
        sum = ivy.sum(exponential, axis=dim, keepdims=keepdim)
        ret = ivy.log(sum)
        if not keepdim:
            c = ivy.squeeze(c, axis=dim)
        ret = ivy.add(ret, c, out=out)
        return ret

    @staticmethod
    def flip(tensor, axis=None):
        if isinstance(axis, int):
            axis = [axis]

        if axis is None:
            return ivy.flip(tensor, axis=[i for i in range(ivy.get_num_dims(tensor))])
        else:
            return ivy.flip(tensor, axis=axis)

    @staticmethod
    def arctanh(x):
        return ivy.atanh(x)

    @staticmethod
    def arcsinh(x):
        return ivy.asinh(x)

    @staticmethod
    def arccosh(x):
        return ivy.acosh(x)

    @staticmethod
    def arctan(x):
        return ivy.atan(x)

    @staticmethod
    def arcsin(x):
        return ivy.asin(x)

    @staticmethod
    def arccos(x):
        return ivy.acos(x)


for name in (
    backend_types
    + backend_array
    + [
        "nan",
        "trace",
    ]
):
    ivyBackend.register_method(name, getattr(ivy, name))


for name in ["svd"]:
    ivyBackend.register_method(name, getattr(ivy.linear_algebra, name))
