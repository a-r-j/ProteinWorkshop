import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, jaxtyped


class ScalarVector(tuple):
    """
    From https://github.com/BioinfoMachineLearning/GCPNet
    """

    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))

    def __getnewargs__(self):
        return self.scalar, self.vector

    @property
    def scalar(self):
        return self[0]

    @property
    def vector(self):
        return self[1]

    # Element-wise addition
    def __add__(self, other):
        if isinstance(other, tuple):
            scalar_other = other[0]
            vector_other = other[1]
        else:
            scalar_other = other.scalar
            vector_other = other.vector

        return ScalarVector(
            self.scalar + scalar_other, self.vector + vector_other
        )

    # Element-wise multiplication or scalar multiplication
    def __mul__(self, other):
        if isinstance(other, tuple):
            other = ScalarVector(other[0], other[1])

        if isinstance(other, ScalarVector):
            return ScalarVector(
                self.scalar * other.scalar, self.vector * other.vector
            )
        else:
            return ScalarVector(self.scalar * other, self.vector * other)

    def concat(self, others, dim: int = -1):
        dim %= len(self.scalar.shape)
        s_args, v_args = list(zip(*(self, *others)))
        return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

    def flatten(self):
        flat_vector = torch.reshape(
            self.vector, self.vector.shape[:-2] + (3 * self.vector.shape[-2],)
        )
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x, vector_dim: int):
        v = torch.reshape(
            x[..., -3 * vector_dim :], x.shape[:-1] + (vector_dim, 3)
        )
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def vs(self):
        return self.scalar, self.vector

    def idx(self, idx):
        return ScalarVector(self.scalar[idx], self.vector[idx])

    def repeat(self, n, c: int = 1, y: int = 1):
        return ScalarVector(
            self.scalar.repeat(n, c), self.vector.repeat(n, y, c)
        )

    def clone(self):
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    @jaxtyped(typechecker=typechecker)
    def mask(self, node_mask: Bool[torch.Tensor, " n_nodes"]):
        return ScalarVector(
            self.scalar * node_mask[:, None],
            self.vector * node_mask[:, None, None],
        )

    def __setitem__(self, key, value):
        self.scalar[key] = value.scalar
        self.vector[key] = value.vector

    def __repr__(self):
        return f"ScalarVector({self.scalar}, {self.vector})"
