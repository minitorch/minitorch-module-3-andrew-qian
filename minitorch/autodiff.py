from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

from minitorch.operators import (
    zipWith,
    map
)

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    # TODO: Implement for Task 1.1.
    vals_plus = list(vals)
    vals_plus[arg] += epsilon 
    vals_plus = tuple(vals_plus)

    vals_minus = list(vals)
    vals_minus[arg] -= epsilon
    vals_minus = tuple(vals_minus)

    
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)
    


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.

    res = []

    vis = set()
    def build_nodes(curr: Variable) -> None:
        if curr.is_constant():
            return
        
        if curr.unique_id in vis:
            return

        vis.add(curr.unique_id)

        for p in curr.parents:
            build_nodes(p)

        res.append(curr)
    
    build_nodes(variable)
    return list(reversed(res))



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.

    topo_sort = topological_sort(variable)
    derivatives: dict[int, Any] = {}
    derivatives[variable.unique_id] = deriv

    for v in topo_sort:
        back_deriv = derivatives[v.unique_id]
        if (v.is_leaf()):
            v.accumulate_derivative(back_deriv)
            continue

        for parent, d_parent in v.chain_rule(back_deriv):
            if (parent.is_constant()):
                continue
            
            if not parent.unique_id in derivatives:
                derivatives[parent.unique_id] = d_parent
            else:
                derivatives[parent.unique_id] += d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
