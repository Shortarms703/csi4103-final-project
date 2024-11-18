import numpy as np


class Variable:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def evaluate(self, values: dict):
        if self.name in values:
            return values[self.name]
        else:
            raise ValueError(f"Variable {self.name} not found in values")


class Constant:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False

    def __repr__(self):
        return str(self.value)

    def evaluate(self, values: dict):
        return self.value


class Operation:
    def __eq__(self, other):
        return NotImplemented

    def __repr__(self):
        return NotImplemented

    def evaluate(self, values: dict):
        return NotImplemented


class Add(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        if isinstance(other, Add):
            return self.left == other.left and self.right == other.right
        return False

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def evaluate(self, values: dict):
        return self.left.evaluate(values) + self.right.evaluate(values)


class Multiply(Operation):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        """
        This is a strict equality check, does not check for mathematical equivalence
        :param other:
        :return:
        """
        if isinstance(other, Multiply):
            return self.left == other.left and self.right == other.right
        return False

    def __repr__(self):
        return f"({self.left} * {self.right})"

    def evaluate(self, values: dict):
        return self.left.evaluate(values) * self.right.evaluate(values)


class Exponent(Operation):
    def __init__(self, base, exponent):
        self.base = base
        if isinstance(exponent, Constant):
            self.exponent = exponent
        else:
            raise ValueError("Exponent must be a constant, further functionality not implemented")

    def __eq__(self, other):
        return self.base == other.base and self.exponent == other.exponent

    def __repr__(self):
        return f"({self.base}^{self.exponent})"

    def evaluate(self, values: dict):
        return self.base.evaluate(values) ** self.exponent.evaluate(values)


class Sigmoid(Operation):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if isinstance(other, Sigmoid):
            return self.x == other.x
        return False

    def __repr__(self):
        return f"sigmoid({self.x})"

    def evaluate(self, values: dict):
        return 1 / (1 + np.exp(-self.x.evaluate(values)))


class Expression:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return str(self.value)

    def evaluate(self, values: dict):
        return self.value.evaluate(values)


def forward_propagation_partial(func: Expression | Variable, var: Variable) -> Expression:
    if isinstance(func, Variable):
        if func.name == var.name:
            return Expression(Constant(1))
        else:
            return Expression(Constant(0))
    elif isinstance(func.value, Variable):
        if func.value.name == var.name:
            return Expression(Constant(1))
        else:
            return Expression(Constant(0))
    elif isinstance(func.value, Constant):
        return Expression(Constant(0))
    elif isinstance(func.value, Add):
        # Sum rule, d(f + g)/dx = df/dx + dg/dx
        f = Expression(func.value.left)
        g = Expression(func.value.right)
        return Expression(Add(forward_propagation_partial(f, var).value, forward_propagation_partial(g, var).value))
    elif isinstance(func.value, Multiply):
        # Product rule, d(f * g)/dx = f * dg/dx + g * df/dx
        f = Expression(func.value.left)
        g = Expression(func.value.right)
        f_dg = Multiply(f.value, forward_propagation_partial(g, var).value)
        g_df = Multiply(g.value, forward_propagation_partial(f, var).value)
        return Expression(Add(f_dg, g_df))
    elif isinstance(func.value, Exponent):
        # Power rule with constant exponent, d(f^c)/dx = c * f^(c-1) * df/dx
        f = Expression(func.value.base)
        c = func.value.exponent
        f_c_minus_1 = Exponent(f.value, Constant(c.value - 1))
        return Expression(Multiply(Multiply(c, f_c_minus_1), forward_propagation_partial(f, var).value))
    elif isinstance(func.value, Sigmoid):
        # Chain rule, d(sigmoid(f))/dx = sigmoid(f) * (1 - sigmoid(f)) * df/dx
        f = Expression(func.value.x)
        sigmoid_f = Sigmoid(f.value)
        one_minus_sigmoid_f = Add(Constant(1), Multiply(Constant(-1), sigmoid_f))
        return Expression(Multiply(
            Multiply(sigmoid_f, one_minus_sigmoid_f),
            forward_propagation_partial(f, var).value))
    else:
        raise ValueError("Unsupported operation")


class ReverseModeSymbolicEvaluator:
    def __init__(self):
        self.derivatives = {}

    def clear(self):
        self.derivatives = {}

    def get_derivative(self, var: Variable):
        return self.derivatives.get(var.name, Expression(Constant(0)))

    def add_derivative(self, var: Variable, value: Expression):
        if var.name in self.derivatives:
            self.derivatives[var.name] = Expression(
                Add(self.derivatives[var.name].value, value.value)
            )
        else:
            self.derivatives[var.name] = value

    def derive(self, expr: Expression, seed: Expression):
        self.clear()
        self._derive(expr.value, seed)

    def _derive(self, z, seed: Expression):
        if isinstance(z, Constant):
            pass
        elif isinstance(z, Variable):
            self.add_derivative(z, seed)
        elif isinstance(z, Add):
            # Sum rule: d(f + g)/dx = df/dx + dg/dx
            self._derive(z.left, seed)
            self._derive(z.right, seed)
        elif isinstance(z, Multiply):
            # Product rule: d(f * g)/dx = f * dg/dx + g * df/dx
            self._derive(z.left, Expression(Multiply(seed.value, z.right)))
            self._derive(z.right, Expression(Multiply(seed.value, z.left)))
        elif isinstance(z, Exponent):
            # Power rule with constant exponent: d(f^c)/dx = c * f^(c-1) * df/dx
            if not isinstance(z.exponent, Constant):
                raise ValueError("Exponent must be a constant for symbolic reverse mode")
            c = z.exponent
            f = Expression(z.base)
            f_c_minus_1 = Expression(Exponent(z.base, Constant(c.value - 1)))
            self._derive(z.base, Expression(Multiply(seed.value, Multiply(c, f_c_minus_1.value))))
        elif isinstance(z, Sigmoid):
            # Chain rule: d(sigmoid(f))/dx = sigmoid(f) * (1 - sigmoid(f)) * df/dx
            sigmoid_expr = Expression(Sigmoid(z.x))
            one_minus_sigmoid = Expression(Add(Constant(1), Multiply(Constant(-1), sigmoid_expr.value)))
            self._derive(z.x, Expression(Multiply(Multiply(seed.value, sigmoid_expr.value), one_minus_sigmoid.value)))
        else:
            raise ValueError(f"Unsupported operation: {z}")


def back_propagation_partial(func: Expression | Variable) -> ReverseModeSymbolicEvaluator:
    evaluator = ReverseModeSymbolicEvaluator()
    evaluator.derive(func, Expression(Constant(1)))

    return evaluator
