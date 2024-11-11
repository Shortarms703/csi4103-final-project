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
        return self.value == other.value

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


def back_propagation_partial(func: Expression | Variable, var: Variable) -> Expression:
    def derivation_helper(f: Expression | Variable, var: Variable, seed: Constant | Operation | Expression) -> Expression:
        # pseudocode
        # void derive(Expression Z, float seed) {
        #    if isVariable(Z)
        #       partialDerivativeOf(Z) += seed;
        #    else if (Z = A + B)
        #       derive(A, seed);
        #       derive(B, seed);
        #    else if (Z = A - B)
        #       derive(A, seed);
        #       derive(B, -seed);
        #    else if (Z = A * B)
        #       derive(A, valueOf(B) * seed);
        #       derive(B, valueOf(A) * seed);
        # }
        if isinstance(f, Variable):
            if f.name == var.name:
                return Expression(seed)
            else:
                return Expression(Constant(0))
        elif isinstance(f.value, Variable):
            if f.value.name == var.name:
                return Expression(seed)
            else:
                return Expression(Constant(0))
        elif isinstance(f.value, Constant):
            return Expression(Constant(0))
        elif isinstance(f.value, Add):
            # Sum rule, d(f + g)/dx = df/dx + dg/dx
            f = Expression(f.value.left)
            g = Expression(f.value.right)
            return Expression(Add(derivation_helper(f, var, seed).value, derivation_helper(g, var, seed).value))
        elif isinstance(f.value, Multiply):
            # Product rule, d(f * g)/dx = f * dg/dx + g * df/dx
            f = Expression(f.value.left)
            g = Expression(f.value.right)
            f_dg = Multiply(f.value, derivation_helper(g, var, f))
            g_df = Multiply(g.value, derivation_helper(f, var, g))
            return Expression(Add(f_dg, g_df))
        elif isinstance(f.value, Exponent):
            # Power rule with constant exponent, d(f^c)/dx = c * f^(c-1) * df/dx
            f = Expression(f.value.base)
            c = f.value.exponent
            f_c_minus_1 = Exponent(f.value, Constant(c.value - 1))
            return Expression(Multiply(Multiply(c, f_c_minus_1), derivation_helper(f, var, f)))
        elif isinstance(f.value, Sigmoid):
            # Chain rule, d(sigmoid(f))/dx = sigmoid(f) * (1 - sigmoid(f)) * df/dx
            f = Expression(f.value.x)
            sigmoid_f = Sigmoid(f.value)
            one_minus_sigmoid_f = Add(Constant(1), Multiply(Constant(-1), sigmoid_f))
            return Expression(Multiply(Multiply(sigmoid_f, one_minus_sigmoid_f), derivation_helper(f, var, f)))
