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

    def get_size(self):
        return 1


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

    def get_size(self):
        return 1


class Operation:
    def __eq__(self, other):
        return NotImplemented

    def __repr__(self):
        return NotImplemented

    def evaluate(self, values: dict):
        return NotImplemented

    def get_size(self):
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

    def get_size(self):
        return 1 + self.left.get_size() + self.right.get_size()


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

    def get_size(self):
        return 1 + self.left.get_size() + self.right.get_size()


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

    def get_size(self):
        return 1 + 1 + self.base.get_size()


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

    def get_size(self):
        return 1 + self.x.get_size()


class Expression:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return str(self.value)

    def evaluate(self, values: dict):
        return self.value.evaluate(values)

    def get_size(self):
        return self.value.get_size()


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
        self.reverse_count = 0
        self._derive(expr.value, seed)

    def _derive(self, z, seed: Expression):
        if isinstance(z, Constant):
            pass
        elif isinstance(z, Variable):
            self.reverse_count += 1
            self.add_derivative(z, seed)
        elif isinstance(z, Add):
            self.reverse_count += 3
            # Sum rule: d(f + g)/dx = df/dx + dg/dx
            self._derive(z.left, seed)
            self._derive(z.right, seed)
        elif isinstance(z, Multiply):
            self.reverse_count += 3
            # Product rule: d(f * g)/dx = f * dg/dx + g * df/dx
            self._derive(z.left, Expression(Multiply(seed.value, z.right)))
            self._derive(z.right, Expression(Multiply(seed.value, z.left)))
        elif isinstance(z, Exponent):
            self.reverse_count += 4
            # Power rule with constant exponent: d(f^c)/dx = c * f^(c-1) * df/dx
            if not isinstance(z.exponent, Constant):
                raise ValueError("Exponent must be a constant for symbolic reverse mode")
            c = z.exponent
            f = z.base
            f_c_minus_1 = Expression(Exponent(f, Constant(c.value - 1)))
            self._derive(f, Expression(Multiply(seed.value, Multiply(c, f_c_minus_1.value))))
        elif isinstance(z, Sigmoid):
            self.reverse_count += 4
            # Chain rule: d(sigmoid(f))/dx = sigmoid(f) * (1 - sigmoid(f)) * df/dx
            sigmoid_expr = Expression(Sigmoid(z.x))
            one_minus_sigmoid = Expression(Add(Constant(1), Multiply(Constant(-1), sigmoid_expr.value)))
            self._derive(z.x, Expression(Multiply(Multiply(seed.value, sigmoid_expr.value), one_minus_sigmoid.value)))
        else:
            raise ValueError(f"Unsupported operation: {z}")

    def get_reverse_ad_circuit(self, z: Expression):
        """
        Generate the computational circuit for reverse mode automatic differentiation
        and compute partial derivatives

        Args:
            z (Expression): The expression to differentiate

        Returns:
            dict: Partial derivatives for each variable
        """
        # Reset the circuit and derivatives
        self.circuit = {}
        self.derivatives = {}

        # Topological order storage
        tape = []

        # Perform a topological traversal and circuit generation
        def process_node(node, current_tape_index):
            """
            Recursively process nodes and build the computational circuit

            Args:
                node: Current node being processed
                current_tape_index: Current index in the computational tape

            Returns:
                int: Next available tape index
            """
            # If the node is already processed, return its existing index
            if id(node) in self.circuit:
                return current_tape_index

            # Process based on node type
            if isinstance(node, Constant):
                # Constants are added to the circuit with their value
                self.circuit[id(node)] = {
                    'type': 'constant',
                    'value': node.value,
                    'tape_index': current_tape_index
                }
                tape.append(node)
                return current_tape_index + 1

            elif isinstance(node, Variable):
                # Variables are added to the circuit with their name
                self.circuit[id(node)] = {
                    'type': 'variable',
                    'name': node.name,
                    'tape_index': current_tape_index
                }
                tape.append(node)
                return current_tape_index + 1

            elif isinstance(node, Add):
                # Process left and right children first
                left_index = process_node(node.left, current_tape_index)
                right_index = process_node(node.right, left_index)

                # Add the addition operation to the circuit
                self.circuit[id(node)] = {
                    'type': 'add',
                    'left': id(node.left),
                    'right': id(node.right),
                    'tape_index': right_index
                }
                tape.append(node)
                return right_index + 1

            elif isinstance(node, Multiply):
                # Process left and right children first
                left_index = process_node(node.left, current_tape_index)
                right_index = process_node(node.right, left_index)

                # Add the multiplication operation to the circuit
                self.circuit[id(node)] = {
                    'type': 'multiply',
                    'left': id(node.left),
                    'right': id(node.right),
                    'tape_index': right_index
                }
                tape.append(node)
                return right_index + 1

            elif isinstance(node, Exponent):
                # Process base first
                base_index = process_node(node.base, current_tape_index)

                # Add the exponentiation operation to the circuit
                self.circuit[id(node)] = {
                    'type': 'exponent',
                    'base': id(node.base),
                    'exponent': node.exponent.value,
                    'tape_index': base_index
                }
                tape.append(node)
                return base_index + 1

            elif isinstance(node, Sigmoid):
                # Process input first
                x_index = process_node(node.x, current_tape_index)

                # Add the sigmoid operation to the circuit
                self.circuit[id(node)] = {
                    'type': 'sigmoid',
                    'input': id(node.x),
                    'tape_index': x_index
                }
                tape.append(node)
                return x_index + 1

            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        # Build the circuit and tape
        process_node(z.value, 0)

        # Backward pass to compute derivatives
        derivatives = {}
        adjoint = {id(z.value): Expression(Constant(1))}

        # Traverse tape in reverse order
        for node in reversed(tape):
            node_id = id(node)

            if node_id not in adjoint:
                continue

            # Derivative computation based on node type
            if isinstance(node, Add):
                # Sum rule: d(f + g)/dx = df/dx + dg/dx
                left_id = id(node.left)
                right_id = id(node.right)

                # Update left child's adjoint
                if left_id in adjoint:
                    adjoint[left_id] = Expression(
                        Add(adjoint[left_id].value, adjoint[node_id].value)
                    )
                else:
                    adjoint[left_id] = adjoint[node_id]

                # Update right child's adjoint
                if right_id in adjoint:
                    adjoint[right_id] = Expression(
                        Add(adjoint[right_id].value, adjoint[node_id].value)
                    )
                else:
                    adjoint[right_id] = adjoint[node_id]

            elif isinstance(node, Multiply):
                # Product rule: d(f * g)/dx = f * dg/dx + g * df/dx
                left_id = id(node.left)
                right_id = id(node.right)

                # Update left child's adjoint
                left_grad = Expression(Multiply(adjoint[node_id].value, node.right))
                if left_id in adjoint:
                    adjoint[left_id] = Expression(
                        Add(adjoint[left_id].value, left_grad.value)
                    )
                else:
                    adjoint[left_id] = left_grad

                # Update right child's adjoint
                right_grad = Expression(Multiply(adjoint[node_id].value, node.left))
                if right_id in adjoint:
                    adjoint[right_id] = Expression(
                        Add(adjoint[right_id].value, right_grad.value)
                    )
                else:
                    adjoint[right_id] = right_grad

            elif isinstance(node, Variable):
                # Accumulate derivatives for variables
                if node.name not in derivatives:
                    derivatives[node.name] = adjoint[node_id]
                else:
                    derivatives[node.name] = Expression(
                        Add(derivatives[node.name].value, adjoint[node_id].value)
                    )

        self.derivatives = derivatives
        return derivatives

def back_propagation_partial(func: Expression | Variable) -> ReverseModeSymbolicEvaluator:
    evaluator = ReverseModeSymbolicEvaluator()
    evaluator.derive(func, Expression(Constant(1)))

    return evaluator


if __name__ == '__main__':
    evaluator = ReverseModeSymbolicEvaluator()
    x = Variable("x")
    y = Variable("y")
    z = Expression(Multiply(Add(x, y), y))
    evaluator.derive(z, Expression(Constant(1)))
    print("Partial Derivatives:")
    for var, deriv in evaluator.derivatives.items():
        print(f"d/d{var}: {deriv}")
    print(evaluator.reverse_count)

