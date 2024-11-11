from circuits import *
from sympy import symbols, diff, exp


def test_single_variable_partial():
    x = Variable('x')
    f = Expression(x)
    partial_x = forward_propagation_partial(f, x)
    assert isinstance(partial_x.value, Constant)
    assert partial_x.value.value == 1


def test_single_variable_partial2():
    x = Variable('x')
    y = Variable('y')
    f = Expression(x)
    partial_y = forward_propagation_partial(f, y)
    assert isinstance(partial_y.value, Constant)
    assert partial_y.value.value == 0


def test_constant_partial():
    x = Variable('x')
    c = Constant(5)
    f = Expression(c)
    partial_x = forward_propagation_partial(f, x)
    assert isinstance(partial_x.value, Constant)
    assert partial_x.value.value == 0


def test_addition_partial():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Add(x, y))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Add(Constant(1), Constant(0)))

    partial_y = forward_propagation_partial(f, y)
    assert partial_y == Expression(Add(Constant(0), Constant(1)))

    partial_z = forward_propagation_partial(f, Variable('z'))
    assert partial_z == Expression(Add(Constant(0), Constant(0)))


def test_multiplication_partial():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Multiply(x, y))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Add(Multiply(Variable('x'), Constant(0)), Multiply(Variable('y'), Constant(1))))


def test_partial_with_negatives():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Add(x, Multiply(Constant(-1), y)))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(
        Add(Constant(1), Add(Multiply(Constant(-1), Constant(0)), Multiply(Variable('y'), Constant(0)))))

    partial_y = forward_propagation_partial(f, y)
    assert partial_y == Expression(
        Add(Constant(0), Add(Multiply(Constant(-1), Constant(1)), Multiply(Variable('y'), Constant(0)))))


def test_partial_with_negatives_and_coefficients():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Add(Multiply(Constant(2), x), Multiply(Constant(-3), y)))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Add(Add(Multiply(Constant(2), Constant(1)), Multiply(Variable('x'), Constant(0))),
                                       Add(Multiply(Constant(-3), Constant(0)), Multiply(Variable('y'), Constant(0)))))


def test_exponent_partials():
    x = Variable('x')
    f = Expression(Exponent(x, Constant(2)))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Multiply(Multiply(Constant(2), Exponent(Variable('x'), Constant(1))), Constant(1)))

    partial_y = forward_propagation_partial(f, Variable('y'))
    assert partial_y == Expression(Multiply(Multiply(Constant(2), Exponent(Variable('x'), Constant(1))), Constant(0)))

    f = Expression(Exponent(x, Constant(3)))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Multiply(Multiply(Constant(3), Exponent(Variable('x'), Constant(2))), Constant(1)))

    f = Expression(Exponent(x, Constant(0)))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Multiply(Multiply(Constant(0), Exponent(Variable('x'), Constant(-1))), Constant(1)))


def test_exponent_partials_addition_base():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Exponent(Add(x, y), Constant(2)))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(
        Multiply(Multiply(Constant(2), Exponent(Add(Variable('x'), Variable('y')), Constant(1))),
                 Add(Constant(1), Constant(0))))

    partial_y = forward_propagation_partial(f, y)
    assert partial_y == Expression(
        Multiply(Multiply(Constant(2), Exponent(Add(Variable('x'), Variable('y')), Constant(1))),
                 Add(Constant(0), Constant(1))))


def test_sigmoid_partial():
    x = Variable('x')
    f = Expression(Sigmoid(x))
    partial_x = forward_propagation_partial(f, x)
    assert partial_x == Expression(Multiply(
            Multiply(Sigmoid(x), Add(Constant(1), Multiply(Constant(-1), Sigmoid(x)))), Constant(1)))

def test_evaluate_variable():
    x = Variable('x')
    assert x.evaluate({'x': 5}) == 5

    y = Variable('y')
    assert y.evaluate({'y': 0, 'x': 10}) == 0


def test_evaluate_constant():
    c = Constant(5)
    assert c.evaluate({}) == 5

    c = Constant(0)
    assert c.evaluate({"a": 10}) == 0


def test_evaluate_addition():
    x = Variable('x')
    y = Variable('y')
    f = Add(x, y)
    assert f.evaluate({'x': 5, 'y': 10}) == 15

    c = Constant(5)
    f = Add(x, c)
    assert f.evaluate({'x': 5}) == 10

    f = Add(c, c)
    assert f.evaluate({}) == 10


def test_evaluate_multiplication():
    x = Variable('x')
    y = Variable('y')
    f = Multiply(x, y)
    assert f.evaluate({'x': 5, 'y': 10}) == 50

    c = Constant(5)
    f = Multiply(x, c)
    assert f.evaluate({'x': 5}) == 25

    f = Multiply(c, c)
    assert f.evaluate({}) == 25


def test_evaluate_exponent():
    x = Variable('x')
    f = Exponent(x, Constant(2))
    assert f.evaluate({'x': 5}) == 25

    f = Exponent(x, Constant(0))
    assert f.evaluate({'x': 5}) == 1

    f = Exponent(x, Constant(1))
    assert f.evaluate({'x': 5}) == 5


def test_evaluate_sigmoid():
    x = Variable('x')
    f = Sigmoid(x)
    assert f.evaluate({'x': 0}) == 0.5

    f = Sigmoid(Add(x, Constant(1)))
    assert f.evaluate({'x': 0}) == 0.7310585786300049

    f = Sigmoid(Add(x, Constant(-1)))
    assert f.evaluate({'x': 0}) == 0.2689414213699951


def test_evaluate_expression():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Add(x, Multiply(x, y)))
    assert f.evaluate({'x': 5, 'y': 10}) == 5 + 5 * 10

    f = Expression(Multiply(x, Multiply(Constant(-1), y)))
    assert f.evaluate({'x': 5, 'y': 10}) == 5 * -10

    f = Expression(Exponent(Multiply(x, Multiply(Constant(-1), y)), Constant(2)))
    assert f.evaluate({'x': 5, 'y': 10}) == (5 * -10) ** 2


def get_sympy_gradients_evaluated_for_loss(W1_1, W1_2, b1, X1, X2, y, sigmoid=False):
    s_W1_1, s_W1_2, s_b1, s_X1, s_X2, s_y = symbols('W1_1 W1_2 b1 X1 X2 y')

    s_z1 = s_X1 * s_W1_1 + s_X2 * s_W1_2 + s_b1
    if sigmoid:
        s_a1 = 1 / (1 + exp(-s_z1))
    else:
        s_a1 = s_z1

    s_loss = (s_a1 - s_y) ** 2

    # Derivatives of the loss w.r.t weights and bias
    s_dloss_dW1_1 = diff(s_loss, s_W1_1)
    s_dloss_dW1_2 = diff(s_loss, s_W1_2)
    s_dloss_db1 = diff(s_loss, s_b1)
    grad_W1_1 = s_dloss_dW1_1.subs(
        {s_W1_1: W1_1, s_W1_2: W1_2, s_b1: b1, s_X1: X1, s_X2: X2, s_y: y})
    grad_W1_2 = s_dloss_dW1_2.subs(
        {s_W1_1: W1_1, s_W1_2: W1_2, s_b1: b1, s_X1: X1, s_X2: X2, s_y: y})
    grad_b1 = s_dloss_db1.subs(
        {s_W1_1: W1_1, s_W1_2: W1_2, s_b1: b1, s_X1: X1, s_X2: X2, s_y: y})
    return {'W1_1': grad_W1_1, 'W1_2': grad_W1_2, 'b1': grad_b1}


def get_sympy_gradients_evaluated_for_simple_func(x, y):
    s_x = symbols('x')
    s_y = symbols('y')
    s_f = 1 / (1 + exp(-(s_x - s_y)))

    # Derivatives of the loss w.r.t weights and bias
    s_dloss_dx = diff(s_f, s_x)
    s_dloss_dy = diff(s_f, s_y)
    grad_x = s_dloss_dx.subs(
        {s_x: x, s_y: y})
    grad_y = s_dloss_dy.subs(
        {s_x: x, s_y: y})
    return {'x': grad_x, 'y': grad_y}


def test_partial_compared_to_sympy_for_sigmoid():
    x = Variable('x')
    y = Variable('y')
    f = Expression(Sigmoid(Add(x, Multiply(Constant(-1), y))))
    partial_x = forward_propagation_partial(f, x)
    partial_y = forward_propagation_partial(f, y)

    def my_partial_x_evaluated(x, y):
        return partial_x.evaluate({'x': x, 'y': y})

    def my_partial_y_evaluated(x, y):
        return partial_y.evaluate({'x': x, 'y': y})

    # test ranges 0 to 10, -10 to 10, 2^0 to 2^20, -(2^0) to -(2^20)
    ranges = [-1, 0, 1, 2, 0.7]  # , *[x ** 2 for x in range(0, 10)], *[-(x ** 2) for x in range(0, 10)]

    for x in ranges:
        for y in ranges:
            sympy_gradients = get_sympy_gradients_evaluated_for_simple_func(x, y)
            partial_x_evaluated = my_partial_x_evaluated(x, y)
            partial_y_evaluated = my_partial_y_evaluated(x, y)
            assert round(float(sympy_gradients['x']), 5) == round(float(partial_x_evaluated), 5)
            assert round(float(sympy_gradients['y']), 5) == round(float(partial_y_evaluated), 5)


def test_partial_compared_to_sympy():
    a = 2

    s_X1 = Variable('X1')
    s_X2 = Variable('X2')
    s_W1_1 = Variable('W1_1')
    s_W1_2 = Variable('W1_2')
    s_b1 = Variable('b1')
    s_y = Variable('y')

    s_z1 = Add(Add(Multiply(s_X1, s_W1_1), Multiply(s_X2, s_W1_2)), s_b1)
    s_loss = Expression(Exponent(Add(s_z1, Multiply(Constant(-1), s_y)), Constant(2)))
    partial_W1_1 = forward_propagation_partial(s_loss, s_W1_1)
    partial_W1_2 = forward_propagation_partial(s_loss, s_W1_2)
    partial_b1 = forward_propagation_partial(s_loss, s_b1)

    # make lambdas
    def my_partial_W1_1_evaluated(W1_1, W1_2, b1, X1, X2, y):
        return partial_W1_1.evaluate({'X1': X1, 'X2': X2, 'W1_1': W1_1, 'W1_2': W1_2, 'b1': b1, 'y': y})

    def my_partial_W1_2_evaluated(W1_1, W1_2, b1, X1, X2, y):
        return partial_W1_2.evaluate({'X1': X1, 'X2': X2, 'W1_1': W1_1, 'W1_2': W1_2, 'b1': b1, 'y': y})

    def my_partial_b1_evaluated(W1_1, W1_2, b1, X1, X2, y):
        return partial_b1.evaluate({'X1': X1, 'X2': X2, 'W1_1': W1_1, 'W1_2': W1_2, 'b1': b1, 'y': y})

    # test ranges 0 to 10, -10 to 10, 2^0 to 2^20, -(2^0) to -(2^20)
    ranges = [-1, 0, 1, 2, 0.7]  # , *[x ** 2 for x in range(0, 10)], *[-(x ** 2) for x in range(0, 10)]]

    for X1 in ranges:
        for X2 in ranges:
            for W1_1 in ranges:
                for W1_2 in ranges:
                    for b1 in ranges:
                        for y in ranges:
                            sympy_gradients = get_sympy_gradients_evaluated_for_loss(W1_1, W1_2, b1, X1, X2, y)
                            partial_W1_1_evaluated = my_partial_W1_1_evaluated(W1_1, W1_2, b1, X1, X2, y)
                            partial_W1_2_evaluated = my_partial_W1_2_evaluated(W1_1, W1_2, b1, X1, X2, y)
                            partial_b1_evaluated = my_partial_b1_evaluated(W1_1, W1_2, b1, X1, X2, y)
                            assert round(float(sympy_gradients['W1_1']), 5) == round(float(partial_W1_1_evaluated), 5)
                            assert round(float(sympy_gradients['W1_2']), 5) == round(float(partial_W1_2_evaluated), 5)
                            assert round(float(sympy_gradients['b1']), 5) == round(float(partial_b1_evaluated), 5)
