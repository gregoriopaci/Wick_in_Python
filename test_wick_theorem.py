"""Tests for the Wick contraction module.

This module defines propagators, coordinate configurations, and tests for
field contractions, derivatives, operator construction, and the DD boundary
two-point function.
"""

import sympy as sp
import pytest

from wick_theorem import (
    Derivative,
    Field,
    contraction,
    Monomial,
    wick_contraction,
    wick_pairings,
    multiply_operators,
    make_Tnn_DD,
    make_Tnn_NN,
)


# ============================================================
# 1. COORDINATES AND PROPAGATORS
# ============================================================

# Number of tangential dimensions along the boundary.
p = 3

x1, x2, x3, z = sp.symbols("x1 x2 x3 z", real=True)
y1, y2, y3, w = sp.symbols("y1 y2 y3 w", real=True)

# Tangential and normal coordinates of the two insertion points.
coordinates = {
    "x": ((x1, x2, x3), z),
    "y": ((y1, y2, y3), w),
}

m = sp.symbols("m", positive=True)


def G_phiphi(
    x_t: tuple[sp.Symbol, ...],
    x_n: sp.Symbol,
    y_t: tuple[sp.Symbol, ...],
    y_n: sp.Symbol,
) -> sp.Expr:
    """Return the propagator <phi(x) phi(y)>.

    The propagator is expressed as the difference between the direct
    contribution and the image contribution.

    Args:
        x_t: Tangential coordinates of the first point.
        x_n: Normal coordinate of the first point.
        y_t: Tangential coordinates of the second point.
        y_n: Normal coordinate of the second point.

    Returns:
        The symbolic phi-phi propagator.
    """
    # Squared tangential distance between the insertion points.
    r2 = sum((xi - yi) ** 2 for xi, yi in zip(x_t, y_t))

    # Squared distances for the direct and image contributions.
    direct = r2 + (x_n - y_n) ** 2
    image = r2 + (x_n + y_n) ** 2

    return (-sp.log(m**2 * direct) + sp.log(m**2 * image)) / (16 * sp.pi**2)


def G_psiphi(x_t, x_n, y_t, y_n):
    """Return the propagator <psi(x) phi(y)>.

    The propagator is given by the difference between the inverse
    squared distances of the direct and image contributions.

    Args:
        x_t: Tangential coordinates of the first point.
        x_n: Normal coordinate of the first point.
        y_t: Tangential coordinates of the second point.
        y_n: Normal coordinate of the second point.

    Returns:
        The symbolic psi-phi propagator.
    """

    # Squared tangential distance between the insertion points.
    r2 = sum((xi - yi) ** 2 for xi, yi in zip(x_t, y_t))

    direct = r2 + (x_n - y_n) ** 2
    image = r2 + (x_n + y_n) ** 2

    return (-1 / direct + 1 / image) / (4 * sp.pi**2)


# Map each ordered pair of field species to its propagator.
propagators = {
    ("phi", "phi"): G_phiphi,
    ("psi", "phi"): G_psiphi,
}

# An optional alternative definition with an explicit psi-psi propagator:
#
# def G_psipsi(x_t, x_n, y_t, y_n):
#     """Return the vanishing psi-psi propagator."""
#     return sp.Integer(0)
#
# propagators = {
#     ("phi", "phi"): G_phiphi,
#     ("psi", "phi"): G_psiphi,
#     ("psi", "psi"): G_psipsi,
# }


# ============================================================
# 2. PROPAGATOR TESTS
# ============================================================


def test_phi_phi_propagator():
    """Check that the phi-phi contraction reproduces its propagator."""
    result = contraction(
        Field("phi", "x"),
        Field("phi", "y"),
        coordinates,
        propagators,
    )

    expected = G_phiphi(*coordinates["x"], *coordinates["y"])

    assert sp.simplify(result - expected) == 0


def test_psi_phi_propagator():
    """Check that the psi-phi contraction reproduces its propagator."""
    result = contraction(
        Field("psi", "x"),
        Field("phi", "y"),
        coordinates,
        propagators,
    )
    # Evaluate the expected propagator directly.
    expected = G_psiphi(*coordinates["x"], *coordinates["y"])

    assert sp.simplify(result - expected) == 0


def test_reverse_propagator_order():
    """Check that the contraction supports reversed field ordering.

    The available psi-phi propagator is used with the insertion points
    exchanged to evaluate the phi-psi contraction.
    """
    result = contraction(
        Field("phi", "x"),
        Field("psi", "y"),
        coordinates,
        propagators,
    )

    expected = G_psiphi(*coordinates["y"], *coordinates["x"])

    assert sp.simplify(result - expected) == 0


# ============================================================
# 3. DERIVATIVE TESTS
# ============================================================


def test_normal_derivatives():
    """Check a contraction with one normal derivative on each field."""
    result = contraction(
        Field("phi", "x", (Derivative("normal"),)),
        Field("phi", "y", (Derivative("normal"),)),
        coordinates,
        propagators,
    )

    G = G_phiphi(*coordinates["x"], *coordinates["y"])
    # Differentiate the propagator directly with respect to z and w.
    expected = sp.diff(G, z, w)

    assert sp.simplify(result - expected) == 0


def test_tangential_derivatives():
    """Check a contraction with tangential derivatives on both fields.

    Both derivatives act along the first tangential coordinate.
    """
    result = contraction(
        Field(
            "phi",
            "x",
            (Derivative("tangential", 0),),
        ),
        Field(
            "phi",
            "y",
            (Derivative("tangential", 0),),
        ),
        coordinates,
        propagators,
    )

    # Compare with direct differentiation with respect to x1 and y1.
    G = G_phiphi(*coordinates["x"], *coordinates["y"])
    expected = sp.diff(G, x1, y1)

    assert sp.simplify(result - expected) == 0


def test_mixed_derivatives():
    """Check a contraction with mixed normal and tangential derivatives.

    The first field carries a normal derivative and a derivative along
    the second tangential coordinate. The second field carries a
    normal derivative.
    """
    result = contraction(
        Field(
            "phi",
            "x",
            (
                Derivative("normal"),
                Derivative("tangential", 1),
            ),
        ),
        Field(
            "phi",
            "y",
            (Derivative("normal"),),
        ),
        coordinates,
        propagators,
    )

    G = G_phiphi(*coordinates["x"], *coordinates["y"])
    expected = sp.diff(G, z, x2, w)

    assert sp.simplify(result - expected) == 0


# ============================================================
# 4. SPECIAL CASES
# ============================================================


def test_same_point_contraction_is_zero():
    """Check that contractions between fields at the same point vanish."""
    result = contraction(
        Field("phi", "x"),
        Field("phi", "x"),
        coordinates,
        propagators,
    )

    assert result == 0


def test_psi_psi_contraction_is_zero():
    """Check that psi-psi contractions vanish."""
    result = contraction(
        Field("psi", "x"),
        Field("psi", "y"),
        coordinates,
        propagators,
    )

    assert result == 0


# ============================================================
# 5. CONSTRUCTION OF THE T_nn OPERATORS
# ============================================================


def test_Tnn_construction():
    """Check the number of monomials in the DD and NN operators."""
    T_DD_x = make_Tnn_DD("x", p)
    T_NN_x = make_Tnn_NN("x", p)

    # DD: one mixed term and p terms for each of the two indexed sums.
    assert len(T_DD_x.terms) == 1 + p + p

    # NN: p^2 + 1 + p + p^2 + p^2 terms.
    assert len(T_NN_x.terms) == 3 * p**2 + p + 1


# ============================================================
# 6. CONNECTED TWO-POINT CORRELATOR
# ============================================================


def test_DD_differentiated_monomial_pair():
    """Check the Wick contraction of a pair of differentiated DD monomials.

    The selected monomials contain (d_a d_n phi)^2 at each insertion
    point, with a = 0. The expected result is proportional to the
    square of the corresponding differentiated propagator.
    """
    T_x = make_Tnn_DD("x", p)
    T_y = make_Tnn_DD("y", p)

    # Select the monomial corresponding to a = 0 in each operator.
    term_x = T_x.terms[1]
    term_y = T_y.terms[1]

    # Construct the product of the selected monomials.
    product_term = Monomial(
        coefficient=term_x.coefficient * term_y.coefficient,
        fields=term_x.fields + term_y.fields,
    )
    # Enumerate all complete pairings for diagnostic purposes.
    pairings = wick_pairings(product_term.fields, coordinates, propagators)

    print("Numero di contrazioni:", len(pairings))

    for value, edges in pairings:
        print("Contrazione:", edges)
        sp.pprint(value)

    # Evaluate the complete Wick contraction.
    result = wick_contraction(product_term, coordinates, propagators)

    # Expected result from the differentiated phi-phi propagator.
    G = G_phiphi(*coordinates["x"], *coordinates["y"])
    D = sp.diff(G, x1, z, y1, w)

    expected = sp.Rational(8, 9) * D**2

    assert sp.simplify(result - expected) == 0


def calculate_DD_boundary_2pt():
    """Calculate the DD boundary two-point function.

    Construct T_nn at two insertion points, multiply the operators,
    evaluate all complete Wick contractions, and restrict the result
    to the boundary.

    The insertion points are separated by a distance r along the first
    tangential direction.

    Returns:
        tuple: The tangential separation r and the simplified correlator.
    """
    T_x = make_Tnn_DD("x", p)
    T_y = make_Tnn_DD("y", p)

    # Construct the product T_nn(x) T_nn(y).
    product = multiply_operators(T_x, T_y)

    r = sp.symbols("r", positive=True)

    boundary_subs = {
        z: 0,
        w: 0,
        x1: r,
        x2: 0,
        x3: 0,
        y1: 0,
        y2: 0,
        y3: 0,
    }

    result = sp.Integer(0)

    for term in product.terms:
        contribution = wick_contraction(term, coordinates, propagators)
        result += contribution.subs(boundary_subs)

    return r, sp.factor(sp.cancel(sp.together(result)))


def test_DD_boundary_two_point():
    """Check the DD displacement-operator two-point function.

    Conformal invariance requires the boundary two-point function to
    scale as r^(-8) in this setup.

    The test checks the expected coefficient and extracts the
    corresponding value of b1 using the convention adopted here.
    """
    r, correlator = calculate_DD_boundary_2pt()

    print("Correlatore DD:")
    sp.pprint(correlator)

    # Extract the coefficient of the r^(-8) dependence.
    C_D = sp.factor(r**8 * correlator)

    # Extract b1 using the normalization convention of this calculation.
    b1 = sp.simplify(sp.Rational(2, 15) * sp.pi**4 * C_D)

    print("C_D =", C_D)
    print("b1 =", b1)

    # Check the expected DD two-point function.
    assert sp.simplify(correlator + 4 / (sp.pi**4 * r**8)) == 0

    # Check the corresponding value of b1.
    assert sp.simplify(b1 + sp.Rational(8, 15)) == 0
