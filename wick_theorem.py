import sympy as sp
import pytest


# ============================================================
# 1. STRUTTURE DATI
# ============================================================


class Derivative:
    """Represent a derivative acting on a field.

    Attributes:
        kind: Type of derivative, either "normal" or "tangential".
        index: Tangential coordinate index. Must be None for normal
            derivatives and specified for tangential derivatives.
    """

    def __init__(self, kind: str, index: int | None = None):
        self.kind = kind
        self.index = index

        # Validate the derivative type and its associated index.
        if self.kind not in ("normal", "tangential"):
            raise ValueError("Invalid derivative kind")

        if self.kind == "normal" and self.index is not None:
            raise ValueError("Normal derivative has no index")

        if self.kind == "tangential" and self.index is None:
            raise ValueError("Tangential derivative needs an index")


class Field:
    """Represent a field insertion, including its derivatives.

    Attributes:
        species: Field species, such as "phi" or "psi".
        point: Label identifying the insertion point.
        derivatives: Ordered tuple of derivatives acting on the field.
    """

    def __init__(
        self, species: str, point: str, derivatives: tuple[Derivative, ...] = ()
    ):
        self.species = species
        self.point = point
        self.derivatives = derivatives


class Monomial:
    """Represent a product of fields multiplied by a coefficient.

    Attributes:
        coefficient: Symbolic prefactor multiplying the fields.
        fields: Tuple of field insertions in the monomial.
    """

    def __init__(self, coefficient: sp.Expr, fields: tuple[Field, ...]):
        self.coefficient = coefficient
        self.fields = fields


class Operator:
    """Represent an operator as a sum of monomials.

    Attributes:
        terms: Tuple of monomials contributing to the operator.
    """

    def __init__(self, terms: tuple[Monomial, ...]):
        self.terms = terms


# ============================================================
# 2. CONTRACTION OF TWO FIELDS
# ============================================================


def contraction(a: Field, b: Field, coordinates, propagators):
    """Compute the two-point contraction of two field insertions.

    The propagator is selected according to the field species.
    Derivatives acting on either field are applied to the corresponding
    arguments of the propagator.

    Contractions at the same point and psi-psi contractions are set to zero.

    Args:
        a: First field insertion.
        b: Second field insertion.
        coordinates: Dictionary mapping point labels to coordinate tuples
            (tangential_coordinates, normal_coordinate).
        propagators: Dictionary mapping ordered pairs of field species
            to propagator functions.

    Returns:
        A SymPy expression representing the differentiated propagator.

    Raises:
        ValueError: If no propagator is defined for the field species.
        IndexError: If a tangential derivative has an invalid index.
    """
    # Exclude self-contractions and psi-psi contractions.
    if a.point == b.point:
        return sp.Integer(0)

    if a.species == "psi" and b.species == "psi":
        return sp.Integer(0)

    # Retrieve the coordinates of the two insertion points.
    xa_t, xa_n = coordinates[a.point]
    xb_t, xb_n = coordinates[b.point]

    key = (a.species, b.species)

    # Look for the propagator in the given ordering.
    if key in propagators:
        G = propagators[key]
    # If necessary, use the propagator with reversed field ordering.
    # The coordinate arguments and field labels must be exchanged too.
    elif (b.species, a.species) in propagators:
        G = propagators[(b.species, a.species)]

        xa_t, xb_t = xb_t, xa_t
        xa_n, xb_n = xb_n, xa_n
        a, b = b, a

    else:
        raise ValueError(f"No propagator defined for {key}")

    # Evaluate the propagator at the two insertion points.
    result = G(xa_t, xa_n, xb_t, xb_n)

    # Apply derivatives acting on the first field.
    for derivative in a.derivatives:
        if derivative.kind == "normal":
            result = sp.diff(result, xa_n)
        else:
            index = derivative.index

            if index is None:
                raise ValueError("Tangential derivative needs an index")

            if not 0 <= index < len(xa_t):
                raise IndexError("Invalid tangential index")
            result = sp.diff(result, xa_t[derivative.index])
    # Apply derivatives acting on the second field.
    for derivative in b.derivatives:
        if derivative.kind == "normal":
            result = sp.diff(result, xb_n)
        else:
            index = derivative.index

            if index is None:
                raise ValueError("Tangential derivative needs an index")

            if not 0 <= index < len(xb_t):
                raise IndexError("Invalid tangential index")
            result = sp.diff(result, xb_t[derivative.index])

    return result


# ============================================================
# 3. PRODUCT OF OPERATORS
# ============================================================


def multiply_operators(operator_a, operator_b):
    """Construct the product of two operators.

    Each monomial in the first operator is multiplied by every monomial
    in the second operator. Coefficients are multiplied and field tuples
    are concatenated.

    Args:
        operator_a: First operator.
        operator_b: Second operator.

    Returns:
        An Operator containing all pairwise products of monomials.
    """
    terms = []

    # Expand the product distributively over all pairs of monomials.
    for term_a in operator_a.terms:
        for term_b in operator_b.terms:
            terms.append(
                Monomial(
                    coefficient=term_a.coefficient * term_b.coefficient,
                    fields=term_a.fields + term_b.fields,
                )
            )

    return Operator(tuple(terms))


# ============================================================
# 4. COMPLETE WICK CONTRACTIONS
# ============================================================


def wick_pairings(fields, coordinates, propagators):
    """Enumerate all complete pairings of a tuple of fields.

    The algorithm recursively pairs the first field with each of the
    remaining fields and computes all possible pairings of the fields
    that are left.

    Pairings containing a vanishing contraction are discarded.

    Args:
        fields: Tuple of field insertions to be contracted.
        coordinates: Dictionary of insertion-point coordinates.
        propagators: Dictionary of propagator functions.

    Returns:
        A list of (value, edges) pairs, where value is the product of
        propagators and edges records the paired insertion-point labels.

        Returns [(1, ())] for an empty field tuple and an empty list
        when the number of fields is odd.
    """

    # An empty set of fields has one empty pairing, with value 1.
    if len(fields) == 0:
        return [(sp.Integer(1), ())]

    # Complete pairings are impossible for an odd number of fields.
    if len(fields) % 2 != 0:
        return []

    first = fields[0]
    results = []

    # Pair the first field with each possible partner.
    for j in range(1, len(fields)):
        partner = fields[j]
        # Remove the paired fields to obtain the remaining problem.
        remaining = fields[1:j] + fields[j + 1 :]

        pair = contraction(first, partner, coordinates, propagators)

        # Skip pairings whose propagator vanishes.
        if pair == 0:
            continue
        # Record the contraction as an edge between insertion points.
        edge = (first.point, partner.point)

        # Complete the pairing recursively and combine its contribution.
        for rest_value, rest_edges in wick_pairings(
            remaining, coordinates, propagators
        ):
            results.append(
                (
                    pair * rest_value,
                    (edge,) + rest_edges,
                )
            )

    return results


def wick_contraction(monomial, coordinates, propagators):
    """Evaluate all complete Wick contractions of a monomial.

    Args:
        monomial: Monomial whose fields are to be contracted.
        coordinates: Dictionary of insertion-point coordinates.
        propagators: Dictionary of propagator functions.

    Returns:
        The sum of all complete pairing contributions, multiplied by
        the monomial coefficient.
    """
    pairings = wick_pairings(monomial.fields, coordinates, propagators)

    # Sum the propagator products over all complete pairings.
    result = sum(
        (value for value, edges in pairings),
        sp.Integer(0),
    )
    return monomial.coefficient * result
    # Optional alternative:
    # return sp.simplify(monomial.coefficient * result)


def wick_operator_product(operator_product, coordinates, propagators):
    """Evaluate all complete Wick contractions of an operator.

    Each monomial is contracted independently, and the results are summed.

    Args:
        operator_product: Operator whose monomials are to be contracted.
        coordinates: Dictionary of insertion-point coordinates.
        propagators: Dictionary of propagator functions.

    Returns:
        A simplified SymPy expression for the complete contraction sum.
    """
    result = sum(
        (
            wick_contraction(term, coordinates, propagators)
            for term in operator_product.terms
        ),
        sp.Integer(0),
    )

    return sp.simplify(result)


# ============================================================
# 7. COSTRUZIONE DI T_nn
# ============================================================


def make_Tnn_DD(point, p):
    """Construct the T_nn operator for Dirichlet-Dirichlet boundary conditions.

    Args:
        point: Label of the insertion point.
        p: Number of tangential coordinate directions.

    Returns:
        An Operator containing the monomials of T_nn for DD conditions.

    Each monomial is stored with an overall factor of 1/2 in its
    coefficient, as implemented by the local add() helper.
    """
    terms = []
    N = Derivative("normal")

    def F(species, *derivatives):
        return Field(species, point, tuple(derivatives))

    def add(coefficient, *fields):
        terms.append(Monomial(sp.sympify(coefficient) / 2, tuple(fields)))

    # -2 (dn psi)(dn phi)
    add(-2, F("psi", N), F("phi", N))

    # -(4/3) (da dn phi)(da dn phi)
    for a in range(p):
        da = Derivative("tangential", a)
        add(
            sp.Rational(-4, 3),
            F("phi", da, N),
            F("phi", da, N),
        )

    # -(4/3) (dn phi)(dn da da phi)
    for a in range(p):
        da = Derivative("tangential", a)
        add(
            sp.Rational(-4, 3),
            F("phi", N),
            F("phi", N, da, da),
        )

    return Operator(tuple(terms))


def make_Tnn_NN(point, p):
    """Construct the T_nn operator for Neumann-Neumann boundary conditions.

    Args:
        point: Label of the insertion point.
        p: Number of tangential coordinate directions.

    Returns:
        An Operator containing the monomials of T_nn for NN conditions.

    Each monomial is stored with an overall factor of 1/2 in its
    coefficient, as implemented by the local add() helper.
    """
    terms = []
    N = Derivative("normal")

    def F(species, *derivatives):
        return Field(species, point, tuple(derivatives))

    def add(coefficient, *fields):
        terms.append(Monomial(sp.sympify(coefficient) / 2, tuple(fields)))

    # -2 (sum_a da da phi)^2
    for a in range(p):
        for b in range(p):
            da = Derivative("tangential", a)
            db = Derivative("tangential", b)
            add(
                -2,
                F("phi", da, da),
                F("phi", db, db),
            )

    # + psi^2
    add(1, F("psi"), F("psi"))

    # +2 (da psi)(da phi)
    for a in range(p):
        da = Derivative("tangential", a)
        add(2, F("psi", da), F("phi", da))

    # -(4/3) (db phi)(db da da phi)
    for a in range(p):
        for b in range(p):
            da = Derivative("tangential", a)
            db = Derivative("tangential", b)
            add(
                sp.Rational(-4, 3),
                F("phi", db),
                F("phi", db, da, da),
            )

    # +(2/3) (da db phi)(da db phi)
    for a in range(p):
        for b in range(p):
            da = Derivative("tangential", a)
            db = Derivative("tangential", b)
            add(
                sp.Rational(2, 3),
                F("phi", da, db),
                F("phi", da, db),
            )

    return Operator(tuple(terms))


# ============================================================
# 6. SELECTION OF CONNECTED CONTRACTIONS (to be tested)
# ============================================================


def is_connected(vertices, edges):
    """Check whether a graph is connected.

    Args:
        vertices: Collection of graph vertices.
        edges: Collection of pairs representing undirected edges.

    Returns:
        True if every vertex can be reached from any other vertex,
        and False otherwise.
    """

    vertices = set(vertices)

    # A graph with zero or one vertex is connected by convention.
    if len(vertices) <= 1:
        return True

    # Build an adjacency list for the undirected graph.
    adjacency = {v: set() for v in vertices}

    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Use depth-first search to visit all vertices reachable from
    # an arbitrary starting vertex.
    visited = set()
    stack = [next(iter(vertices))]

    while stack:
        vertex = stack.pop()

        if vertex in visited:
            continue

        visited.add(vertex)
        stack.extend(adjacency[vertex] - visited)

    return visited == vertices


def connected_wick_monomial(monomial, coordinates, propagators):
    """Evaluate only the connected Wick contractions of a monomial.

    Each complete pairing defines a graph whose vertices are the
    insertion points and whose edges are the contractions.

    Args:
        monomial: Monomial whose fields are to be contracted.
        coordinates: Dictionary of insertion-point coordinates.
        propagators: Dictionary of propagator functions.

    Returns:
        The sum of contributions from pairings whose graph is connected,
        multiplied by the monomial coefficient.
    """

    vertices = {field.point for field in monomial.fields}

    total = sp.Integer(0)

    # Generate complete pairings and retain only connected graphs.
    for value, edges in wick_pairings(monomial.fields, coordinates, propagators):
        if is_connected(vertices, edges):
            total += value

    return monomial.coefficient * total


def connected_wick(operator, coordinates, propagators):
    """Evaluate the connected Wick contractions of an operator.

    The connected contributions from all monomials are summed.

    Args:
        operator: Operator whose connected contractions are required.
        coordinates: Dictionary of insertion-point coordinates.
        propagators: Dictionary of propagator functions.

    Returns:
        A SymPy expression containing the sum of connected contractions.
    """

    return sum(
        (
            connected_wick_monomial(term, coordinates, propagators)
            for term in operator.terms
        ),
        sp.Integer(0),
    )
