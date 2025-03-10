import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax.scipy.linalg import expm
from pennylane import I, X, Z

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

qubits = [2,4,6,8,10]

for n in qubits: 
    # n = 10  # number of qubits.
    # generators = [Z(i) @ Z(i + 1) for i in range(n - 1)]
    # generators = [sum(Z(i) @ Z(i + 1) for i in range(n - 1))]
    # generators = [Z(0)@Z(1)+Z(1)@Z(2)+Z(2)@Z(3)]

    generators = []
    sum_Z = []
    for i in range(n):
        for j in range(i + 1, n):
            sum_Z.append(Z(i)@Z(j))
    generators.append(sum(sum_Z))
    # generators += [X(i) for i in range(n)]
    generators += [sum(X(i) for i in range(n))]
    # generators += [sum(Z(i) for i in range(n))]
    # generators += [X(0)+X(1)+X(2)+X(3)]

    # work with PauliSentence instances for efficiency
    generators = [op.pauli_rep for op in generators]

    dla = qml.pauli.lie_closure(generators, pauli=True)
    dim_g = len(dla)

    print(dim_g)
exit()
# compute initial expectation value vector
e_in = np.zeros(dim_g, dtype=float)

w = np.zeros(dim_g, dtype=float)
w[: len(generators)] = 0.5
w = jnp.array(w)


for i, h_i in enumerate(dla):
    # initial state |0x0| = (I + Z)/2, note that trace function
    # below already normalizes by the dimension,
    # so we can ommit the explicit factor /2
    rho_in = qml.prod(*(I(i) + Z(i) for i in h_i.wires))
    rho_in = rho_in.pauli_rep

    e_in[i] = (h_i @ rho_in).trace()

e_in = jnp.array(e_in)
print(rho_in)
adjoint_repr = qml.pauli.structure_constants(dla)

depth = 10
gate_choice = np.random.choice(dim_g, size=depth)
gates = adjoint_repr[gate_choice]


def forward(theta):
    # simulation
    e_t = e_in
    for i in range(depth):
        e_t = expm(theta[i] * gates[i]) @ e_t

    # final expectation value
    result_g_sim = w @ e_t

    return result_g_sim.real


theta = jax.random.normal(jax.random.PRNGKey(0), shape=(10,))

gsim_forward, gsim_backward = forward(theta), jax.grad(forward)(theta)


H = 0.5 * qml.sum(*[op.operation() for op in generators])


@qml.qnode(qml.device("default.qubit", wires=n), interface="jax")
def qnode(theta):
    for i, mu in enumerate(gate_choice):
        qml.exp(-1j * theta[i] * dla[mu].operation())
    return qml.expval(H)


statevec_forward, statevec_backward = qnode(theta), jax.grad(qnode)(theta)

print(
    qml.math.allclose(statevec_forward, gsim_forward),
    qml.math.allclose(statevec_backward, gsim_backward),
)
