import pennylane as qml

def layer_hamiltonian_encoding_ansatz(x, input_scaling_params, rotational_params, wires):
    """
    x: input (batch_size,num_features)
    input_scaling_params: vector of parameters (num_features)
    rotational_params:  vector of parameters (num_features*2)
    """
    for i, wire in enumerate(wires):
        qml.RX(input_scaling_params[i] * x[:,i], wires = [wire])
    
    for i, wire in enumerate(wires):
        qml.RY(rotational_params[i], wires = [wire])

    for i, wire in enumerate(wires):
        qml.RZ(rotational_params[i+len(wires)], wires = [wire])

    if len(wires) == 2:
        qml.CZ(wires = wires)
    else:
        for i in range(len(wires)):
            qml.CZ(wires = [wires[i],wires[(i+1)%len(wires)]])


def hamiltonian_encoding_ansatz(x, input_scaling_weights,variational_weights, wires, layers, type_, observables = "None"):
    for layer in range(layers):
        layer_hamiltonian_encoding_ansatz(x, input_scaling_weights[layer], variational_weights[layer], wires)
    if type_ == "critic":
        return qml.expval(qml.PauliZ(0))
    elif type_ == "actor":
        if observables == "local":
            half = len(wires) // 2

            # Create the tensor product for the first half
            left_observable = qml.PauliZ(0)
            for i in range(1, half):
                left_observable = left_observable @ qml.PauliZ(i)

            # Create the tensor product for the second half
            right_observable = qml.PauliZ(half)
            for i in range(half + 1, len(wires)):
                right_observable = right_observable @ qml.PauliZ(i)
        
        elif observables == "global":

            left_observable = qml.PauliZ(0)
            right_observable = qml.PauliX(0)

            for i in range(1,len(wires)):
                left_observable = left_observable @ qml.PauliZ(i)
                right_observable = right_observable @ qml.PauliX(i)
            
            left_observable = qml.Hamiltonian([1.0], [left_observable])
            right_observable = qml.Hamiltonian([1.0], [right_observable])
        
        return [qml.expval(left_observable), qml.expval(right_observable)]
    

