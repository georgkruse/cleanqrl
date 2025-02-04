import torch
import numpy as np
import pennylane as qml


def measurement(config, type=None, H=None):
    '''
    Creates the measurement block for the VQC.
    '''

    measurement_gates = {"paulix": qml.PauliX, "pauliz": qml.PauliZ, "pauliy": qml.PauliY}
    measurement_gate = measurement_gates[config["measurement_gate"].lower()]

    if type == 'actor':
        if config['measurement_type_actor'] == 'exp':
            return [qml.expval(measurement_gate(i)) for i in range(config['num_qubits'])]
        elif config['measurement_type_actor'] == 'expX':
            return [qml.expval(qml.PauliX(i)) for i in range(config['num_qubits'])]
        elif config['measurement_type_actor'] == 'exp0':
            return [qml.expval(measurement_gate(0))]
        elif config['measurement_type_actor'] == 'hamiltonian':
            return [qml.expval(H)]
        elif config['measurement_type_actor'] == 'probs':
            measurement = qml.probs(wires=[i for i in range(config['num_qubits'])])
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_actor'] == 'exp0_@_exp1':
            return [qml.expval(measurement_gate(0)@(measurement_gate(1)))]
        elif config['measurement_type_actor'] == 'exp_@_exp':
            return [qml.expval(measurement_gate(i)@measurement_gate(i+1)) for i in range(0, config['num_qubits']-1, 2)]
        elif config['measurement_type_actor'] == 'exp_@_exp_@_exp':
            return [qml.expval(measurement_gate(i)@measurement_gate(i+1)@measurement_gate(i+2)) for i in range(0, config['num_qubits']-1, 3)]
        elif config['measurement_type_actor'] == 'exp_@_exp+exp':
            measurement =  [[qml.expval(measurement_gate(i)@measurement_gate(i+1)),qml.expval(measurement_gate(i+2))] for i in range(0, config['num_qubits']-1, 3)]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_actor'] == 'exp_/_var':
            measurement = [[qml.expval(measurement_gate(i)), qml.var(measurement_gate(i+1))] for i in range(0, config['num_qubits']-1, 2)]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_actor'] == 'exp_+_var':
            measurement =  [[qml.expval(measurement_gate(i)), qml.var(measurement_gate(i))] for i in range(config['num_qubits'])]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_actor'] == 'jerbi':
            return [qml.expval(measurement_gate(0)@measurement_gate(1)@measurement_gate(2)@measurement_gate(3))]
        elif config['measurement_type_actor'] == 'edge':
            return  [qml.expval(measurement_gate(config['edge_measurement'][0].to(torch.int).item())@measurement_gate(config['edge_measurement'][1].to(torch.int).item()))] 
        else:
            print(f'Measurement type for {type} not implemented.')
        
    elif type == 'critic':
        if config['measurement_type_critic'] == 'exp':
            return [qml.expval(measurement_gate(i)) for i in range(config['num_qubits'])]
        elif config['measurement_type_critic'] == 'exp4':
            return [qml.expval(measurement_gate(i)) for i in range(4)]
        elif config['measurement_type_critic'] == 'exp0':
            return [qml.expval(measurement_gate(0))]
        elif config['measurement_type_critic'] == 'hamiltonian':
            return [qml.expval(H)]
        elif config['measurement_type_critic'] == 'probs':
            measurement = qml.probs(wires=[i for i in range(config['num_qubits'])])
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_critic'] == 'exp_@_exp':
            return [qml.expval(measurement_gate(i)@measurement_gate(i+1)) for i in range(0, config['num_qubits']-1, 2)]
        elif config['measurement_type_critic'] == 'exp_/_var':
            measurement = [[qml.expval(measurement_gate(i)), qml.var(measurement_gate(i+1))] for i in range(0, config['num_qubits']-1, 2)]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_critic'] == 'exp_+_var':
            measurement =  [[qml.expval(measurement_gate(i)), qml.var(measurement_gate(i))] for i in range(config['num_qubits'])]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type_critic'] == 'jerbi':
            return [qml.expval(measurement_gate(0)@measurement_gate(1)@measurement_gate(2)@measurement_gate(3))]
        else:
            print(f'Measurement type for {type} not implemented.')
        
    elif type == 'es' or type == 'ga':
        if config['measurement_type'] == 'exp':
            return [qml.expval(measurement_gate(i)) for i in range(config['num_qubits'])]
        elif config['measurement_type'] == 'exp_@_exp':
            return [qml.expval(measurement_gate(i)@measurement_gate(i+1)) for i in range(0, config['num_qubits']-1, 2)]
        elif config['measurement_type'] == 'exp_/_var':
            measurement = [[qml.expval(measurement_gate(i)), qml.var(measurement_gate(i+1))] for i in range(0, config['num_qubits']-1, 2)]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type'] == 'exp_+_var':
            measurement =  [[qml.expval(measurement_gate(i)), qml.var(measurement_gate(i))] for i in range(config['num_qubits'])]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type'] == 'probs':
            measurement = qml.probs(wires=[i for i in range(config['num_qubits'])])
            return list(np.reshape(measurement, -1))
        elif config['measurement_type'] == 'jerbi':
            return [qml.expval(measurement_gate(0)@measurement_gate(1)@measurement_gate(2)@measurement_gate(3))]
        elif config['measurement_type'] == 'probs_lunar':
            measurement = [qml.probs(wires=[i for i in range(0,4)]), qml.probs(wires=[i for i in range(4,8)])]
            return list(np.reshape(measurement, -1))
        elif config['measurement_type'] == 'probs_hopper':
            measurement = [qml.probs(wires=[i for i in range(0,3)]), qml.probs(wires=[i for i in range(3,6)]), qml.probs(wires=[i for i in range(6,9)])]
            return list(np.reshape(measurement, -1))
        else:
            print(f'Measurement type for {type} not implemented.')