import torch
import pennylane as qml

from circuits.graph.graph_circuits import graph_encoding_block

def encoding_block(config, theta, weights, layer, type=None):
    '''
    Creates the encoding block for the VQC.
    '''
    qubit_idx = 0
    
    if type == 'actor':
        if 'use_input_scaling' in config.keys():
            use_input_scaling = config['use_input_scaling']
            params = weights[f'input_scaling_{type}'][layer]
        elif 'use_input_scaling_actor' in config.keys():
            use_input_scaling = config['use_input_scaling_actor']
            params = weights[f'input_scaling_{type}'][layer]
    elif type == 'critic':
        use_input_scaling = config['use_input_scaling_critic']
        if use_input_scaling:
            params = weights[f'input_scaling_{type}'][layer]
    elif (type == 'es' or type == 'ga'):
        idx = int(config['num_layers']*config['num_qubits'])
        params = weights[2*idx:4*idx]
        params = params[layer*config['num_qubits']*config['num_scaling_params']:(layer+1)*config['num_qubits']*config['num_scaling_params']]
    
    if config['noise']['depolarizing'][0]:
        if config['encoding_type'] == 'angular_classical':
            if use_input_scaling:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                    qml.RZ(theta[:,i], wires=i)
                    qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
        else:
            print('ERROR: No encoding block selected.')

    else:
        if config['encoding_type'] == 'custom':
            if use_input_scaling:
                qubit_idx = 0
                if layer % 2 == 0:
                    for i in range(5):
                        qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                        qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                        qubit_idx += 2
                else:
                    for i in range(5):
                        qml.RY(theta[:,i+5]*params[qubit_idx], wires=i)
                        qml.RZ(theta[:,i+5]*params[qubit_idx+1], wires=i)
                        qubit_idx += 2

        elif config['encoding_type'] == 'graph_encoding':
            graph_encoding_block(config, theta, weights, layer, type)

        elif config['encoding_type'] == 'angular_classical':
            if use_input_scaling:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(theta.shape[1]):
                    qml.RY(theta[:,i], wires=i)
                    qml.RZ(theta[:,i], wires=i)       

        elif config['encoding_type'] == 'angular_classical_qubit':
            if use_input_scaling:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i], wires=i)
                    qml.RZ(theta[:,i], wires=i)

        elif config['encoding_type'] == 'angular_times_2':
            if use_input_scaling:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i]*params[qubit_idx], wires=i)
                    qml.RZ(2*theta[:,i]*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(theta[:,i], wires=i)
                    qml.RZ(2*theta[:,i], wires=i)
        
        elif config['encoding_type'] == 'angle_encoding_RX':
            if use_input_scaling:
                for i in range(theta.shape[1]):
                    qml.RX(theta[:,i]*params[i],wires=i)
            else:
                for i in range(theta.shape[1]):
                    qml.RX(theta[:,i],wires=i)
        
        elif config['encoding_type'] == 'angle_encoding_RX_all_qubits':
            if use_input_scaling:
                for i in range(theta.shape[1]):
                    qml.RX(theta[:,layer]*params[i],wires=i)
            else:
                for i in range(theta.shape[1]):
                    qml.RX(theta[:,layer],wires=i)

        elif config['encoding_type'] == 'angle_encoding_RX_atan':
            if use_input_scaling:
                for i in range(theta.shape[1]):
                    qml.RX(torch.arctan(theta[:,i]*params[i]),wires=i)
            else:
                for i in range(theta.shape[1]):
                    qml.RX(torch.arctan(theta[:,i]),wires=i)

        elif config['encoding_type'] == 'angular_arctan':
            if use_input_scaling:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i]*params[qubit_idx]), wires=i)
                    qml.RZ(torch.arctan(theta[:,i]*params[qubit_idx+1]), wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i]), wires=i)
                    qml.RZ(torch.arctan(theta[:,i]), wires=i)
        
        elif config['encoding_type'] == 'angular_arctan_ext':
            if use_input_scaling:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.arctan(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.arctan(theta[:,i]), wires=i)
                    qml.RZ(torch.arctan(theta[:,i]), wires=i)

        elif config['encoding_type'] == 'angular_sigmoid':
            if use_input_scaling:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i]*params[qubit_idx]), wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i]*params[qubit_idx+1]), wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i]), wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i]), wires=i)

        elif config['encoding_type'] == 'angular_sigmoid_ext':
            if use_input_scaling:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i])*params[qubit_idx], wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i])*params[qubit_idx+1], wires=i)
                    qubit_idx += 2
            else:
                for i in range(config['num_qubits']):
                    qml.RY(torch.sigmoid(theta[:,i]), wires=i)
                    qml.RZ(torch.sigmoid(theta[:,i]), wires=i)
    
        else:
            print('ERROR: No encoding block selected.')