import torch
import pennylane as qml


def variational_block(config, weights, layer, type=None):
    '''
    Creates the variational block for the VQC.
    '''
    z = 0
    if (type == 'actor' or type =='critic'):
        params = weights[f'weights_{type}'][layer]
        if config['encoding_type'] == 'graph_encoding':
            params = params[-config['num_qubits']*config['num_variational_params']:]
    elif (type == 'es' or type == 'ga') and config['use_input_scaling']:
        idx = int(config['num_layers']*config['num_qubits'])
        params = weights[:2*idx]
        params = params[layer*config['num_qubits']*config['num_variational_params']:(layer+1)*config['num_qubits']*config['num_variational_params']]

    if config['noise']['coherent'][0]:
        noise = torch.normal(0., config['noise']['coherent'][1]**2, params.shape)
    
        if config['variational_type'] == 'RY_RZ':
            for i in range(config['num_qubits']):
                qml.RY(params[z], wires=i)
                qml.RY(noise[z], wires=i)
                qml.RZ(params[z+1], wires=i)
                qml.RZ(noise[z+1], wires=i)
                z += 2

        elif config['variational_type'] == 'RZ_RY':
            for i in range(config['num_qubits']):
                qml.RZ(params[z], wires=i)
                qml.RZ(noise[z], wires=i)
                qml.RY(params[z+1], wires=i)
                qml.RY(noise[z+1], wires=i)
                z += 2
        else:
            print('ERROR: No variational block selected.')

    elif config['noise']['depolarizing'][0]:
    
        if config['variational_type'] == 'RY_RZ':
            for i in range(config['num_qubits']):
                qml.RY(params[z], wires=i)
                qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                qml.RZ(params[z+1], wires=i)
                qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                z += 2

        elif config['variational_type'] == 'RZ_RY':
            for i in range(config['num_qubits']):
                qml.RZ(params[z], wires=i)
                qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                qml.RY(params[z+1], wires=i)
                qml.DepolarizingChannel(config['noise']['depolarizing'][1], wires=i)
                z += 2
        else:
            print('ERROR: No variational block selected.')

    else:
        if config['variational_type'] == 'RY_RZ':
            for i in range(config['num_qubits']):
                qml.RY(params[z], wires=i)
                qml.RZ(params[z+1], wires=i)
                z += 2

        elif config['variational_type'] == 'RZ_RY':
            for i in range(config['num_qubits']):
                qml.RZ(params[z], wires=i)
                qml.RY(params[z+1], wires=i)
                z += 2
        else:
            print('ERROR: No variational block selected.')