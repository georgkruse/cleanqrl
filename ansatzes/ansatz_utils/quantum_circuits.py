import pennylane as qml
from circuits.encoding_block import encoding_block
from circuits.entangling_block import entangling_block
from circuits.variational_block import variational_block
from circuits.measurement import measurement

# from circuits.pooling.pooling_circuits import *
# from circuits.qcnn.qcnn_circuits import *

entangling_gates = {'CZ': qml.CZ, 'CNOT': qml.CNOT, 'CH': qml.Hadamard}

def vqc_generator(weights, theta, config, type, activations, H):
    '''
    General generator function to build the VQC.
    '''
    if config['use_hadamard']:
        for i in range(config['num_qubits']):
            qml.Hadamard(wires=i)
    
    if config['block_sequence'] == 'enc_ent_var':
        for layer in range(config['num_layers']):
            encoding_block(config, theta, weights, layer, type)
            entangling_block(config, type)
            variational_block(config, weights, layer, type)

    elif ((config['block_sequence'] == 'enc_var_ent') or (config['graph_encoding_type'] in ['angular-hea', 'hamiltonian-hea'])):
        for layer in range(config['num_layers']):
            encoding_block(config, theta, weights, layer, type)
            variational_block(config, weights, layer, type)
            entangling_block(config, type)
    
    elif config['block_sequence'] == 'var_ent':
        encoding_block(config, theta, weights, layer, type)
        for layer in range(config['num_layers']):
            variational_block(config, weights, layer, type)
            entangling_block(config, type)

    elif config['block_sequence'] == 'enc':
        for layer in range(config['num_layers']):
            encoding_block(config, theta, weights, layer, type)

    elif config['block_sequence'] == 'enc_ent':
        for layer in range(config['num_layers']):
            encoding_block(config, theta, weights, layer, type)
            entangling_block(config, type)

    elif config['block_sequence'] == 'enc_var':
        for layer in range(config['num_layers']):
            encoding_block(config, theta, weights, layer, type)
            variational_block(config, weights, layer, type)

    return measurement(config, type, H)   
    