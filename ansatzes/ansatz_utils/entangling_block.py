import pennylane as qml


entangling_gates = {'CZ': qml.CZ, 'CNOT': qml.CNOT, 'CH': qml.Hadamard}

def entangling_gate_hadamard(control, target, config):
    qml.ctrl(entangling_gates[config['entangling_gate']], control=control)(wires=target)


def entangling_block(config, type=None):
    '''
    Creates the entangling block for the VQC.
    '''
    entangling_gate = entangling_gates[config['entangling_gate']]
        
    if config['entangling_type'] == 'full':
        for i in range(config['num_qubits']):
            for j in range(i + 1, config['num_qubits']):
                entangling_gate(wires=[i,j])

    elif config['entangling_type'] == 'chain':
        for i in range(config['num_qubits']):
            x = i
            xx = x+1
            if i == config['num_qubits'] - 1:
                xx = 0
            if config['entangling_gate'] == 'CH':
                entangling_gate_hadamard(x, xx, config)
            else:
                entangling_gate(wires=[x,xx])
    
    elif config['entangling_type'] == 'chain_reverse':
        entangling_gate(wires=[0,7])
        entangling_gate(wires=[7,6])
        entangling_gate(wires=[6,5])
        entangling_gate(wires=[5,4])
        entangling_gate(wires=[4,3])
        entangling_gate(wires=[3,2])
        entangling_gate(wires=[2,1])
        entangling_gate(wires=[1,0])


    elif config['entangling_type'] == 'lunar_custom_1':
        for i in range(1, config['num_qubits']):
            entangling_gate(wires=[i,0])

            if i != 2:
                entangling_gate(wires=[i,2])

            entangling_gate(wires=[0,1])
            entangling_gate(wires=[2,3])
    
    elif config['entangling_type'] == 'lunar_custom_2':
        if type == 'actor':
            if config['measurement_type_actor'] == 'exp_@_exp':
                for i in range(config['num_qubits']):
                    x = i
                    xx = x+1
                    if i == config['num_qubits'] - 1:
                        xx = 0
                    entangling_gate(wires=[x,xx])
            elif config['measurement_type_actor'] == 'exp':
                for i in range(1, config['num_qubits']):
                    entangling_gate(wires=[i,0])
                    if i != 2:
                        entangling_gate(wires=[i,2])
                    entangling_gate(wires=[0,1])
                    entangling_gate(wires=[2,3])

        elif type == 'critic':
            if config['measurement_type_critic'] == 'exp_@_exp':
                for i in range(1, config['num_qubits']):
                    entangling_gate(wires=[i,0])
                    if i != 1:
                        entangling_gate(wires=[i,1])
                                    
            elif config['measurement_type_critic'] == 'exp':
                for i in range(1, config['num_qubits']):
                    entangling_gate(wires=[i,0])

    elif config['entangling_type'] == 'inverse_1':
    
        entangling_gate(wires=[4,5])
        entangling_gate(wires=[3,2])

        entangling_gate(wires=[5,6])
        entangling_gate(wires=[2,1])

        entangling_gate(wires=[6,7])
        entangling_gate(wires=[1,0])

        entangling_gate(wires=[7,3])
        entangling_gate(wires=[0,4])

    elif config['entangling_type'] == 'inverse_2':
    
        entangling_gate(wires=[4,3])
        entangling_gate(wires=[3,5])

        entangling_gate(wires=[5,2])
        entangling_gate(wires=[2,6])

        entangling_gate(wires=[6,1])
        entangling_gate(wires=[1,7])

        entangling_gate(wires=[7,0])
        entangling_gate(wires=[0,4])
       
    else:
        print('ERROR: No entangling block selected.')