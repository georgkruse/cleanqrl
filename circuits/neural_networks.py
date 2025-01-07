import tensorflow as tf    
from keras.layers import Dense, Input

def create_neural_network(seed):

    kernel_init = tf.keras.initializers.glorot_normal(seed)
    bias_init = tf.keras.initializers.constant(0)
    model = tf.keras.Sequential([
        Input(shape=(17,)),
        Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(6, activation='linear', kernel_initializer=kernel_init, bias_initializer=bias_init)
    ])  
    model.compile()
    return model


def set_model_weights(model, weights):
    # This part is used to update the weights for the online network
    last_used = 0
    weights = weights.astype('float32')
    for i in range(len(model.layers)):
        if 'conv' in model.layers[i].name or 'dense' in model.layers[i].name: 
            weights_shape = model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            model.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if model.layers[i].use_bias:
                weights_shape = model.layers[i].bias.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                model.layers[i].bias = new_weights
                last_used += no_of_weights
    return model