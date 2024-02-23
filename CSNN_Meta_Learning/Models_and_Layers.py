# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:47:38 2023

@author: nalmrb
"""

# This script houses the layers and models used in the training and testing
# scripts

import tensorflow as tf
import numpy as np
from tools_and_functions import SpikeFunction, exp_convolve

### Layers ###

class LIF_activation(tf.keras.layers.Layer):
    def __init__(self, thr = 1, beta = 0.5, dampening_factor = 0.3):
        self.description = "LIF activation layer"
        self.decay = beta
        self.dampening_factor = dampening_factor
        self.thr = thr
        
        super(LIF_activation,self).__init__()
        
    def call(self, inputs, states = None, constants = None):
        if states == None:
            old_v = tf.zeros(shape = tf.shape(inputs))
            old_spike = tf.zeros(shape = tf.shape(inputs))
        else:
            old_v = states[0]
            old_spike = states[1]
        
        #update voltage
        d = self.decay
        i_reset = self.thr * old_spike
        new_v = d * old_v + inputs - i_reset
        
        v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        
        new_state = (new_v, new_z)
        return new_state

class LIF(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr = 0.05, tau = 0.01, dampening_factor = 0.3):
        self.description = "LIF layer"
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tau #tf.exp(-1 / tau)       
        self.dampening_factor = dampening_factor
        self.thr = thr        
        self.input_weights = None
        
        super(LIF,self).__init__()
        
    def build(self, input_shape):
        n = self.num_neurons
        rand_init = tf.keras.initializers.glorot_uniform()

        self.input_weights = self.add_weight(shape = (int(input_shape[-1]), n), 
                                            initializer = rand_init,
                                            name = 'input_weights', trainable = True)
        
    def call(self, inputs, states, constants = None):
        if states == None:
            old_v = tf.zeros(shape = tf.shape(inputs))
            old_spike = tf.zeros(shape = tf.shape(inputs))
        else:
            old_v = states[0]
            old_spike = states[1]
        
        #compute the input currents
        i_in = inputs @ self.input_weights
        
        # update the voltage
        d = self.decay
        i_reset = self.thr * old_spike
        new_v = d * old_v + (1-d) * i_in - i_reset
        
        v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        
        new_state = (new_v, new_z)
        return new_state
    
class RSNN_alt(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr = 0.25, tau = 0.1, dampening_factor = 0.3):
        super().__init__()
        self.state_size = (num_neurons,num_neurons) #overwrite later
        self.decay = tau #tf.exp(-1 / tau) #alpha
        self.dampening_factor = dampening_factor
        self.thr = thr
        
        self.input_weights = None
        self.recurrent_weights = None
        
    def build(self, input_shape):
        self.state_size = (input_shape,input_shape)
        rand_init = tf.keras.initializers.glorot_uniform()
        
        # define the input weight variable
        self.input_weights = self.add_weight(
            shape = input_shape,
            initializer = rand_init,
            name = 'input_weights', trainable = True, dtype = tf.float32)
        
        # define the recurrent weight variable
        self.recurrent_weights = self.add_weight(
            shape = input_shape,
            initializer = rand_init,
            name = 'recurrent_weights', trainable = True, dtype = tf.float32)
        
        super().build(input_shape)
        
    def call(self, inputs, states, constants = None):
        old_v = states[0]
        old_spike = states[1]
        
        #compute the input currents
        i_in = tf.math.multiply(inputs, self.input_weights) +\
            tf.math.multiply(old_spike, self.recurrent_weights)
        
        # update the voltage
        d = self.decay #alpha
        i_reset = - self.thr * old_spike
        new_v = d * old_v + i_in + i_reset
        
        v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        
        new_state = (new_v, new_z)
        return (new_z, new_v), new_state
        

class RSNN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr = 0.03, tau = 0.5, dampening_factor = 0.3):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.state_size = (num_neurons, num_neurons)
        self.decay = tau #tf.exp(-1 / tau) #alpha
        
        self.dampening_factor = dampening_factor
        self.thr = thr
        
        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        
    def build(self, input_shape):
        if len(input_shape) == 5:
            n_in = input_shape[-4] * input_shape[-3] * input_shape[-2] * input_shape[-1]
        elif len(input_shape) == 4:
            n_in = input_shape[-3] * input_shape[-2] * input_shape[-1]
        elif len(input_shape) == 3:
            n_in = input_shape[-2] * input_shape[-1]
        elif len(input_shape) == 2:
            n_in = input_shape[-1]
        n = self.num_neurons
        
        rand_init = tf.keras.initializers.glorot_normal()
        
        # define the input weight variable
        self.input_weights = self.add_weight(
            shape = (n_in, n),
            initializer = rand_init,
            name = 'input_weights', trainable = True, dtype = tf.float32)
        
        # define the recurrent weight variable
        self.disconnect_mask = tf.cast(np.diag(np.ones(n, dtype = np.bool_)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape = (n,n),
            initializer = rand_init,
            name = 'recurrent_weights', trainable = True, dtype = tf.float32)
        
        super().build(input_shape)
        
    def get_recurrent_weights(self):
        w_rec = tf.where(self.disconnect_mask,
                         tf.zeros_like(self.recurrent_weights),
                         self.recurrent_weights)
        return w_rec
        
    def call(self, inputs, states, constants = None):
        old_v = states[0]
        old_spike = states[1]
        
        #compute the input currents
        w_rec = self.get_recurrent_weights()
        i_shape = tf.shape(inputs)
        if len(i_shape) == 4:
            inputs = tf.reshape(inputs, shape =
                                [i_shape[0], i_shape[-3]*i_shape[-2]*i_shape[-1]])
        i_in = inputs @ self.input_weights + old_spike @ w_rec
        
        # update the voltage
        d = self.decay #alpha
        i_reset = - self.thr * old_spike
        new_v = d * old_v + i_in + i_reset
        
        v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        
        new_state = (new_v, new_z)
        return (new_z, new_v), new_state
        
class LSNN(RSNN):
    def __init__(self, num_neurons, thr = 0.03, tau_adaptation = 5,
                 beta = 1.1, tau = 0.5, dampening_factor = 0.3,
                 stop_z_gradients = False):
        super().__init__(num_neurons, thr, tau, dampening_factor)
        
        self.state_size = (num_neurons, num_neurons, num_neurons, num_neurons)
        self.beta = beta
        self.decay_adaptation = tf.exp(-1 / tau_adaptation) #rho
        self.stop_z_gradients = stop_z_gradients
        
    def call(self, inputs, states, constants = None, stop_gradient = None):
        old_z = states[0]
        old_v = states[1]
        old_b = states[2]
        old_z_local = states[3]
        
        #stop local gradient of z in backward pass for autodiff of eprop
        use_stop_gradient = stop_gradient if stop_gradient is not None else self.stop_z_gradients
        if use_stop_gradient:
            old_z = tf.stop_gradient(old_z)
        
        #update b first
        da = self.decay_adaptation #rho
        new_b = da * old_b + old_z_local
        
        #update threshold
        thr = self.thr + self.beta * old_b
        
        #compute input currents
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights),
                         self.recurrent_weights)
        i_shape = tf.shape(inputs)
        if len(i_shape) == 4:
            inputs = tf.reshape(inputs, shape =
                              [i_shape[0], i_shape[-3]*i_shape[-2]*i_shape[-1]])
        
        #update input
        i_in = inputs @ self.input_weights + old_z @ w_rec
        
        #update the voltage
        d = self.decay #alpha
        i_reset = - self.thr * old_z
        new_v = d * old_v + i_in + i_reset
        
        v_scaled = (new_v - thr) / self.thr
        new_z       = SpikeFunction(v_scaled, self.dampening_factor)
        new_z_local = SpikeFunction(v_scaled, self.dampening_factor)
        
        new_states = (new_z, new_v, new_b, new_z_local)
        return (new_z, new_v, new_b, new_z_local, thr), new_states
    
    
# LSG output class
class LSG_output(tf.keras.layers.Layer):
    def __init__(self, n_neurons):
        self.num_neurons_out = n_neurons
        super(LSG_output, self).__init__()
        
    def build(self, input_shape):
        n = self.num_neurons_out
        rand_init = tf.keras.initializers.RandomNormal(mean = 0.,
                                                       stddev = 1. / np.sqrt(n))
        
        #initialize weights
        self.output_weights = self.add_weight(shape = (int(input_shape[-1]), n), 
                                            initializer = rand_init,
                                            name = 'output_weights', trainable = True)
        
    def call(self,z,decay):
        filtered_z = exp_convolve(z, decay)
        
        out = tf.einsum('btj,jk->btk', filtered_z, self.output_weights)
        return out

    
#define RSNN function without importing model
def LSNN_func(inputs, weights, params):
    from numpy import concatenate, zeros, ones
    thr = params.get('LN_thr')
    tau = params.get('LN_decay')
    dampening_factor = params.get('LN_dampening_factor')
    decay_adaptation = tf.exp(-1/params.get('LN_decay_adaptation'))
    beta = params.get('Beta_LN')
    n_reg_LN = params.get('n_reg_LN')
    n_ada_LN = params.get('n_ada_LN')
    
    #create beta
    beta = concatenate([zeros(n_reg_LN), beta * ones(n_ada_LN)])
    
    disconnect_mask = tf.cast(np.diag(np.ones(tf.shape(weights[1])[0],
                                              dtype = np.bool)), tf.bool)
    
    in_w = weights[0]
    rec_w = tf.where(disconnect_mask, tf.zeros_like(weights[1]),
                         weights[1])
    
    i_shape = tf.shape(inputs)
    old_b = zeros([i_shape[0],(n_reg_LN + n_ada_LN)])
    old_v = zeros([i_shape[0],(n_reg_LN + n_ada_LN)])
    old_z = zeros([i_shape[0],(n_reg_LN + n_ada_LN)])
    for ii in range(i_shape[1]):
        if len(i_shape) == 4:
            this_input = inputs[:,ii,:,:]
            this_input = tf.reshape(this_input,shape = 
                                    [i_shape[0], i_shape[-2]*i_shape[-1]])
        else:
            this_input = inputs[:,ii,:,:,:]
            this_input = tf.reshape(this_input,shape = 
                                    [i_shape[0], i_shape[-3]*i_shape[-2]*i_shape[-1]])
            
        #determine new_b
        new_b = decay_adaptation * old_b + old_z
                                          
        #update threshold
        A = thr + beta * old_b
        
        #compute input current
        i_in = this_input @ in_w + old_z @ rec_w
        
        #update voltage
        i_reset = - thr * old_z
        new_v = tau * old_v + i_in + i_reset
        
        #determine spike
        v_scaled = (new_v - A)/thr
        new_z = SpikeFunction(v_scaled, dampening_factor)
        
        if ii == 0:
            hid_spike = tf.expand_dims(new_z, axis = 1)
            hid_v = tf.expand_dims(new_v, axis = 1)
            hid_b = tf.expand_dims(new_b, axis = 1)
            
        else: 
            #concatenate outputs
            hid_spike = tf.concat([hid_spike,
                                   tf.expand_dims(new_z, axis = 1)], axis = 1)
            hid_v = tf.concat([hid_v,
                               tf.expand_dims(new_v, axis = 1)], axis = 1)
            hid_b = tf.concat([hid_b,
                               tf.expand_dims(new_b, axis = 1)], axis = 1)
                      
        #set old to new
        old_b = new_b
        old_v = new_v
        old_z = new_z
            
    return (hid_spike, hid_v, hid_b)


#create convolutional layers dict
def create_conv_layer(conv_layer_params, load_weights_path = ""):
    from tensorflow import convert_to_tensor, shape
    from tools_and_functions import load_weights
    from os import listdir
    from os.path import join
    from Models_and_Layers import LIF_activation
    
    window_size = conv_layer_params.get('window_size')
    if window_size is None:
        window_size = 1
    conv_filters = conv_layer_params.get('conv_filters')
    conv_kernel_size = conv_layer_params.get('conv_kernel_size')
    input_dims  = conv_layer_params.get('input_dims')
    if input_dims is None:
        input_dims = [19, 996, 1]
    num_conv_layers = len(conv_filters)
    
    #convolutional layers
    conv_dict = {}
    lif_dict = {}
    for layer_ind in range(num_conv_layers):
        if layer_ind == 0:
            #conv layer
            layer_input = tf.keras.Input(shape = (input_dims[0],
                                         input_dims[1]*window_size,
                                         input_dims[2]))
            dummy_input = convert_to_tensor(np.zeros([1,input_dims[0],
                                                      input_dims[1]*window_size,
                                                      input_dims[2]]))
        else:
            shape_lif_out = shape(lif(conv_out))
            layer_input = tf.keras.Input(shape = [shape_lif_out[2],
                                      shape_lif_out[3], shape_lif_out[4]])
            dummy_input = lif(conv_out)[1]
        
        #define conv layer
        conv_layer = tf.keras.layers.Conv2D(conv_filters[layer_ind],
                                            conv_kernel_size[layer_ind],
                                            activation = 'relu')
        #pooling layer
        max_pool = tf.keras.layers.MaxPooling2D(pool_size = (2))
        #batch norm layers
        batch_norm = tf.keras.layers.BatchNormalization()
        
        #build
        x = conv_layer(layer_input)
        x = max_pool(x)
        out = batch_norm(x)
        
        conv_layer = tf.keras.Model(inputs = layer_input, outputs = out, 
                                    name = "LN_conv_layer_" + f"{layer_ind + 1}")
        
        #initialize
        conv_out = conv_layer(dummy_input)
        
        #load in previous weights if applicable
        if load_weights_path != "":
            if len(listdir(load_weights_path)) > 0:
                #load in previous conv model weights
                conv_layer = load_weights(conv_layer, 
                                               join(load_weights_path,
                                                    'LN_conv_layer_' + f'{layer_ind + 1}'+ '_weights.npz'))
        
        #add to dictionary
        conv_dict["layer_" + f"{layer_ind + 1}"] = conv_layer
        
        #create LIF layer
        lif = LIF_activation()
        
        lif_dict['layer_' + f"{layer_ind + 1}"] = lif
    
    return conv_dict, lif_dict
    
    
#define forward pass function
#@tf.function
def conv_forward_pass_old(conv_layer_dict, lif_layer_dict, input_data, num_steps,
                      conv_params = None, training = False, 
                      experimental_relax_shapes = True):
    if conv_params is None:
        kernel_size = 0
        lif_1_neurons = 0
    else:
        kernel_size = conv_params["LN_conv_1_kernel_size"]
        lif_1_neurons = conv_params["LN_lif_1_n_out"]
    
    for step_ind in range(num_steps):
        if step_ind == 0:
            inputs = input_data
            keyCount = 0
            for key in conv_layer_dict:
                keyCount+=1
                conv_layer = conv_layer_dict[key]
                conv_output = conv_layer(inputs, training = training)
                lif_layer = lif_layer_dict[key]
                if keyCount == 1:
                    lif_out_1 = lif_layer(conv_output)
                    if lif_layer.description == "LIF layer":
                        inputs = tf.reshape(lif_out_1[1],shape = (tf.shape(lif_out_1[1])[0],2*kernel_size,
                                                            tf.cast((lif_1_neurons/(2*kernel_size)),dtype = tf.int32)))
                    else:
                        inputs = lif_out_1[1]
                else:
                    lif_out_2 = lif_layer(conv_output)
            out = tf.expand_dims(lif_out_2[1],1)
        else:
            inputs = input_data
            keyCount = 0
            for key in conv_layer_dict:
                keyCount+=1
                conv_layer = conv_layer_dict[key]
                conv_output = conv_layer(inputs,training = training)
                lif_layer = lif_layer_dict[key]
                if keyCount == 1:
                    lif_out_1 = lif_layer(conv_output,lif_out_1)
                    if lif_layer.description == "LIF layer":
                        inputs = tf.reshape(lif_out_1[1],shape = (tf.shape(lif_out_1[1])[0],2*kernel_size,
                                                            tf.cast((lif_1_neurons/(2*kernel_size)),dtype = tf.int32)))
                    else:
                        inputs = lif_out_1[1]
                else:
                    lif_out_2 = lif_layer(conv_output,lif_out_2)
            out = tf.concat([out,tf.expand_dims(lif_out_2[1],1)],1)
           
    return out

def conv_forward_pass(conv_layer_dict, lif_layer_dict, input_data, num_steps,
                      training = False, experimental_relax_shape = True):
    
    num_layers = len(lif_layer_dict)
    lif_outs = []
    for step_ind in range(num_steps):
        inputs = input_data
        for layer_ind in range(num_layers):
            #index layer by layer_ind
            conv_layer = conv_layer_dict.get('layer_' + f'{layer_ind+1}')
            lif_layer = lif_layer_dict.get('layer_' + f'{layer_ind+1}')
            #pass inputs to layer and save lif output
            conv_out = conv_layer(inputs, training = training)
            if step_ind == 0:
                lif_out = lif_layer(conv_out)
                lif_outs.append(lif_out)
            else:
                lif_out = lif_layer(conv_out, lif_outs[layer_ind])
                lif_outs[layer_ind] = lif_out
            if layer_ind < (num_layers - 1):
                #input to next layer is spike output of lif layer
                inputs = lif_out[1]
            else:
                #store output
                if step_ind == 0:
                    out = tf.expand_dims(lif_out[1], axis = 1)
                else:
                    out = tf.concat([out, tf.expand_dims(lif_out[1],1)],
                                    axis = 1)

    return out            


#create LN output layer
def create_LN_output_layer(output_version):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import BatchNormalization
    
    #create structure
    if output_version == 1:
        LN_output_layer = Sequential(name = "LN_output_layer_v1")
    elif output_version == 2:
        LN_output_layer = Sequential(name = "LN_output_layer_v2")
    
    #create layers
    flatten = Flatten()
    batch_norm = BatchNormalization()
    if output_version == 1:
        dense = Dense(1, activation = "sigmoid")
    elif output_version == 2:
        dense = Dense(2, activation = "linear")
    
    #build
    LN_output_layer.add(flatten)
    LN_output_layer.add(batch_norm)
    LN_output_layer.add(dense)
    
    return LN_output_layer
    

#LN output function
def get_output(inputs, output_layer, ol_training):
    
    #do different things based on layer name
    if output_layer.name == "LN_output_layer_v2":
        #get input shape
        i_shape = tf.shape(inputs)
        
        #add spiking layer to end
        LN_out_spiking = LIF_activation()
        
        #define storage
        out_spike  = tf.constant(0)
        out_mem    = tf.constant(0)
        out_pred   = tf.constant(0)
        output_old = tf.constant(0)
        
        #get outputs
        for step in range(i_shape[1]):
            this_input = inputs[:,step,:]       
            #pass to output layer    
            spiking_in = output_layer(this_input, training = ol_training)
            
            #get spike out and organize outputs
            if step == 0:
                output = LN_out_spiking(spiking_in)
                output_old = tf.identity(output)
                out_spike = tf.expand_dims(output[1], axis = 1)
                out_mem = tf.expand_dims(output[0], axis = 1) 
            else:
                output = LN_out_spiking(spiking_in, output_old)
                output_old = tf.identity(output)
                out_spike = tf.concat([out_spike, tf.expand_dims(output[1], axis = 1)], axis = 1)
                out_mem = tf.concat([out_mem, tf.expand_dims(output[0], axis = 1)], axis = 1)
            
        #determine predicted class labels
        for out_spike_ind in range(tf.shape(out_spike)[0]): 
            if sum(out_spike[out_spike_ind,:,0]) >= sum(out_spike[out_spike_ind,:,1]):
                pred = tf.one_hot(indices = 0, depth = 2, dtype = tf.int64)
            else:
                pred = tf.one_hot(indices = 1, depth = 2, dtype = tf.int64)
            if out_spike_ind == 0:
                out_pred = tf.expand_dims(pred, axis = 0)
            else:
                out_pred = tf.concat([out_pred, tf.expand_dims(pred, axis = 0)], axis = 0)
                
        total_outs = (out_spike, out_mem, out_pred)
        
    elif output_layer.name == "LN_output_layer_v1":
        output = output_layer(inputs, training = ol_training)
        out_pred = tf.cast(tf.math.round(output), dtype = tf.int64)
        
        total_outs = (output, out_pred)
        
    elif output_layer.name == "LSG_output_layer":
        #get input shape
        i_shape = tf.shape(inputs)
        
        #get outputs
        for step in range(i_shape[1]):
            this_input = inputs[:, step, :]
            
            #pass to output layer
            output = output_layer(this_input, training = ol_training)
            
            #organize_outputs
            if step == 0:
                learn_sig = tf.expand_dims(output, axis = 1)
            else:
                learn_sig = tf.concat([learn_sig, tf.expand_dims(output, axis = 1)], axis = 1)
        
        total_outs = learn_sig
        
    return total_outs


#create LN model
def create_LN_model(LN_params, load_weights_path = "", dummy_in = None, 
                    output_version = 1):
    from tools_and_functions import load_weights
    from numpy import concatenate
    from os import listdir
    from os.path import join
    
    #distribute into variables
    LN_decay_adaptation = LN_params.get('LN_decay_adaptation')
    LN_thr = LN_params.get('LN_thr')
    LN_decay = LN_params.get('LN_decay')
    LN_dampening_factor = LN_params.get('LN_dampening_factor')
    n_reg_LN = LN_params.get('n_reg_LN')
    n_ada_LN = LN_params.get('n_ada_LN')
    LN_Beta = LN_params.get('LN_Beta')
    stop_z_gradients = LN_params.get('stop_z_gradients')
    if stop_z_gradients is None:
        stop_z_gradients = False
    use_random_LSNN_weights = LN_params.get('use_random_LSNN_weights')
    if use_random_LSNN_weights is None:
        use_random_LSNN_weights = False
    
    #create Hidden layer
    beta_LN = concatenate([np.zeros(n_reg_LN), LN_Beta * np.ones(n_ada_LN)])
    cell_LN = LSNN(num_neurons = n_reg_LN + n_ada_LN, beta = beta_LN,
                   thr = LN_thr, tau_adaptation = LN_decay_adaptation,
                   tau = LN_decay, dampening_factor = LN_dampening_factor,
                   stop_z_gradients = stop_z_gradients)
    Rec_LN = tf.keras.layers.RNN(cell_LN, return_sequences = True)

    #create Output layer
    LN_output_layer = create_LN_output_layer(output_version)
    
    #initialize
    if dummy_in is not None:
        dummy_Rec_LN_outs = Rec_LN(dummy_in)
        dummy_out = get_output(dummy_Rec_LN_outs[0], LN_output_layer,
                               ol_training = True)
        
    #load in previous weights
    if load_weights_path != "":
        if len(listdir(load_weights_path)) > 0:
            if use_random_LSNN_weights is False:
                Rec_LN = load_weights(Rec_LN, join(load_weights_path,
                                                   'LN_Hidden_weights.npz'))
            LN_output_layer = load_weights(LN_output_layer,
                                           join(load_weights_path,
                                                                 'LN_output_weights.npz'))
            
    return Rec_LN, LN_output_layer, dummy_Rec_LN_outs


#create LSG model
def create_LSG_model(LSG_params, load_weights_path = "", dummy_in = None):
    from tools_and_functions import load_weights
    from numpy import concatenate
    from os import listdir
    from os.path import join
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import BatchNormalization
    
    #distribute to variables
    n_reg_LSG = LSG_params.get("n_reg_LSG")
    n_ada_LSG = LSG_params.get("n_ada_LSG")
    n_out_LSG = LSG_params.get("n_out_LSG")
    beta_LSG = LSG_params.get("beta_LSG")
    
    #hidden layer
    beta_LSG = concatenate([np.zeros(n_reg_LSG), beta_LSG * np.ones(n_ada_LSG)])
    cell_LSG = LSNN(n_reg_LSG + n_ada_LSG, beta = beta_LSG)
    Hidden_LSG = tf.keras.layers.RNN(cell_LSG, return_sequences = True)
    
    #output layer
    #create structure
    LSG_output_layer = Sequential(name = "LSG_output_layer")
    
    #create layers
    flatten = Flatten()
    dense_LSG = Dense(n_out_LSG,activation = "linear")
    batch_norm_LSG = BatchNormalization()
    
    #build
    LSG_output_layer.add(flatten)
    LSG_output_layer.add(batch_norm_LSG)
    LSG_output_layer.add(dense_LSG)
    
    #initialize
    if dummy_in is not None:
        dummy_LSG_hidden_out = Hidden_LSG(dummy_in)
        LSG_out = get_output(dummy_LSG_hidden_out[0], LSG_output_layer, ol_training = True)
        
    #load in previous weights
    if load_weights_path != "":
        if len(listdir(load_weights_path)) > 0:
            Hidden_LSG = load_weights(Hidden_LSG,
                                      join(load_weights_path,
                                                       'Hidden_LSG_weights.npz'))
            LSG_output_layer = load_weights(LSG_output_layer,
                                            join(load_weights_path,
                                                                   'LSG_out_weights.npz'))
            
    return Hidden_LSG, LSG_output_layer
        
    



    