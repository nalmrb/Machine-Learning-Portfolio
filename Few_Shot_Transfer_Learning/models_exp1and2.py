# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 08:36:42 2023

@author: nalmrb
"""

def compile_model(net):
    
    import tensorflow as tf
    import keras
    
    #define custom metrics
    class TPR(tf.keras.metrics.Metric):
        
        def __init__(self, name = 'TPR', **kwargs):
            super().__init__(name=name, **kwargs)
            self.true_positives = self.add_weight(name = 'true_p',
                                                      initializer = 'zeros')
            self.total_positives = self.add_weight(name = 'total_p',
                                                   initializer = 'zeros')
            self.pred_class_len = self.add_weight(name = 'pred_class_len',
                                                  initializer = 'zeros')
            self.y_true_len = self.add_weight(name = 'y_true_len',
                                              initializer = 'zeros')
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.bool)
            self.y_true_len.assign(tf.cast(tf.size(y_true), self.dtype))
            pred_class = tf.cast(tf.round(tf.nn.sigmoid(y_pred)[:,1]), tf.bool)
            self.pred_class_len.assign(tf.cast(tf.size(pred_class), self.dtype))
            
            true_pos_values = tf.zeros(shape = [], dtype = self.dtype)
            for idx in range(len(y_true)):
                if tf.logical_and(tf.equal(y_true[idx], True),
                                  tf.equal(pred_class[idx], True)):
                    true_pos_values = tf.add(true_pos_values,
                                             tf.constant(1, self.dtype))
                    
            total_pos_values = tf.reduce_sum(tf.cast(tf.equal(y_true, True),
                                              self.dtype))
            
            self.true_positives.assign_add(true_pos_values)
            self.total_positives.assign_add(total_pos_values)
            
        def result(self):
            self.TPR = tf.divide(self.true_positives, self.total_positives)
            return self.TPR
        
        def reset_state(self):
            self.true_positives.assign(0)
            self.total_positives.assign(0)
            self.pred_class_len.assign(0)
                   
    class TNR(tf.keras.metrics.Metric):
        
        def __init__(self, name = 'TNR', **kwargs):
            super().__init__(name=name, **kwargs)
            self.true_negatives = self.add_weight(name = 'true_n',
                                                  initializer = 'zeros')
            self.total_negatives = self.add_weight(name = 'total_n',
                                                   initializer = 'zeros')
            self.pred_class_len = self.add_weight(name = 'pred_class_len',
                                                  initializer = 'zeros')
            self.y_true_len = self.add_weight(name = 'y_true_len',
                                              initializer = 'zeros')
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.bool)
            self.y_true_len.assign(tf.cast(tf.size(y_true), self.dtype))
            pred_class = tf.cast(tf.round(tf.nn.sigmoid(y_pred)[:,1]), tf.bool)
            self.pred_class_len.assign(tf.cast(tf.size(pred_class), self.dtype))
            
            true_neg_values = tf.zeros(shape = [], dtype = self.dtype)
            for idx in range(len(y_true)):
                if tf.logical_and(tf.equal(y_true[idx], False),
                                  tf.equal(pred_class[idx], False)):
                    true_neg_values = tf.add(true_neg_values,
                                             tf.constant(1, self.dtype))
            
            total_neg_values = tf.reduce_sum(tf.cast(tf.equal(y_true, False), 
                                                     self.dtype))
            
            self.true_negatives.assign_add(true_neg_values)
            self.total_negatives.assign_add(total_neg_values)
            
        def result(self):
            self.TNR = tf.divide(self.true_negatives, self.total_negatives)
            return self.TNR
        
        def reset_state(self):
            self.true_negatives.assign(0)
            self.total_negatives.assign(0)
            self.pred_class_len.assign(0)
            
    TPR_object = TPR()
    TNR_object = TNR()
    custom_metrics = {'TPR':TPR, 'TNR':TNR}        
    
    #compile network
    net.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                optimizer = 'adam',
                metrics = ['accuracy',
                            TPR_object,
                            TNR_object])
    
    return net, custom_metrics


def build_and_compile_CNN(dummy_input_shape):
    
    import tensorflow as tf
    
    #build and initialize network
    net = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape = [19,996,1], filters = 12, kernel_size = 5), 
        tf.keras.layers.MaxPooling2D(pool_size = 2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters = 64, kernel_size = 3),
        tf.keras.layers.MaxPooling2D(pool_size = 2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2)])
    
    dummy_input = tf.zeros(shape = dummy_input_shape)
    net_out = net(dummy_input)
    
    #check for akida compatibility
    from cnn2snn import check_model_compatibility
    print('Model compatible for Akida conversion:', 
          check_model_compatibility(net))
    
    #compile model
    from models import compile_model
    net, custom_metrics = compile_model(net)
    
    return net, custom_metrics



    