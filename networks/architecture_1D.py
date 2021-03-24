# General network architecture for a 1D polynomial Poincare map

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Conjugacy Network
class Conjugacy(Model):
    def __init__(self,
						width = 100,
						size_x = 1,
						activation = 'selu',
						degree = 2,
						steps = 1,
						numblks_in = 1,
						numblks_out = 1,
						coeff1 = 3.5,
						coeff2 = -3.5,
						coeff3 = 0,
						coeff4 = 0.0,
						coeff5 = 0.0,
						l1reg = 1e-15,
						l2reg = 1e-15,
						**kwargs):
        """
     	Create network architecture for 1D polynomial Poincare mappings
     
     	Inputs: 
     		width -- the number of units in each hidden layer
     		size_x -- number of dimensions of the input data x_n
     		activation -- activation function used for each hidden layer
     		degree -- degree of the latent polynomial g(y)
     		steps -- number of iterations in the future the network will use to compute the loss
     		numblks_in --  number of hidden layers used to construct the conjugacy h(x)
     		numblks_out -- number of hidden layers used to construct the inverse of the conjugacy h^-1(y)
     		l1reg = L^1 regularization value
     		l2reg = L^2 regularization value
            **kwargs -- additional keyword arguments. 
        '"""          	    	
        super(Conjugacy, self).__init__()
        self.width = width 
        self.size_x = size_x
        self.activation = activation
        self.degree = degree
        self.steps = steps
        self.numblks_in = numblks_in
        self.numblks_out = numblks_out
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.coeff3 = coeff3
        self.coeff4 = coeff4
        self.coeff5 = coeff5
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)
   
        # Latent mapping variables
        #		g(y) = c1*y + c2*y^2 + c3*y^3 + c4*y^4 + c5*y^5
    
        # Minimum degree is 2
        self.c1 = tf.Variable(self.coeff1, trainable = True)
        self.c2 = tf.Variable(self.coeff2, trainable = True)
    
        # Cubic terms
        if self.degree >= 3:
        	self.c3 = tf.Variable(self.coeff3, trainable = True)
        else:
        	self.c3 = tf.Variable(0.0, trainable = False)
    
    	# Quartic terms
        if self.degree >= 4:
        	self.c4 = tf.Variable(self.coeff4, trainable = True)
        else:
        	self.c4 = tf.Variable(0.0, trainable = False)
    
        # Quintic terms
        if self.degree >= 5:
        	self.c5 = tf.Variable(self.coeff5, trainable = True)
        else:
        	self.c5 = tf.Variable(0.0, trainable = False)
    
        # Encoder function (homeomorphism)
        self.encoder = tf.keras.Sequential()
    
        for n in range(self.numblks_in): #adding more layers
            self.encoder.add(layers.Dense(self.width, activation=self.activation, 
                                      kernel_initializer=self.initializer, 
                                      kernel_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg), 
                                      bias_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg)))
    
        self.encoder.add(layers.Dense(1, activation=self.activation, 
                                  kernel_initializer=self.initializer,
                                  kernel_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg), 
                                  bias_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg)))

        # Decoder function (inverse homeomorphism)
        self.decoder = tf.keras.Sequential()
    
        for n in range(self.numblks_out): #adding more layers
            self.decoder.add(layers.Dense(self.width, activation=self.activation, 
                                      kernel_initializer=self.initializer, 
                                      kernel_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg), 
                                      bias_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg)))
    
        self.decoder.add(tf.keras.layers.Dense(size_x, activation= self.activation, 
                                           kernel_initializer=self.initializer, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg), 
                                           bias_regularizer=tf.keras.regularizers.L1L2(self.l1reg, self.l2reg)))
    
    
    def call(self, x):
    	encoded = self.encoder(x[0])
    	encoded_p1 = self.c1*encoded + self.c2*tf.square(encoded) + self.c3*tf.math.multiply(tf.square(encoded),encoded) + self.c4*tf.math.multiply(tf.square(encoded),tf.square(encoded)) + self.c5*tf.math.multiply(tf.math.multiply(tf.square(encoded),tf.square(encoded)),encoded)
    	decoded = self.decoder(encoded_p1)
    	# Conjugacy loss
    	x_recon = self.decoder(encoded)
    	self.add_loss(tf.reduce_mean(tf.math.square(x[0] - x_recon)))
    	
    	# Build stepping loss components
    	yn = encoded
    	for s in range(self.steps):
    		ynp1 = self.encoder(x[s+1])
    		
    		# Iteration loss in y
    		y_step = self.c1*yn + self.c2*tf.square(yn) + self.c3*tf.math.multiply(tf.square(yn),yn) + self.c4*tf.math.multiply(tf.square(yn),tf.square(yn)) + self.c5*tf.math.multiply(tf.math.multiply(tf.square(yn),tf.square(yn)),yn) 
    		self.add_loss((tf.reduce_mean(tf.math.square(ynp1 - y_step)))/self.steps)
    		
    		# Iteration loss in x
    		xnp1 = self.decoder(y_step)
    		self.add_loss((tf.reduce_mean(tf.math.square(x[s+1] - xnp1)))/self.steps)
    		
    		yn = y_step
    		
    		return decoded