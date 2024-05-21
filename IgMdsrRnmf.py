import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.models import Model, load_model
from keras import regularizers
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

class IgMdsrRnmf:
    def __init__(self, nme, x, dc, factor, red_dim):
        self.nme = nme
        self.x = x
        self.dc = dc
        self.factor = factor
        self.r = red_dim
        self.m, self.n = self.x.shape
        print("m: ", self.m, "\tn: ", self.n, "\tr: ", self.r)
    
    def model(self):
        data = self.x
        train_data, test_data = data, data
        print(train_data.shape, test_data.shape)
        inp = Input(shape=(self.n,), name="inp")
        h1 = Dense(units = self.r[0],
                   activation = 'sigmoid',
                   use_bias=False,
                   kernel_initializer = tf.compat.v1.keras.initializers.glorot_normal(),
                   bias_initializer = tf.keras.initializers.Constant(0.0),
                   name="h1"
                   )(inp)
        h2 = Dense(units = self.r[1],
                   activation = 'sigmoid',
                   use_bias=False,
                   kernel_initializer = tf.compat.v1.keras.initializers.glorot_normal(),
                   bias_initializer = tf.keras.initializers.Constant(0.0),
                   name="h2"
                   )(concatenate([inp, h1]))
        h3 = Dense(units = self.r[2],
                   activation = 'sigmoid',
                   use_bias=False,
                   kernel_initializer = tf.compat.v1.keras.initializers.glorot_normal(),
                   bias_initializer = tf.keras.initializers.Constant(0.0),
                   name="h3"
                   )(concatenate([inp, h2]))
        oup = Dense(units = self.n,
                   activation = 'relu',
                   use_bias=False,
                   kernel_initializer = tf.compat.v1.keras.initializers.glorot_normal(),
                   bias_initializer = tf.keras.initializers.Constant(0.0),
                   name="oup"
                   )(h3)
        model = Model(inputs=inp, outputs=oup)
        print(model.summary())
        f = self.nme + "/" + self.nme + "_igmdsrrnmf_best_model_" + str(self.dc) + "_" + str(self.r[2]) + "_" + str(self.factor[2]) + ".h5"
        callback = [EarlyStopping(monitor='loss', mode='auto', min_delta=0.0000000000001, patience=5),
                     ModelCheckpoint(filepath=f, monitor='accuracy', verbose=0, save_best_only=True, mode='auto')]
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(train_data, train_data, batch_size=None, verbose=0, 
                            callbacks=callback, validation_split=0.0, shuffle=True, epochs=9)#99999)
        model = tf.keras.models.load_model(f)
        _, accuracy = model.evaluate(train_data, train_data, verbose=0)
        print('Train Accuracy: %.2f' % (accuracy*100))
        _, accuracy = model.evaluate(test_data, test_data, verbose=0)
        print('Test Accuracy: %.2f' % (accuracy*100))
        cost_array = history.history['loss']
        print("len(cost_array): ", len(cost_array))
        print("cost_array: ", cost_array)
        latent_layer = model.get_layer('h3')
        latent_model = tf.keras.models.Model(inputs=model.input, outputs=latent_layer.output)
        latent_output = latent_model.predict(test_data)
        print("b.shape: ", latent_output.shape)
        print("b", latent_output)
        output_layer = model.get_layer('oup')
        output_weights = output_layer.get_weights()
        print("w.shape: ", output_weights[0].shape)
        print("w", output_weights)
        o1, o2 = output_weights[0].shape
        print(o1,o2)
        fname1 = self.nme + "/" + self.nme + "_igmdsrrnmf_b_" + str(self.m) + "_" + str(self.r[2]) + "_" + str(self.factor[2]) + ".txt"
        f1 = open(fname1, "w")
        np.savetxt(f1, latent_output, fmt = '%.17f', delimiter = ',')
        f1.close()
        fname2 = self.nme + "/" + self.nme + "_igmdsrrnmf_w_" + str(self.r[2]) + "_" + str(self.dc) + "_" + str(self.factor[2]) + ".txt"
        f2 = open(fname2, "w")
        np.savetxt(f2, output_weights[0], fmt = '%.17f', delimiter = ',')
        f2.close()
        fname3 = self.nme + "/" + self.nme + "_igmdsrrnmf_cost_array_" + str(self.dc) + "_" + str(self.r[2]) + "_" + str(self.factor[2]) + ".txt"
        f3 = open(fname3, "w")
        np.savetxt(f3, cost_array, fmt = '%.17f', delimiter = ',')
        f3.close()

class General:
    def readData(self, choice):
        print("data choice", choice)
        if choice == 1 :
            nme = "GLRC"
            fn = nme + "/" + "data.txt"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[1:, 2:701]
        elif choice == 2 :
            nme = "ONP"
            fn = nme + "/" + "OnlineNewsPopularity.csv"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[1:, 1:60]
        elif choice == 3 :
            nme = "PDC"
            fn = nme + "/" + "pd_speech_features.csv"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[2:, 1:754]
        elif choice == 4 :
            nme = "SP"
            fn = nme + "/" + "student_mat.csv"
            f = open(fn, "r")
            d1 = np.genfromtxt(f, delimiter = ',')
            d2 = d1[1:, :32]
        elif choice == 5 :
            nme = "MovieLens"
            fn = nme + "/" + "data_labled.csv"
            df = pd.read_csv(fn)
            d1 = df
            d2 = df.iloc[:,2:-3].to_numpy()
        else :
            print("wrong cmd line i/p")
        print("original data shape:\t", d1.shape, "\nreduced data shape:\t", d2.shape)
        return (nme, d2)
    
    def z_score_normalize(self, matrx):
        r, c = matrx.shape
        mn = np.mean(matrx, axis = 0)
        sd = np.std(matrx, axis = 0)
        for j in range(c):
            if (sd[j] == 0):
                sd[j] = 0.01
            matrx[:, j] = (matrx[:, j] - mn[j]) / sd[j]
        return matrx
    
    def min_max_normalize(self, matrx):
        r, c = matrx.shape
        mini = np.amin(matrx, axis = 0)
        maxi = np.amax(matrx, axis = 0)
        new_min = 0.0
        new_max = 1.0
        for j in range(c):
            scale = maxi[j] - mini[j]
            matrx[:, j] = (((matrx[:, j] - mini[j]) / scale) * (new_max - new_min)) + new_min
        return matrx
    
    def col_double(self, d1):
        d2 = d1.copy()
        d3 = d1.copy()
        d2[d2 < 0] = 0
        d3[d3 > 0] = 0
        d3 = np.absolute(d3)
        d4 = np.append(d2, d3, axis = 1)
        return d4
    
    def add_noise(self, matrx):
        r, c = matrx.shape
        mini = np.amin(matrx, axis = 0)
        maxi = np.amax(matrx, axis = 0)
        for j in range(c):
            if (maxi[j] == mini[j]):
                print("adding noise: ", j, maxi[j])
                matrx[:, j] = matrx[:, j] + np.random.normal(0, .1, matrx[:, j].size)
        return matrx
    
    def preprocess(self, nme, d):
        d1 = self.z_score_normalize(d)
        d2 = self.col_double(d1)
        print("processed data shape:\t", d2.shape)
        x = d2[:, :]
        if (x.min() < 0):
            print("Input matrix elements can not be negative!!!")
        print("working data shape:\t", x.shape)
        fn = nme + "/" + nme + "_processed_data.txt"
        f = open(fn, "w")
        np.savetxt(f, x, fmt = '%.17f', delimiter = ',')
        f.close()
        return x

def main(data_choice, factor):
    data_choice = int(data_choice)
    factor = float(factor)
    obj1 = General()
    nme, d = obj1.readData(data_choice)
    dr, dc = d.shape
    x = obj1.preprocess(nme, d)
    target_factor3 = factor
    target_factor2 = 1.0 - 2*((1.0-target_factor3)/3.0)
    target_factor1 = 1.0 - (1.0-target_factor3)/3.0
    factors = np.array([target_factor1, target_factor2, target_factor3])
    red_dim = dc * factors
    red_dim = red_dim.astype(int)
    print("factors : ", factors, ", red_dim : ", red_dim)
    obj = IgMdsrRnmf(nme, x, dc, factors, red_dim)
    obj.model()

if __name__ == '__main__':
    data_choice = 1 # 1: GastrointestinalLesionsInRegularColonoscopy, 2: OnlineNewsPopularity, 3: ParkinsonsDiseaseClassification, 4: StudentPerformance, 5: MovieLens
    f = 0.25
    main(data_choice, f)