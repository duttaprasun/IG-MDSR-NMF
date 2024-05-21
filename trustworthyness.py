import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

def readData(choice):
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
    d_original = d2
    print(d_original.shape)
    return (nme, d_original)

def main(model_choice, data_choice, factor):
    data_choice = int(data_choice)
    factor = float(factor)
    nme, d_original = readData(data_choice)
    m, n = d_original.shape
    red_dim = int(n * factor)
    print("data_choice: ", data_choice, ", factor: ", factor, ", red_dim: ", red_dim, "\n")
    fn1 = nme + "/" + nme + "_" + model_choice + "_b_" + str(m) + "_" + str(red_dim) + "_" + str(factor) + ".txt"
    f1 = open(fn1, "r")
    d_reduced = np.genfromtxt(f1, delimiter = ',')
    t = trustworthiness(d_original, d_reduced)
    print("trustworthiness: ", t)

if __name__ == '__main__':
    model_choice = "igmdsrnmf" # igmdsrrnmf
    data_choice = 1 # 1: GastrointestinalLesionsInRegularColonoscopy, 2: OnlineNewsPopularity, 3: ParkinsonsDiseaseClassification, 4: StudentPerformance, 5: MovieLens
    f = 0.25
    main(model_choice, data_choice, f)