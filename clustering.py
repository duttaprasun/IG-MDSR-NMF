import numpy as np
import xlwt
from warnings import simplefilter
from sklearn import cluster, metrics, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.exceptions import ConvergenceWarning
from fcmeans import FCM
import warnings
import pandas as pd

def clustering(sheet, i, fn, x, y, nc, nme, factor):
    m, n = x.shape
    '''
    if nme == "GastrointestinalLesionsInRegularColonoscopy" :
        epsiln = 0.05
    elif nme == "StudentPerformance" :
        epsiln = 1.0
    else:
        epsiln = 0.5

    default_base = {'quantile': .3,
                'eps': epsiln,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': nc,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}
    params = default_base.copy()
    bandwidth = cluster.estimate_bandwidth(x, quantile=params['quantile'])
    if bandwidth == 0.0 :
        bandwidth = 2.0
    connectivity = kneighbors_graph(x, n_neighbors=params['n_neighbors'], include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    '''
    two_means = cluster.MiniBatchKMeans(n_clusters=nc)
    birch = cluster.Birch(n_clusters=nc, threshold = 0.01)
    gmm = mixture.GaussianMixture(init_params='random', n_components=2)
    fcm = FCM(n_clusters=nc)
    
    clustering_algorithms = (
        ('Mini Batch k-Means', two_means),
        ('BIRCH', birch),
        ('Gaussian Mixture Models', gmm),
        ('Fuzzy c-Means', fcm),
    )
    
    for name, algorithm in clustering_algorithms:
        try:
            print("----------", name, "----------", fn, "----------")
            j = 0
            sheet.write(i, j, name)
            j = 1
            sheet.write(i, j, fn)
            flag = 0
            try :
                algorithm.fit(x)
            except Exception as e :
                print("error/exception in fit(x): ", e)
                flag = 1
            if flag == 0 :
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                elif hasattr(algorithm, 'u'):
                    y_pred = algorithm.u.argmax(axis=1)
                else:
                    y_pred = algorithm.predict(x)
                npcl = np.unique(y_pred).size # number of predicted cluster labels
                print("npcl: ", npcl)
                j = 2
                if npcl == 1 or npcl == m :
                    ars = "npcl"
                    sheet.write(i, j, ars)
                    j += 1
                    ji = "npcl"
                    sheet.write(i, j, ji)
                    j += 1
                    nmis = "npcl"
                    sheet.write(i, j, nmis)
                    j += 1
                    amis = "npcl"
                    sheet.write(i, j, amis)
                    j += 1
                else :
                    ars = metrics.adjusted_rand_score(y, y_pred)
                    print("adjusted rand score: ", ars)
                    sheet.write(i, j, round(ars, 6))
                    j += 1
                    ji = metrics.jaccard_score(y, y_pred, average = 'weighted')
                    print("jaccard index: ", ji)
                    sheet.write(i, j, round(ji, 6))
                    j += 1
                    nmis = metrics.normalized_mutual_info_score(y, y_pred)
                    print("normalized mutual info score: ", nmis)
                    sheet.write(i, j, round(nmis, 6))
                    j += 1
                    amis = metrics.adjusted_mutual_info_score(y, y_pred)
                    print("adjusted mutual info score: ", amis)
                    sheet.write(i, j, round(amis, 6))
                    j += 1
        except Exception as e:
            print("other error: ", e)
        i += 1

def init(model_choice, nme, d_original, d_class, dc, factor, red_dim):
    print(" nme:", nme, "\n", "dc:", dc, "\n", "red_dim:", red_dim, "\n", "factor:", factor)
    m, n = d_original.shape
    fn1 = nme + "/" + nme + "_" + model_choice + "_b_" + str(m) + "_" + str(red_dim) + "_" + str(factor) + ".txt"
    f1 = open(fn1, "r")
    d1 = np.genfromtxt(f1, delimiter = ',')
    simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    nc = np.unique(d_class).size
    print("original number of clusters: ", nc)
    book = xlwt.Workbook(encoding="utf-8")
    sheet = book.add_sheet("clustering_performance")
    j = 2
    sheet.write(0, j, "external evaluation")
    sheet.write(1, j, "adjusted rand score")
    sheet.write(2, j, "(-1 to +1)")
    sheet.write(3, j, "(best value: +1)")
    j += 1
    sheet.write(1, j, "jaccard index")
    sheet.write(2, j, "(0 to +1)")
    sheet.write(3, j, "(best value: +1)")
    j += 1
    sheet.write(1, j, "normalized mutual info score")
    sheet.write(2, j, "(0 to +1)")
    sheet.write(3, j, "(best value: +1)")
    j += 1
    sheet.write(1, j, "adjusted mutual info score")
    sheet.write(2, j, "upperlimited by +1")
    sheet.write(3, j, "(best value: +1)")
    j += 1
    i = 4
    clustering(sheet, i, model_choice, d1, d_class, nc, nme, factor)   
    book.save(nme + "/" + nme + "_cluster_performance_" + model_choice + "_" + str(dc) + "_" + str(red_dim) + "_" + str(factor) + ".xls")

def readData(choice):
    print("data choice", choice)
    if choice == 1 :
        nme = "GLRC"
        fn = nme + "/" + "data.txt"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[1:, 2:701]
        d_class = d[1:, 701:]
    elif choice == 2 :
        nme = "ONP"
        fn = nme + "/" + nme + ".csv"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[1:, 1:60]
        d_class = d[1:, 61:]
    elif choice == 3 :
        nme = "PDC"
        fn = nme + "/" + "pd_speech_features.csv"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[2:, 1:754]
        d_class = d[2:, 754:]
    elif choice == 4 :
        nme = "SP"
        fn = nme + "/" + "student_mat.csv"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[1:, :32]
        d_class = d[1:, 33:]
    elif choice == 5 :
        nme = "MovieLens"
        fn = nme + "/" + "data_labled.csv"
        d = pd.read_csv(fn)
        d_original = d.iloc[:,3:-3].to_numpy()
        d_class = d["gender"].to_numpy()
    else :
        print("wrong cmd line i/p")
    if choice != 5 :
        f.close()
    print("original data shape:\t", d.shape)
    d_class = np.ravel(d_class)
    d_class = d_class.astype(int)
    #print(d_original.shape, d_original[0], d_class.shape, d_class[0])
    return (nme, d_original, d_class)

def main(model_choice, data_choice, f):
    data_choice = int(data_choice)
    nme, d_original, d_class = readData(data_choice)
    dr, dc = d_original.shape
    red_dim = int(f * dc)
    init(model_choice, nme, d_original, d_class, dc, f, red_dim)

if __name__ == '__main__':
    model_choice = "igmdsrnmf" # igmdsrrnmf
    data_choice = 1 # 1: GastrointestinalLesionsInRegularColonoscopy, 2: OnlineNewsPopularity, 3: ParkinsonsDiseaseClassification, 4: StudentPerformance, 5: MovieLens
    f = 0.25
    main(model_choice, data_choice, f)