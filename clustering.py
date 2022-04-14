import pandas
import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score,silhouette_score

class database:
    def __init__(self,config):
        self.config = config
        self.db = pandas.read_csv(self.config['path_db'],sep = ';',engine = 'python',header = None)
        self.db.columns = ['id','date','indicator','value']

    def compute_db(self,x):
        x = x.rename(columns = {'value':str(x['indicator'].iloc[0])})
        self.res = self.res.merge(x,how = 'left',on = ['id','date'])
        self.res = self.res.drop(['indicator'],axis = 1)

    def build_db(self):
        self.res = self.db[['id','date']]
        self.res = self.res.drop_duplicates()
        self.db.groupby(['indicator']).apply(lambda x: self.compute_db(x))

    def write(self):
        self.res.to_csv(self.config['folder'] + 'db.csv',sep = ';',index = False)

    def run(self):
        self.build_db()
        self.write()

class cleaning:
    def __init__(self,config):
        self.config = config
        self.db = pandas.read_csv(self.config['path_db'],sep = ';',engine = 'python')

    def select_db(self):
        self.db = self.db[self.config['col_keep']]

    def sel_na(self):
        for c in self.db.columns:
            sel = numpy.array(self.db[c].astype(str) == 'nan')
            print(sum(sel) / sel.shape[0])
            self.db = self.db[sel == False]

    def run(self):
        self.select_db()
        self.sel_na()
        self.boite_moustache()
        self.write()

    def write(self):
        self.db.to_csv(self.config['folder'] + 'db_clean.csv',sep = ';',index = False)

    def boite_moustache(self):
        Q1 = numpy.quantile(self.db,q = 0.25,axis = 0)
        Q2 = numpy.quantile(self.db,q = 0.5,axis = 0)
        Q3 = numpy.quantile(self.db,q = 0.75,axis = 0)
        i = 0
        for c in self.db.columns:
            sel = numpy.array(self.db[c] > Q2[i] + 1.5 * Q3[i]) + numpy.array(self.db[c] < Q2[i] - 1.5 * Q1[i])
            sel = sel.astype(bool)
            self.db = self.db[sel == False]
            i += 1

class clustering:
    def __init__(self,config):
        self.config = config
        self.db = pandas.read_csv(self.config['path_db'],sep = ';',engine = 'python')
        self.dist = pandas.DataFrame()
        self.distrib = pandas.DataFrame()
        if self.config['sample']:
            sel = numpy.random.uniform(low = 0,high = 1,size = self.db.shape[0])
            sel = sel < self.config['pct']
            self.db = self.db[sel]


    def compute_db(self):
        self.db = self.db[self.config['col_expert']]
        self.db = self.db.fillna(0)

    def compute_kmeans(self,k):
        self.nb_cluster = k
        km = KMeans(n_clusters = k)
        km.fit(self.db)
        self.db['cluster'] = km.predict(self.db)

    def compute_standardization(self):
        self.m = numpy.mean(self.db,axis = 0)
        self.sigma = numpy.var(self.db,axis = 0)**(0.5)
        self.db = (self.db - self.m) / self.sigma

    def compute_var_coude(self,x):
        c = numpy.mean(x[self.config['col_expert']],axis = 0)
        res = numpy.array(x[self.config['col_expert']]) - numpy.array(c)
        res = res**(2)
        res = numpy.sum(res,axis = 1)
        res = numpy.sum(res)
        return res

    def compute_coude(self):
        coude = self.db.groupby(['cluster'])
        coude = coude.apply(lambda x: self.compute_var_coude(x))
        coude = numpy.sum(coude)
        temp = pandas.DataFrame()
        temp['nb_cluster'] = [self.nb_cluster]
        temp['coude'] = coude
        self.dist = self.dist.append(temp)

    def compute_scores(self):
        temp = pandas.DataFrame()
        x = numpy.array(self.db.drop(['cluster'],axis = 1))
        temp['calinski_harabasz_score'] = [calinski_harabasz_score(x,numpy.array(self.db['cluster']))]
        temp['silhouette_score'] = silhouette_score(x,numpy.array(self.db['cluster']))
        temp['nb_cluster'] = self.nb_cluster

        # self.sil = numpy.array(self.sil)
        # self.sil = (self.sil[:, 2] - self.sil[:, 1]) / numpy.max(self.sil[:, 1:3], axis=1)
        # self.sil = numpy.mean(self.sil)
        # temp['silhouette_max'] = self.sil

        self.dist = self.dist.append(temp)

    def compute_dist_intra(self):
        self.sil = self.db[['cluster']]
        self.sil = self.sil.drop_duplicates()
        res = self.db.merge(self.db,how = 'inner',on = ['cluster'],suffixes = ('_left','_right'))
        col_left = numpy.char.find(numpy.array(res.columns).astype(str),sub = '_left') != -1
        col_right = numpy.char.find(numpy.array(res.columns).astype(str),sub = '_right') != -1
        res['dist'] = numpy.sum((numpy.array(res)[:,col_left] - numpy.array(res)[:,col_right])**(2),axis = 1)
        res = res.groupby(['cluster'])
        res = res['dist'].max()
        self.sil = self.sil.merge(res,how = 'inner',on = 'cluster')
        self.sil = self.sil.rename(columns = {'dist':'intra'})

    def compute_dist_extra(self):
        self.db['id'] = 1
        res = self.db.merge(self.db,how = 'inner',on = ['id'],suffixes = ('_left','_right'))
        sel = numpy.array(res['cluster_left'] != res['cluster_right'])
        res = res[sel]
        res = res.drop(['cluster_right'],axis = 1)
        res = res.rename(columns = {'cluster_left':'cluster'})

        col_left = numpy.char.find(numpy.array(res.columns).astype(str),sub = '_left') != -1
        col_right = numpy.char.find(numpy.array(res.columns).astype(str),sub = '_right') != -1
        res['dist'] = numpy.sum((numpy.array(res)[:,col_left] - numpy.array(res)[:,col_right])**(2),axis = 1)
        res = res.groupby(['cluster'])
        res = res['dist'].min()
        self.sil = self.sil.merge(res,how = 'inner',on = 'cluster')
        self.sil = self.sil.rename(columns = {'dist':'extra'})
        self.db = self.db.drop(['id'],axis = 1)

    def compute_distrib(self):
        distr = self.db.drop(['cluster'],axis = 1)
        distr = distr * self.sigma + self.m
        distr['cluster'] = numpy.array(self.db['cluster'])
        for i in range(numpy.max(distr['cluster'])+1):
            sel = numpy.array(distr['cluster'] == i)
            x = distr
            col = x.columns
            x = numpy.array(x)
            x = x[sel,:]
            temp = numpy.quantile(x,numpy.array(range(20)) / 20,axis = 0)
            temp = pandas.DataFrame(temp)
            temp.columns = col
            temp['cluster'] = i
            self.distrib = self.distrib.append(temp)

