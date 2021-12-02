import os
import rasterio
from pickle import dump, load
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)


class Handler:
    def __init__(self):
        pass
    
    def filter_files(self, extension):
        """
        Method to get list of path files acoording of extension in inputs folder
        :param extension: extension of file, eg: gpkg, csv
        :return: a list of path files
        """
        path_files = []
        for root, directory, files in os.walk('../inputs'):
            for file in files:
                if file.endswith(extension):
                    path_files.append(os.path.join(root, file))
        
        return path_files


class DeepModels:
    def __init__(self, name, extension):
        self.dm_handler = Handler()
        self.training_data(name, extension)
    
    def run_geopackage(self, name, extension):
        """
        Method to run a geopackage and return a geodataframe
        :param name: name of geopackage in inputs folder
        :param extension: extension of file eg: gpkg, csv
        :return: geodataframe
        """
        paths = self.dm_handler.filter_files(extension)
        matching = [s for s in paths if name in s]
        
        return gpd.read_file(matching[0], driver='ogr')
    
    def training_data(self, name, extension):
        """
        Method to get data to be training and tested in models. NOTE: Need to configure slice of columns data X y
        :param name: file name in inputs folders
        :param extension: extension of file eg: gpkg
        :return: dataframe X y X_train X_test Y_train Y_test as self variables
        """
        df = self.run_geopackage(name, extension)
        df = df.dropna()
        oversample = SMOTE()
        self.X, self.y = oversample.fit_resample(df.iloc[:, 1:5], df.iloc[:, 5])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=0.20,
                                                                                random_state=42)
    
    def classifiers(self):
        """
        Method to save all classifiers inside
        :return: dump models in folder xxxx
        """
        
        def random_forest(X_train, Y_train):
            """
            Method to create a svm model fitted
            :param X_train: Variables
            :param Y_train: Targets
            :return: Random Forest Model fitted
            """
            rf = RandomForestClassifier(n_estimators=500,
                                        max_depth=4,
                                        oob_score=True,
                                        n_jobs=1,
                                        verbose=True)
            
            return rf.fit(X_train, Y_train)
        
        def svm(X_train, Y_train):
            """
            Method to create a svm model fitted
            :param X_train: Variables
            :param Y_train: Targets
            :return: SVM Model fitted
            """
            print('runing model svm')
            svm_ = SVC(kernel='rbf',
                       C=10,
                       gamma=1)
            
            # svm_ = SVC(kernel='linear',
            #            C=1
            #            )
            
            return svm_.fit(X_train, Y_train)
        
        def gb(X_train, Y_train):
            """
            Method to create a gradiente boosting model fitted
            :param X_train: Variables
            :param Y_train: Targets
            :return: Gradient boosting Model fitted
            """
            gb_ = GradientBoostingRegressor(random_state=0)
            
            return gb_.fit(X_train, Y_train)
        
        dir_models = os.getcwd() + '/models'
        if not os.path.exists(dir_models):
            os.makedirs(dir_models)
        
        models_list = ['random_forest', 'svm', 'gb']
        #models_list = ['svm', ]
        
        for model in models_list:
            dump(locals()[model](self.X_train, self.Y_train),
                 open(f'{dir_models}/model_{model}.pkl', 'wb'))
    
    def get_estimates(self):
        """
        Method to load models get predicts ands statistics
        :return: statistics and compare between test and predicts
        """
        dir = os.path.join(os.getcwd(), 'models')
        statistics = []
        compare_tests_predicts = []
        for root, directory, files in os.walk(dir):
            for file in files:
                n_model = file[:-4]
                model = load(open(os.path.join(dir, file), 'rb'))
                if not file == 'model_gb.pkl':
                    Y_pred = model.predict(self.X_test)
                    Y_pred_train = model.predict(self.X_train)
                else:
                    Y_pred = model.predict(self.X_test)
                    Y_pred[np.where(Y_pred < 0.80)] = 0
                    Y_pred[np.where(Y_pred >= 0.80)] = 1
                    Y_pred_train = model.predict(self.X_train)
                    Y_pred_train[np.where(Y_pred_train < 0.80)] = 0
                    Y_pred_train[np.where(Y_pred_train >= 0.80)] = 1
                
                accuracy_train = f'{n_model} Train Accuracy :: {round(accuracy_score(self.Y_train, Y_pred_train), 3)}'
                accuracy_test = f'{n_model} Test Accuracy  :: {round(accuracy_score(self.Y_test, Y_pred), 3)}'
                confusion_matrix = f'{n_model} Confussion matrix:\n {metrics.confusion_matrix(self.Y_test, Y_pred)}'
                report = f'{n_model} Classification report:\n {metrics.classification_report(self.Y_test, Y_pred)}'
                accuracy = f'{n_model} Classification accuracy: {round(metrics.accuracy_score(self.Y_test, Y_pred), 3)}'
                kappa = f'{n_model} Kappa accuracy:{round(cohen_kappa_score(self.Y_test, Y_pred), 3)}'
                statistics.append([accuracy_train, accuracy_test, confusion_matrix, report, accuracy, kappa])
                compare_tests_predicts.append({n_model: {'truth': self.Y_test,
                                                         'predict': Y_pred}})
        
        return statistics, compare_tests_predicts


class Classification:
    def __init__(self, name, extension):
        self.handler = Handler()
        self.dpmodels = DeepModels(name, extension)
        self.models()
    
    def path_models(self, dir):
        """
        Method to get paths of models trainned to execute classification (random forest, gradiente boosting and svm)
        :param dir: path to models folder
        :return: List of paths
        """
        paths_models = []
        for root, directory, files in os.walk(dir):
            for file in files:
                paths_models.append(os.path.join(dir, file))
        
        return paths_models
    
    def models(self):
        """
        Method to get paths models and if not exist run training dataset to create it
        :return: List of path models
        """
        dir = os.getcwd() + '/models'
        paths_models = self.path_models(dir)
        if not paths_models:
            self.dpmodels.classifiers()
            paths_models = self.path_models(dir)
        
        return paths_models
    
    def stack_bands(self):
        """
        Method to stack rgb+nir in array numpy to classify
        :return: shape, profile and stackbands
        """
        array = []
        shape = []
        profile = []
        for raster in self.handler.filter_files('.tif'):
            img = rasterio.open(raster)
            profile.append(img.profile)
            band1 = img.read(1).flatten()
            shape.append(img.shape)
            band2 = img.read(2).flatten()
            band3 = img.read(3).flatten()
            band4 = img.read(4).flatten()
            array.append(band1)
            array.append(band2)
            array.append(band3)
            array.append(band4)
            img.close()
        
        return shape, profile, np.column_stack((array[0:]))
    
    def classify(self):
        """
        Method to predict values considering three models in models folders
        :return: profile of raster and predicts
        """
        shape, profile, stack = self.stack_bands()
        
        predicts = []
        for model in self.models():
            n_model = model.split('/')[-1][:-4]
            print(f'Rodando modelo {n_model}')
            model = load(open(model, 'rb'))
            if n_model == 'model_gb':
                gb_predict = model.predict(stack)
                gb_predict[np.where(gb_predict < 0.80)] = 0
                gb_predict[np.where(gb_predict >= 0.80)] = 1
                predicts.append({n_model: {'predict': gb_predict.reshape(shape[0][0], -1)}})
            
            else:
                predicts.append({n_model: {'predict': model.predict(stack).reshape(shape[0][0], -1)}})
        
        return profile, predicts
    
    def classified_tif(self):
        """
        Method to export predict in raster tif with two classes (not vegetation 1, vegetation 0),
        considering structure of input raster
        :return: raster tif of each classifier
        """
        path_predicteds = '/home/newmar/Downloads/fontes_tcc/research_ds/predicts'
        profile, predicts = self.classify()
        struct = profile[0]
        struct.update({'count': 1, 'dtype': rasterio.int8})
        del struct['tiled']
        del struct['interleave']
        
        for model in predicts:
            for key, value in model.items():
                with rasterio.open(os.path.join(path_predicteds, f'{key}.tif'),
                                   'w', **struct) as dst:
                    dst.write_band(1, value['predict'].astype(rasterio.int8))


if __name__ == '__main__':
    ## Initialize class and create pkl models if not exists
    cl = Classification('amostra', 'gpkg')
    # Run classification
    cl.classified_tif()
    #
    for i in cl.dpmodels.get_estimates()[0]:
        for est in i:
            print(est)

    # gridsearchcv para parametros svm
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 10, 100], 'C': [0.02, 0.03, 1, 10, 100]}, ]
    # # {'kernel': ['linear'], 'C': [1, 10, 100]}]
    #
    # # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.5,], 'C': [10, ]},]
    # # tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},]#1/28 0.25/4
    # scores = ['precision', 'recall']
    # # #scores = ['recall',]
    # #
    # #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % score)
    #     clf.fit(cl.dpmodels.X_train, cl.dpmodels.Y_train)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #     print()
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = cl.dpmodels.Y_test, clf.predict(cl.dpmodels.X_test)
    #     print(metrics.classification_report(y_true, y_pred))
    #     print()
