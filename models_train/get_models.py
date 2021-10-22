import os
from pickle import dump, load
import geopandas as gpd
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
        self.X = df.iloc[:, 1:5]
        self.y = df.iloc[:, 5]
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
                       C=1,
                       gamma='auto')

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

        # models_list = ['random_forest', 'svm', 'gb']
        models_list = ['svm', ]

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
                if not file == 'model_gb.pkl':
                    n_model = file[:-4]
                    model = load(open(os.path.join(dir, file), 'rb'))
                    Y_pred = model.predict(self.X_test)
                    Y_pred_train = model.predict(self.X_train)
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
    def __init__(self, raster_path, name, extension):
        self.dpmodels = DeepModels(name, extension)
        self.models()

    def path_models(self, dir):
        paths_models = []
        for root, directory, files in os.walk(dir):
            for file in files:
                paths_models.append(os.path.join(dir, file))

        return paths_models

    def models(self):
        dir = os.getcwd() + '/models'
        paths_models = self.path_models(dir)
        if not paths_models:
            self.dpmodels.classifiers()
            paths_models = self.path_models(dir)

        return paths_models

if __name__ == '__main__':
    mode = DeepModels('amostra', 'gpkg')
    mode.classifiers()
    # for i in mode.get_estimates()[0]:
    #     for est in i:
    #         print(est)
    # x = mode.X_train
    # print(np.isnan(x))
    # print(np.isfinite(x))
