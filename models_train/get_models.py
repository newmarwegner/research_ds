import os
from pickle import dump
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
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
            svm_ = SVC(kernel='rbf',
                       C=1,
                       gamma=1)

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
        for model in models_list:
            dump(locals()[model](self.X_train, self.Y_train),
                 open(f'{dir_models}/model_{model}.pkl', 'wb'))


if __name__ == '__main__':
    mode = DeepModels('amostra', 'gpkg')
    print(mode.classifiers())
    # x = mode.X_train
    # print(np.isnan(x))
    # print(np.isfinite(x))
