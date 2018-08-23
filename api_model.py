import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class TEAM_NAME_classifier():
    def __init__(self):
        ##############################################################
        ##  В этом методе инициализируем объекты, то есть создаем   ##
        ##  нужные модели машинного обучения, привязывая их к self  ##
        ##############################################################

        # self.myTree = DecisionTreeClassifier(max_depth=3)
        self.myKNN = KNeighborsClassifier(n_neighbors=2)


    def fit(self):
        ##############################################################
        ##  Данный метод обучает ваши модели(-ль). Здесь же можно   ##
        ##  загрузить нужные данные или вынести загрузку и препро-  ##
        ##  цессинг в отдельный метод или функцию. It's up to you!  ##
        ##############################################################

        self.df = self.__readDF()
        self.df = self.__preproc(self.df)

        self.X_train = self.df.drop(['class', 'area'], axis=1)
        self.y_train = self.df['class']

        # здесь можно пофитить параметры при помощи GridSearch
        # ...
        # ...

        self.myKNN.fit(self.X_train, self.y_train)


    def predict(self, X_raw_test=None):
        ##############################################################
        ##  Этот метод предсказывает вектор y по данной матрице X   ##
        ##                                                          ##
        ##  ВАЖНО: данные в матрице X никаким образом не предобра-  ##
        ##  ботаны!                                                 ##
        ##############################################################
        
        X_test = self.__preproc(X_raw_test).drop(['area'], axis=1)

        # инжинирим фичи тут
        # ... 
        # ...

        y_pred = self.myKNN.predict(X_test)
        return y_pred


    def __readDF(self):
        blue = pd.read_csv('./data/blue.csv')
        green = pd.read_csv('./data/green.csv')
        orange = pd.read_csv('./data/orange.csv')
        red = pd.read_csv('./data/red.csv')
        yellow = pd.read_csv('./data/yellow.csv')
        white = pd.read_csv('./data/white.csv')

        blue['class'] = blue['RGB'].apply(lambda x: 'blue')
        green['class'] = green['RGB'].apply(lambda x: 'blue')
        orange['class'] = orange['RGB'].apply(lambda x: 'orange')
        red['class'] = red['RGB'].apply(lambda x: 'red')
        yellow['class'] = yellow['RGB'].apply(lambda x: 'yellow')
        white['class'] = white['RGB'].apply(lambda x: 'white')

        return pd.concat([blue, green, orange, red], axis=0)


    def __getSepRGB(self, s):
        return s.strip('[]()').replace(' ', '').split(',')


    def __preproc(self, df):
        df['B'] = df['RGB'].apply(lambda x: int(float(self.__getSepRGB(x)[0])))
        df['G'] = df['RGB'].apply(lambda x: int(float(self.__getSepRGB(x)[1])))
        df['R'] = df['RGB'].apply(lambda x: int(float(self.__getSepRGB(x)[2])))

        df.drop('RGB', axis=1, inplace=True)

        df['cont'] = df['cont'].apply(lambda x: x.strip('[]'))
        df['area'] = df['area'].apply(float).apply(int)

        # инжинирим фичи из cont
        # ....
        # ....

        df.drop('cont', axis=1, inplace=True)

        return df

