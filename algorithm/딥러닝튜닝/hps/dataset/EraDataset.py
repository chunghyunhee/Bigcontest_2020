import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# class : MNISTDataset
class EraDataset(object):
    @staticmethod
    def get():
        # 데이터 불러오기
        df_train = pd.read_csv("/tmp/pycharm_project_125/goldmine/hps/df_train.csv")

        y_target = df_train['label']
        X_data = df_train.drop('label', axis=1)

        X_train = X_data.values
        y_train = y_target.values

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    print(EraDataset.get())
