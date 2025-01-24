import pandas as pd

class Scaler:
    dico = {}

    def fit_transform(self, df):
        for name, series in df.items():
            max = series.max()
            self.dico[name] = max
            df[name] = series / max

    def transform(self, df):
        for name, series in df.items():
            df[name] = series / self.dico[name]
