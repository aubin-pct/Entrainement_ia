import pandas as pd

class Scaler:
    max = {}

    def fit_transform(self, df):
        for name, series in df.items():
            self.max[name] = abs(series.max() if series.max() > abs(series.min()) else series.min())
            df[name] = series / self.max[name]

    def transform(self, df):
        for name, series in df.items():
            df[name] = series / self.max[name]

    def fit(self, df):
        for name, series in df.items():
            self.max[name] = abs(series.max() if series.max() > abs(series.min()) else series.min())
