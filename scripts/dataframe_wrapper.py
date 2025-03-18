# dataframe_wrapper.py
class DataFrameWrapper:
    def __init__(self, df):
        self.df = df

    def __getattr__(self, name):
        if name in self.df.columns:
            return self.df[name]
        raise AttributeError(f"'DataFrameWrapper' object has no attribute '{name}'")

    def to_dict(self, orient="records"):
        return self.df.to_dict(orient=orient)