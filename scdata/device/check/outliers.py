import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

from scdata._config import config
from scdata.tools.custom_logger import logger
from scdata.device.process.error_codes import StatusCode, ProcessResult

def find_outliers(dataframe, **kwargs):
    columns = kwargs.get('columns', list(dataframe.columns))
    zscore_value = kwargs.get('zscore', 3)
    window = kwargs.get('window', 60)

    df = dataframe.copy()
    cols = []

    for col in columns:
        if '__' in col: continue # Internal code for healthchecks
        if col not in df.columns:
            logger.warning(f'{col} not in columns. Skipping')
            continue

        logger.info (f'Calculating outliers for {col}')
        df[f'{col}_z_score'] = zscore(df[col].rolling(window=window).mean(), nan_policy='omit')

        df.loc[:, f'__{col}'] = False
        df.loc[(df[f'{col}_z_score'] > zscore_value) | (df[f'{col}_z_score'] < -zscore_value), f'__{col}'] = True

        cols.append(f'__{col}')


    return ProcessResult(df[cols], StatusCode.SUCCESS)

def find_outliers_isolation_forest(dataframe, **kwargs):
    columns = kwargs.get('columns', list(dataframe.columns))
    detector = kwargs.get('detector', None)

    if detector is None:
        logger.error(f'Detector cant be null. Aborting')
        return ProcessResult(None, StatusCode.ERROR_MISSING_INPUTS)

    df = dataframe.copy()
    cols = []

    for col in columns:
        if '__' in col: continue # Internal code for healthchecks
        if col not in df.columns:
            logger.warning(f'{col} not in columns. Skipping')
            continue

        logger.info (f'Calculating outliers for {col}')
        df[f'__{col}'] = detector.predict(df)

        cols.append(f'__{col}')

    return ProcessResult(df[cols], StatusCode.SUCCESS)

class MultiDeviceIForest:

    def __init__(
        self,
        sensor_cols,
        lags=(1, 2, 3, 5, 10),
        contamination=0.01,
        normalize_per_device=True,
        custom_checks_fn=None
    ):
        self.sensor_cols = sensor_cols
        self.lags = lags
        self.contamination = contamination
        self.normalize_per_device = normalize_per_device
        self.custom_checks_fn = custom_checks_fn

        self.models = {}
        self.feature_map = {}

    def _build_features(self, df):

        df = df.copy()
        df = df[~df.index.duplicated(keep='first')]
        df = df.dropna()
        # df = df.resample('5Min').mean()

        for sensor in self.sensor_cols:
            if sensor not in df.columns:
                # logger.warn(f"Device doesn't have {sensor}")
                continue
            if self.normalize_per_device:
                mean = df[sensor].mean()
                std = df[sensor].std() + 1e-6
                df[f"{sensor}_norm"] = (df[sensor] - mean) / std
                base = f"{sensor}_norm"
            else:
                base = sensor

            # Lags
            for lag in self.lags:
                df[f"{base}_lag_{lag}"] = df[base].shift(lag)

            # Rolling
            df[f"{base}_roll_mean_10"] = df[base].rolling(10).mean()
            df[f"{base}_roll_std_10"] = df[base].rolling(10).std()

        # Time features
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek

        if self.custom_checks_fn:
            df = self.custom_checks_fn(df)

        df = df.dropna()

        return df

    def fit(self, devices, train_device_ids, sample_frac=0.1):

        samples = []

        for device in devices:
            if device.id not in train_device_ids:
                continue
            logger.info(f"Adding {device.id} to training")
            df = self._build_features(device.data)

            if len(df) == 0:
                continue

            sample = df.sample(frac=sample_frac)
            samples.append(sample)

        if not samples:
            raise ValueError("No training data collected")

        train_df = pd.concat(samples)

        for sensor in self.sensor_cols:
            logger.info(f"Training {sensor}")

            base = f"{sensor}_norm" if self.normalize_per_device else sensor

            feature_cols = [
                c for c in train_df.columns
                if c.startswith(base + "_lag_")
                or c.startswith(base + "_roll_")
            ] + ["hour", "dayofweek"]

            X = train_df[feature_cols]

            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=150,
                n_jobs=-1,
                random_state=42
            )

            model.fit(X)

            self.models[sensor] = model
            self.feature_map[sensor] = feature_cols

        return self

    def predict(self, dataframe):

        df = self._build_features(dataframe)
        if df is None:
            return None
        all_new_cols = []

        for sensor in self.sensor_cols:
            #logger.info(f"Predicting {sensor}")
            if sensor not in dataframe.columns:
                # logger.warn(f"\tOutliers prediction: {sensor} not in data columns")
                continue
            model = self.models[sensor]
            feature_cols = self.feature_map[sensor]

            X = df[feature_cols]

            labels = model.predict(X)
            scores = model.decision_function(X)

            prefix = f"{sensor}"

            cols = [
                f"{prefix}_LABEL",
                f"{prefix}_SCORE",
                f"{prefix}_OUTL",
            ]

            dataframe = dataframe.drop(columns=cols, errors="ignore")

            new_cols = pd.DataFrame({
                f"{prefix}_LABEL": labels,
                f"{prefix}_SCORE": scores,
                f"{prefix}_OUTL": (labels == -1).astype(np.float64),
            }, index=df.index)

            all_new_cols.append(new_cols)

        dataframe = pd.concat([dataframe] + all_new_cols, axis=1)
        dataframe = dataframe.copy()

        return True