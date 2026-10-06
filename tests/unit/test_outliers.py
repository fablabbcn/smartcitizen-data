import numpy as np
import pandas as pd

from scdata.device.check.outliers import find_outliers_isolation_forest
from scdata.device.process.error_codes import StatusCode


class Detector:
    ''' Flags one row of A as outlier and drops the first row (no features) '''
    def predict(self, dataframe):
        prediction = dataframe.iloc[1:].copy()
        prediction['A_OUTL'] = 0.0
        prediction.loc[prediction.index[2], 'A_OUTL'] = 1.0
        return prediction


def test_isolation_forest_flags_predicted_rows():
    index = pd.date_range('2026-01-01', periods=5, freq='1min', tz='UTC')
    df = pd.DataFrame({'A': np.arange(5.0), 'B': np.arange(5.0)}, index=index)

    result = find_outliers_isolation_forest(df, columns=['A', 'B'], detector=Detector())

    assert result.status_code == StatusCode.SUCCESS
    # B is not predicted by the detector, row 0 has no prediction
    assert list(result.data.columns) == ['__A']
    assert result.data['__A'].tolist() == [False, False, False, True, False]


def test_isolation_forest_requires_detector():
    result = find_outliers_isolation_forest(pd.DataFrame({'A': [1.0]}))

    assert result.status_code == StatusCode.ERROR_MISSING_INPUTS
