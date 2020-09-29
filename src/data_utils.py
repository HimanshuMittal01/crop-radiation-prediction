import os
import pandas as pd
from sklearn.model_selection import KFold

def prepare_data(
    filepath :str,
    sheetname :str,
    delimiter :str = ',',
    num_folds :int = 5,
    random_state :int = 42
    ) -> pd.DataFrame:
    """Loads and prepares data
    Automatically detects format and load accordingly
    """

    ext = os.path.splitext(filepath)[1]
    data = None
    if ext=='.xlsx':
        data = _read_excel_data(filepath, sheetname)
    elif ext=='.csv':
        data = _read_csv_data(filepath, delimiter)
    else:
        raise Exception(f"{ext} format is not yet supported")

    # Create K folds
    data.loc[:, "kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)

    kf = KFold(n_splits=num_folds, shuffle=False, random_state=random_state)

    for f, (t_, v_) in enumerate(kf.split(data)):
        data.loc[v_, "kfold"] = f
    
    # TODO: Save data for caching later
    # data.to_csv('filename.csv', index=False)

    return data

def _read_excel_data(filepath, sheetname):
    print("Loading xlsx data....")
    if sheetname is None or sheetname.isnumeric():
        print("Multiple sheets are not supported yet for training, hence using sheet 0")
        sheetname = 0
    
    return pd.read_excel(filepath, sheet_name=sheetname)

def _read_csv_data(filepath, delimiter=','):
    return pd.read_csv(filepath, delimiter=delimiter)

def define_data(df, feature_cols=None, target_cols=None):
    """Defines X and y
    If features and target is None then selects the last column as target
    If only features is None then all features except target are selected as features
    """
    all_cols = df.columns
    feature_cols, target_cols = _get_xy_cols(all_cols, feature_cols, target_cols)
    
    return df[feature_cols].values, df[target_cols].values

def _get_xy_cols(all_cols, feature_cols, target_cols):
    if feature_cols is None:
        if target_cols is None:
            feature_cols = all_cols[:-1]
            target_cols = all_cols[-1]
        else:
            target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
            feature_cols = [c for c in all_cols if c not in target_cols]
    else:
        feature_cols = feature_cols if isinstance(feature_cols, list) else [feature_cols]
        if target_cols is None:
            # TODO: Can check if any column is not selected as feature
            # Remove exception if column is found
            raise Exception("If features are specified, then target column must be present")
        else:
            target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
    
    return feature_cols, target_cols