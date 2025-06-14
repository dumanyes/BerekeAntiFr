import pandas as pd
from functools import reduce
from sklearn.preprocessing import LabelEncoder
import joblib


# --- Константы
S3_AMPL = "s3://data-for-case-3/tabledata/valid_data/valid_amplitude_chunk_00.parquet"
S3_APP = "s3://data-for-case-3/tabledata/valid_data/valid_app_data.parquet"
S3_TARGET = "s3://data-for-case-3/tabledata/valid_data/valid_target_df.parquet"

STORAGE_OPTIONS = {
    "key": "YCB2BAvtu7_kXfgsmzA_0xv4i",
    "secret": "YCNnsH6n4uHb6_OQ5fpkhN8F4ZhgNbZLwJqsccoV",
    "client_kwargs": {"endpoint_url": "https://storage.yandexcloud.kz"}
}

numeric_cols = [
    'TOTALAMOUNT', 'SUM_CREDIT_KZT', 'DM5DPD1GCVPSUM', 'DM5DPD1SALARYSUM', 'DM5EXPSUM',
    'DM5INCSUM', 'DM5INC', 'DM5EXP', 'DM7INC', 'DM7EXP', 'DM6SCOREN6PD', 'DM6SCOREN6',
    'FINALKDN', 'CREDITTERM_RBL0', 'CLI_AGE', 'DEPENDANT_COUNT', 'IS_SERVWS_MNLI'
]
categorical_cols = [
    'PURPOSE_LOAN', 'OPV_REASON', 'KANAL_PRODAZH', 'APPLICATION_ISA0AUTO',
    'GENDER', 'BKI', 'VKI', 'MARITALSTATUS'
]
label_cols = ['REGCOUNTY', 'BRANCH']
fin_num_cols = [
    'APPLICATIONID', 'CREATE_DATETIME', 'TOTALAMOUNT', 'SUM_CREDIT_KZT',
    'DM5DPD1GCVPSUM', 'DM5DPD1SALARYSUM', 'DM5EXPSUM', 'DM5INCSUM', 'DM5INC',
    'DM5EXP', 'DM7INC', 'DM7EXP', 'DM6SCOREN6PD', 'DM6SCOREN6', 'FINALKDN',
    'CREDITTERM_RBL0', 'CLI_AGE', 'DEPENDANT_COUNT', 'IS_SERVWS_MNLI'
]
important_features = ['TOTALAMOUNT',
 'DM5DPD1GCVPSUM',
 'DM5EXPSUM',
 'DM5INCSUM',
 'DM5INC',
 'DM5EXP',
 'DM7INC',
 'DM7EXP',
 'DM6SCOREN6PD',
 'DM6SCOREN6',
 'FINALKDN',
 'CREDITTERM_RBL0',
 'CLI_AGE',
 'DEPENDANT_COUNT',
 'PURPOSE_LOAN_На потребительские цели',
 'KANAL_PRODAZH_B-Bank',
 'KANAL_PRODAZH_QR',
 'GENDER_Женский',
 'BKI_Плохая',
 'VKI_Хорошая',
 'MARITALSTATUS_Женат/Замужем',
 'country_kz_ratio_1d',
 'n_unique_device_id_1d',
 'n_unique_event_type_1d',
 'n_unique_ip_1d',
 'n_event_id_1d',
 'n_event_id_2d',
 'country_kz_ratio_3d',
 'n_event_id_3d',
 'country_kz_ratio_30d',
 'n_unique_event_type_30d',
 'n_unique_ip_30d',
 'n_event_id_30d']

def read_parquet_from_s3(s3_path, storage_options):
    try:
        return pd.read_parquet(s3_path, storage_options=storage_options)
    except Exception as e:
        print(f"Error reading {s3_path}: {e}")
        return None

def clean_numeric(col):
    return pd.to_numeric(
        col.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

def prepare_table_data(df_app, df_target):
    print("Aplication validation Data cleaning and transformation ...")
    df_merged = df_app.merge(df_target[['APPLICATIONID']], on='APPLICATIONID', how='inner')
    df_merged.columns = df_merged.columns.str.strip().str.replace(" ", "")
    df_merged['IS_SERVWS_MNLI'] = (df_merged['MNG_NAME_LOGIN_INIT'] == 'servws').astype(int)
    drop_cols = [
        'CREATE_DATE','DATA_ISSUE', 'VINTAGE','SPF','REGREGION','REGTOWN',
        'COMPANY_NAME','BIRTHCOUNTRY','PRODUCT_GROUP','MNG_NAME_LOGIN_INIT','MNG_NAME_INIT'
    ]
    df_merged = df_merged.drop(columns=[col for col in drop_cols if col in df_merged.columns], errors='ignore')
    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = clean_numeric(df_merged[col])
    df_categorical_ohe = pd.get_dummies(df_merged[categorical_cols], prefix=categorical_cols, dtype=int)
    df_encoded = df_merged[label_cols].apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
    df_numeric = df_merged[[col for col in fin_num_cols if col in df_merged.columns]]
    # Сброс индексов для корректного объединения
    df_numeric = df_numeric.reset_index(drop=True)
    df_categorical_ohe = df_categorical_ohe.reset_index(drop=True)
    df_encoded = df_encoded.reset_index(drop=True)
    return pd.concat([df_numeric, df_categorical_ohe, df_encoded], axis=1)

def extract_event_features(df, window_days):
    mask = (
        (df["client_event_time"] <= df["CREATE_DATETIME"]) &
        (df["client_event_time"] > df["CREATE_DATETIME"] - pd.Timedelta(days=window_days))
    )
    df_window = df[mask]
    return (
        df_window.groupby("APPLICATIONID").agg(
            **{
                f"country_kz_ratio_{window_days}d": ("country", lambda x: (x == "Kazakhstan").mean()),
                f"n_unique_device_id_{window_days}d": ("device_id", pd.Series.nunique),
                f"n_unique_event_type_{window_days}d": ("event_type", pd.Series.nunique),
                f"n_unique_ip_{window_days}d": ("ip_address", pd.Series.nunique),
                f"n_event_id_{window_days}d": ("event_id", "count")
            }
        ).reset_index()
    )

def extract_os_features_30d(df):
    mask = (
        (df["client_event_time"] <= df["CREATE_DATETIME"]) &
        (df["client_event_time"] > df["CREATE_DATETIME"] - pd.Timedelta(days=30))
    )
    df_window = df[mask]
    df_last = (
        df_window.sort_values(["APPLICATIONID", "client_event_time"], ascending=[True, False])
        .groupby("APPLICATIONID", as_index=False)
        .first()[["APPLICATIONID", "os_name"]]
        .rename(columns={"os_name": "last_os_name_30d"})
    )
    df_mode = (
        df_window.groupby("APPLICATIONID")['os_name']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"os_name": "most_common_os_name_30d"})
    )
    return df_last.merge(df_mode, on="APPLICATIONID", how="left")

def predict_from_table_data():
    print("Amplitude validation Data cleaning and transformation ...")
    # --- Загрузка данных
    df_app = read_parquet_from_s3(S3_APP, STORAGE_OPTIONS)
    df_target = read_parquet_from_s3(S3_TARGET, STORAGE_OPTIONS)
    df_table_data = prepare_table_data(df_app, df_target)

    # --- Загрузка и обработка амплитудных данных
    df_all = read_parquet_from_s3(S3_AMPL, STORAGE_OPTIONS)
    df_target['target'] = None  # если нужно, иначе уберите
    df_all = df_all.drop_duplicates(subset=['applicationid', 'event_id'], keep='first')
    df_all['client_event_time'] = pd.to_datetime(df_all['client_event_time'], utc=True).dt.tz_localize(None)
    df_app['CREATE_DATETIME'] = pd.to_datetime(df_app['CREATE_DATETIME'], format="%d.%m.%Y %H:%M", errors="coerce")
    df_final = (
        df_app[['APPLICATIONID', 'CREATE_DATETIME']]
        .merge(df_target[['APPLICATIONID', 'target']], on='APPLICATIONID', how='left')
        .merge(df_all, left_on='APPLICATIONID', right_on='applicationid', how='left')
    )
    df_final['user_creation_time'] = pd.to_datetime(df_final['user_creation_time'], errors='coerce')
    df_final['days_after_registration'] = (df_final['CREATE_DATETIME'] - df_final['user_creation_time']).dt.days

    # --- Извлечение признаков
    features_list = [extract_event_features(df_final, days) for days in [1, 2, 3, 30]]
    df_event_features = reduce(lambda left, right: pd.merge(left, right, on="APPLICATIONID", how="outer"), features_list)
    df_os_30d = extract_os_features_30d(df_final)
    # --- Финальный датасет
    df_ampl_data = (
        df_final[["APPLICATIONID", "CREATE_DATETIME", "target"]]
        .drop_duplicates()
        .merge(df_event_features, on="APPLICATIONID", how="left")
        .merge(df_os_30d, on="APPLICATIONID", how="left")
    )
    df_model_input = pd.merge(df_ampl_data, df_table_data, on='APPLICATIONID', how='inner')
    df_model_input['new_os'] = df_model_input.apply(lambda x: 1 if x['most_common_os_name_30d'] == x['last_os_name_30d'] else 0, axis=1)
    le = LabelEncoder()
    df_model_input['last_os_name_30d_encoded'] = le.fit_transform(df_model_input['last_os_name_30d'])
    df_model_input['most_common_os_name_30d_encoded'] = le.fit_transform(df_model_input['most_common_os_name_30d'])
    
    print('Data merged and ready to predict. Loading of trained model... Predicting validation data..')
    # Загрузка модели
    final_model = joblib.load('model_antifraud.pkl')
    X_val_proba = df_model_input[important_features]
    y_val_proba = final_model.predict_proba(X_val_proba)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    df_result = df_model_input.loc[X_val_proba.index, ['APPLICATIONID']].copy()
    df_result.rename(columns={'APPLICATIONID':'id'},inplace=True)
    df_result['isFrod'] = y_val_pred
    df_result['probability'] = y_val_proba
    assert len(df_model_input) == len(y_val_proba)
    print('Prediction ready!')
    print('Fraud/ no fraud cases distribution: ', df_result['isFrod'].value_counts())

    df_result.to_csv('result.csv',index=False)
    print('Aplication and amplitude data prediction results saved to csv: result.csv')
    return df_result
    

def main():
    df = predict_from_table_data()
    print(f"DF saved:{df.shape}")

if __name__ == '__main__':
    main()