import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# =====================================================
# TAMBAHAN SAJA: REMOVE OUTLIERS (IQR) — SESUAI NOTEBOOK
# =====================================================
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def preprocess_data(input_path, output_path):
    # 1. LOAD DATA
    df = pd.read_excel(input_path)

    categorical_cols = ['plan_type', 'device_brand']
    numerical_cols = [
        'avg_data_usage_gb', 'pct_video_usage', 'avg_call_duration',
        'sms_freq', 'monthly_spend', 'topup_freq',
        'travel_score', 'complaint_count'
    ]
    target_col = 'target_offer'
    id_col = 'customer_id'

    # 2. HANDLE NILAI NEGATIF (SESUAI NOTEBOOK)
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    for col in numerical_cols:
        df[col] = df[col].mask(df[col] < 0).fillna(df[col].median())

    # =================================================
    # 2.5 OUTLIER (IQR) — INI YANG DIMINTA REVIEWER
    # =================================================
    df = remove_outliers_iqr(df, numerical_cols)

    # 3. LABEL ENCODING (TARGET & ID)
    le_target = LabelEncoder()
    le_customer = LabelEncoder()

    df['target_offer_encoded'] = le_target.fit_transform(df[target_col])
    df['customer_id_encoded'] = le_customer.fit_transform(df[id_col])

    # 4. SCALING NUMERIK (MINMAX — SAMA DENGAN NOTEBOOK)
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 5. ONE HOT ENCODING KATEGORIK
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_data = ohe.fit_transform(df[categorical_cols])

    ohe_df = pd.DataFrame(
        ohe_data,
        columns=ohe.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # 6. GABUNGKAN SEMUA (SESUAI AUTOMATE ASLI)
    df_final = pd.concat([
        df[['customer_id_encoded'] + numerical_cols],
        ohe_df,
        df['target_offer_encoded']
    ], axis=1)

    # 7. SIMPAN DATASET FINAL
    df_final.to_csv(output_path, index=False)

    return df_final


if __name__ == "__main__":
    preprocess_data(
        input_path="../ac-01_telco_customer_behavior_raw/ac-01_telco_customer_behavior_mock_data.xlsx",
        output_path="ac-01_telco_customer_behavior_preprocessing/data_preprocessed.csv"
    )
