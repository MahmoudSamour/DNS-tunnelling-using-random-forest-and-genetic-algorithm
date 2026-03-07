import os
import requests
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def download_dataset():
    file_urls = [
        ("https://drive.google.com/uc?export=download&id=1cictwnxUyu1vCa4H9iefIrQeVLCC3RCv", "benign-chrome.csv"),
        ("https://drive.google.com/uc?export=download&id=1cms99qEylyvesqcX3dQRZOUQRAONy2uS", "benign-firefox.csv"),
        ("https://drive.google.com/uc?export=download&id=1cqDL7A_kdOCL4Km4uUifRPllFmB3WaZ_", "mal-dns2tcp.csv"),
        ("https://drive.google.com/uc?export=download&id=1cxeTvXNV-OY_4T6xs4sUB98lmanROw3m", "mal-dnscat2.csv"),
        ("https://drive.google.com/uc?export=download&id=1czNRMpNyicFNYW2fbK_WjsoF77qB9_XA", "mal-iodine.csv")
    ]

    if not os.path.exists("DoHBrw-2020"):
        os.makedirs("DoHBrw-2020")

    print("--- Checking Data Files ---")
    for url, filename in file_urls:
        file_path = os.path.join("DoHBrw-2020", filename)
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                print(f"{filename} ready.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists.")

def load_and_preprocess_dns_data():
    download_dataset()

    # Load and Concatenate
    df_benign = pd.concat([pd.read_csv('DoHBrw-2020/benign-chrome.csv'),
                           pd.read_csv('DoHBrw-2020/benign-firefox.csv')], ignore_index=True)
    df_benign['labels'] = 0

    df1_malic = pd.read_csv('DoHBrw-2020/mal-iodine.csv'); df1_malic['labels'] = 1
    df2_malic = pd.read_csv('DoHBrw-2020/mal-dns2tcp.csv'); df2_malic['labels'] = 2
    df3_malic = pd.read_csv('DoHBrw-2020/mal-dnscat2.csv'); df3_malic['labels'] = 3

    data = shuffle(pd.concat([df_benign, df1_malic, df2_malic, df3_malic], ignore_index=True), random_state=1)

    # Cleaning and Imputation
    data_dropped = data.drop(columns=[col for col in data.columns if data[col].nunique() == 1])
    data_filled = data_dropped.fillna(0)
    X = data_filled.drop(["TimeStamp", "labels", "SourceIP", "DestinationIP"], axis=1, errors='ignore')
    feature_names = list(X.columns)
    y = data_filled['labels'].values

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    # Exact Split used in study (50% test, 37.5% train, 12.5% val)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return X_train, X_test, X_val, y_train, y_test, y_val, feature_names
