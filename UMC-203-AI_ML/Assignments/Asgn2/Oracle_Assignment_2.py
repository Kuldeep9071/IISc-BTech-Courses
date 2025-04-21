import json
import requests
import zipfile
import requests
import json

SERVER_URL = "http://10.24.32.171:8000" # Don't try being funny


def download_zip(srn, save_path):
    try:
        response = requests.get(SERVER_URL + "/q2", params={"srn": srn}, stream=True)
        if response.status_code == 418:
            raise ValueError(f"I'm a teapot. {response.content}")
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded ZIP file to {save_path}")
    except requests.RequestException as e:
        print(f"Failed to download ZIP: {e}")


def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print(zip_ref)
            zip_ref.extractall(extract_to)
        print(f"Extracted ZIP contents to {extract_to}")
    except zipfile.BadZipFile:
        print("Error: Invalid ZIP file")


def q2_get_mnist_jpg_subset(srn: int):
    zip_path = "./q2_data.zip"
    download_zip(srn, zip_path)
    extract_zip(zip_path, "q2_data")


def q1_get_cifar100_train_test(srn: int):
    try:
        response = requests.get(f"{SERVER_URL}/q1", params={"srn": srn})
        response.raise_for_status()  # Raise an error for HTTP failures

        data = response.json()
        train_data = data.get("train_data")
        test_data = data.get("test_data")

        return train_data, test_data
    except requests.RequestException as e:
        print("Error calling server:", e)
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
    except Exception as e:
        print("Unexpected error:", e)

def q3_stocknet(srn:int):
    try:
        response = requests.get(f"{SERVER_URL}/q3_stock", params={"srn": srn})
        response.raise_for_status()  # Raise an error for HTTP failures

        data = response.json()
        stock_ticker = data.get("stock")

        return stock_ticker
    except requests.RequestException as e:
        print("Error calling server:", e)
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
    except Exception as e:
        print("Unexpected error:", e)
        

def q3_linear_1(srn:int):
    try:
        response = requests.get(f"{SERVER_URL}/q3_l1", params={"srn": srn})
        response.raise_for_status()  # Raise an error for HTTP failures

        data = response.json()
        X_train = data.get("X_train")
        y_train = data.get("y_train")
        X_test = data.get("X_test")
        y_test = data.get("y_test")

        return X_train, y_train, X_test, y_test
    except requests.RequestException as e:
        print("Error calling server:", e)
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
    except Exception as e:
        print("Unexpected error:", e)
        

def q3_linear_2(srn:int):
    try:
        response = requests.get(f"{SERVER_URL}/q3_l2", params={"srn": srn})
        response.raise_for_status()  # Raise an error for HTTP failures

        data = response.json()
        X_train = data.get("X_train")
        y_train = data.get("y_train")
        X_test = data.get("X_test")
        y_test = data.get("y_test")

        return X_train, y_train, X_test, y_test
    except requests.RequestException as e:
        print("Error calling server:", e)
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
    except Exception as e:
        print("Unexpected error:", e)

