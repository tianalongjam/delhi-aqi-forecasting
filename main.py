import kagglehub
import pandas as pd

# Download latest version
# path = kagglehub.dataset_download("sohails07/delhi-weather-and-aqi-dataset-2025")

# print("Path to dataset files:", path)

#and then moved it to my main directory

df = pd.read_csv("delhi-weather-aqi-2025.csv")
print(len(df))