import tkinter as tk
import pandas as pd

df = pd.read_csv("datei.csv")
df_sample = df.sample(n=1000, random_state=42)
print(df_sample.head())
