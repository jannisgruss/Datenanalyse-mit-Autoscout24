import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



zielverzeichnis = "E:/!Data Science Institute/GitRepo/Autoscout24"
data = pd.read_csv(f"{zielverzeichnis}/autoscout24.csv")

data.info()
data.describe()
# abfragbar sind Kilometerstand, Marke, Model, Kraftstoffart, Getriebe, Zustand, Preis, Leistung u. Verkaufsjahr
# ca. 46405 Einträge


values_marke = data['make'].unique()

entries = len(data.dropna())


values_Angebot = data['offerType'].unique()
print(values_Angebot)
print(entries)

min_jahr = data['year'].min()
max_jahr = data['year'].max()

print(f"Anfang: {min_jahr}\nEnde:   {max_jahr}")

new_cars = (data['offerType'] =='New').sum()
print(new_cars)

new_cars = data[data['offerType'] == 'New']

unique_years_new_cars = new_cars['year'].unique()

# Optional: Sortiere die Jahre, wenn gewünscht
unique_years_new_cars_sorted = np.sort(unique_years_new_cars)

print(f"Jahre, in denen 'New' Fahrzeuge verkauft wurden: {unique_years_new_cars_sorted}")

sns.pairplot(data)

