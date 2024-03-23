import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

zielverzeichnis = "E:/!Data Science Institute/GitRepo/Autoscout24"
data = pd.read_csv(f"{zielverzeichnis}/autoscout24.csv")

data.info()
data.describe()
# abfragbar sind Kilometerstand, Marke, Model, Kraftstoffart, Getriebe, Zustand, Preis, Leistung u. Verkaufsjahr
# ca. 46405 Einträge

values_marke = data['make'].unique()
entries = len(data.dropna())

values_Angebot = data['offerType'].unique()
#print(values_Angebot)
#print(entries)

min_jahr = data['year'].min()
max_jahr = data['year'].max()

#print(f"Anfang: {min_jahr}\nEnde:   {max_jahr}")

new_cars = (data['offerType'] =='New').sum()
#print(new_cars)

new_cars = data[data['offerType'] == 'New']

unique_years_new_cars = new_cars['year'].unique()

# Optional: Sortiere die Jahre, wenn gewünscht
unique_years_new_cars_sorted = np.sort(unique_years_new_cars)

#print(f"Jahre, in denen 'New' Fahrzeuge verkauft wurden: {unique_years_new_cars_sorted}")

#sns.pairplot(data)

data = data.dropna()



#### Präsentation

st.title("Datenanalyse am Beispiel von Autoscout24\n")

st.header("Erste Analysen zum Datensatz")

st.subheader("Überblick")

st.markdown(f"""
- Es wurden 46405 Autos verkauft.
- Die Autos wurden zwischen den Jahren {min_jahr} und {max_jahr} verkauft.
- Insgesamt wurden 13 Neuwägen verkauft - alle im Jahr 2021.
- Insgesamt gibt es 334 fehlende Werte (NaNs).
""")

st.subheader("Liste aller verfügbaren Marken")

marken = sorted(values_marke)
marken_liste = ", ".join(marken)

st.markdown(f"{marken_liste}")

st.subheader("Nenneswerte Veränderungen über die Jahre")

fig, ax = plt.subplots()
sns.lineplot(data=data, x="year", y="price", hue="offerType", ax=ax)

# Legende anpassen
plt.legend(title="Angebotstypen", loc='upper left')

ax.set_title('Preisentwicklung nach Jahr und Angebotstyp')  # Titel des Plots
ax.set_xlabel('Jahr')  # X-Achsen-Beschriftung
ax.set_ylabel('Preisentwicklung (inklusive Volatilität)')
ax.set_ylim(None, 80000)

# Zeige den Plot in Streamlit
st.pyplot(fig)
# das gleiche Prozedere für die Kraftstofart

fig2, ax2 = plt.subplots()
sns.lineplot(data=data, x="year", y="price", hue="fuel", ci=None, ax=ax2)
ax2.set_title('Preisentwicklung nach Jahr und Kraftstoff')
ax2.set_xlabel('Jahr')
ax2.set_ylabel('Preis')
plt.legend(title="Kraftstofftypen", loc='upper left', prop={'size':8})
st.pyplot(fig2)

count_per_year_fuel = data.groupby(['year', 'fuel']).size().reset_index(name='count')

# Berechne die Gesamtanzahl der Fahrzeuge pro Jahr
total_per_year = count_per_year_fuel.groupby('year')['count'].transform('sum')

# Berechne die prozentuale Verteilung
count_per_year_fuel['percentage'] = (count_per_year_fuel['count'] / total_per_year) * 100

# Schritt 2: Linienplot erstellen
fig3, ax3 = plt.subplots()
sns.lineplot(data=count_per_year_fuel, x='year', y='percentage', hue='fuel', ax=ax3)

# Beschriftungen hinzufügen
ax3.set_title('Prozentuale Verteilung der Kraftstofftypen')
ax3.set_xlabel('Jahr')
ax3.set_ylabel('Prozentualer Anteil')
plt.legend(title='Kraftstofftypen', loc='lower left', prop={'size':5})

# Zeige den Plot in Streamlit
st.pyplot(fig3)

st.subheader("\nKorrelationen")

st.markdown(f""" Zunächst lassen sich einige Korrelationen anhand der Korrelationsmatrix ablesen:
- Mit zunehmender Leistung der Autos steigt auch der Preis.
- Höhere Kilometerstände lassen den Preis sinken.
- Über die Jahre steigt das Preisniveau allgemein
- Autos mit neuerem Verkaufsjahr haben tendenziell niedrigere Kilometerstände.          
""")

# Wähle nur numerische Spalten für die Korrelationsmatrix
numerical_data = data.select_dtypes(include=['int64', 'float64'])

# Berechne die Korrelationsmatrix für die numerischen Spalten
corr = numerical_data.corr()

# Visualisiere die Korrelationsmatrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Korrelationsmatrix')

# Zeige die Heatmap in Streamlit
st.pyplot(plt)

st.header("Top 5 Marken")

st.markdown("""Für die folgenden Analysen ziehen wir ausschließlich die 5 meistverkauften Automarken auf autoscout24 in Betracht.""")

col1, col2 = st.columns(2)
# Die Top 5 Marken nach Verkaufszahlen sortieren

top_5 = data['make'].value_counts().head(5)

with col1:
    # Die Grafik erstellen
    fig5, ax5 = plt.subplots()
    bars5 = ax5.bar(top_5.index, top_5.values)
    ax5.set_ylabel('Verkaufszahlen')
    ax5.set_xlabel('Marke')
    ax5.set_title('Top 5 meistverkaufte Autos')
    ax5.tick_params(axis='x', rotation=45)

    for bar in bars5:
        ax5.annotate(f'{bar.get_height()}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 1),  # Vertikaler Abstand über den Balken
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Die Grafik in Streamlit anzeigen
    st.pyplot(fig5)

# Top 5 Marken nach Verkaufszahlen identifizieren
top_5_marken = data['make'].value_counts().head(5).index

# DataFrame nur mit den Top 5 Marken erstellen
top_5_data = data[data['make'].isin(top_5_marken)]

# Durchschnittlichen Preis für jede der Top 5 Marken berechnen
mean_prices = top_5_data.groupby('make')['price'].mean().reset_index()
mean_prices['price'] = mean_prices['price'].round(2)

# Sortiere die Marken nach dem durchschnittlichen Preis
mean_prices_sorted = mean_prices.sort_values(by='price', ascending=False)


with col2:
    # Erstelle das Balkendiagramm
    fig6, ax6 = plt.subplots()
    bars6 = ax6.bar(mean_prices_sorted['make'], mean_prices_sorted['price'])
    ax6.set_xlabel('Marke')
    ax6.set_ylabel('Durchschnittlicher Preis in €')
    ax6.set_title('Durchschnittspreise der 5 meistverkauften Marken')
    ax6.tick_params(axis='x', rotation=45)  # Drehe die Markennamen für bessere Lesbarkeit

    for bar in bars6:
        ax6.annotate(f'{bar.get_height()}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 1),  # Vertikaler Abstand über den Balken
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Zeige den Plot in Streamlit
    st.pyplot(fig6)

st.header("Korrelationen Top 5")

top_5 = ['Volkswagen', 'Ford', 'Skoda', 'Renault', 'Opel']

st.subheader("Preisentwicklung unter Berücksichtigung ausgewählter Variablen")

st.subheader("Modell")

for marke in top_5:
    marke_data = data[data['make'] == marke]

    # Berechne den mittleren Preis für jedes Modell
    mean_prices = marke_data.groupby('model')['price'].mean().reset_index()
    # Sortiere die Modelle nach ihrem mittleren Preis
    mean_prices_sorted = mean_prices.sort_values(by='price')
    
    # Zeichne den Boxplot mit sortierten Modellen
    plt.figure(figsize=(14, 7))
    sorted_boxplot = sns.boxplot(x='model', y='price', data=marke_data, order=mean_prices_sorted['model'])
    plt.xticks(rotation=45)
    plt.title(f'{marke}')
    plt.xlabel('Modell')
    plt.ylabel('Preis')
    
    # Trendlinie hinzufügen: Mittelwert für jedes Modell
    # Da Boxplots und Linienplots unterschiedliche Achsen haben, normalisieren wir die x-Achsenposition der Linienpunkte
    x_positions = range(len(mean_prices_sorted))
    plt.plot(x_positions, mean_prices_sorted['price'], color='red', linestyle='-', marker='o', markersize=5, label='Mittlerer Preis')
    
    plt.legend()
    st.pyplot(plt)
    plt.close()

st.subheader("Verkaufsjahr")

for marke in top_5:
    marke_data = data[data['make'] == marke]
    # Berechne den mittleren Preis für jedes Jahr
    mean_prices = marke_data.groupby('year')['price'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    # Erstelle einen Boxplot für jedes Verkaufsjahr
    sns.boxplot(x='year', y='price', data=marke_data, order=mean_prices['year'].unique())
    
    plt.title(f'{marke}')
    plt.xlabel('Verkaufsjahr')
    plt.ylabel('Preis')
    plt.xticks(rotation=45)
    
    # Die x-Positionen basieren nun direkt auf der Länge von mean_prices['year']
    x_positions = np.arange(len(mean_prices['year']))
    
    # Stelle sicher, dass die Länge von x_positions und mean_prices['price'] übereinstimmt
    plt.plot(x_positions, mean_prices['price'], color='red', linestyle='-', marker='o', markersize=5, label='Mittlerer Preis')
    
    plt.legend()
    st.pyplot(plt)
    plt.close()

st.subheader("Kilometerstand")

for marke in top_5:
    marke_data = data[data['make'] == marke]
    
    # Kilometerstand in Intervalle einteilen und den mittleren Preis für jedes Intervall berechnen
    # Hier nehmen wir beispielhaft 10.000 km Intervalle
    marke_data['mileage_interval'] = (marke_data['mileage'] // 10000) * 10000
    mean_price_per_interval = marke_data.groupby('mileage_interval')['price'].mean().reset_index()
    mean_price_per_interval_sorted = mean_price_per_interval.sort_values('mileage_interval')
    plt.figure(figsize=(14, 8))
    plt.plot(mean_price_per_interval_sorted['mileage_interval'], mean_price_per_interval_sorted['price'], marker='o', linestyle='-', color='blue', label='Mittlerer Preis')
    z = np.polyfit(mean_price_per_interval_sorted['mileage_interval'], mean_price_per_interval_sorted['price'], 1)
    p = np.poly1d(z)
    plt.plot(mean_price_per_interval_sorted['mileage_interval'], p(mean_price_per_interval_sorted['mileage_interval']), linestyle='--', color='red', label='Trendlinie')
    plt.title(f'{marke}')
    plt.xlabel('Kilometerstand-Intervall')
    plt.ylabel('Mittlerer Preis')
    plt.legend()
    plt.xticks(rotation=45)

    st.pyplot(plt)
    plt.close()


st.subheader("Leistung (in PS)")

for marke in top_5:
    marke_data = data[data['make'] == marke]

    # Pferdestärke in Intervalle einteilen und den mittleren Preis für jedes Intervall berechnen
    # Hier nehmen wir beispielhaft Intervalle basierend auf der Verteilung der 'hp' Werte
    hp_intervals = pd.qcut(marke_data['hp'], q=10, duplicates='drop')  # Vermeidung von Duplikaten in den Kategorien
    marke_data['hp_interval'] = hp_intervals
    mean_price_per_hp_interval = marke_data.groupby('hp_interval')['price'].mean().reset_index()
    mean_price_per_hp_interval['hp_interval'] = mean_price_per_hp_interval['hp_interval'].apply(lambda x: x.mid).astype(int)  # Verwende den Mittelpunkt des Intervalls für die Darstellung
    mean_price_per_hp_interval_sorted = mean_price_per_hp_interval.sort_values('hp_interval')

    plt.figure(figsize=(14, 8))
    plt.plot(mean_price_per_hp_interval_sorted['hp_interval'], mean_price_per_hp_interval_sorted['price'], marker='o', linestyle='-', color='blue', label='Mittlerer Preis')

    # Berechne die Trendlinie
    z = np.polyfit(mean_price_per_hp_interval_sorted['hp_interval'], mean_price_per_hp_interval_sorted['price'], 1)
    p = np.poly1d(z)
    plt.plot(mean_price_per_hp_interval_sorted['hp_interval'], p(mean_price_per_hp_interval_sorted['hp_interval']), linestyle='--', color='red', label='Trendlinie')

    plt.title(f'{marke} - Preis nach HP')
    plt.xlabel('HP (Pferdestärke)')
    plt.ylabel('Mittlerer Preis')
    plt.legend()
    plt.xticks(rotation=45)

    st.pyplot(plt)
    plt.close()

st.subheader("Angebotstyp")

for marke in top_5:
    marke_data = data[data['make'] == marke]
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='offerType', y='price', data=marke_data, order=sorted(marke_data['offerType'].unique()))
    plt.title(f'{marke}')
    plt.xlabel('Angebotstyp')
    plt.ylabel('Preis')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()

### Machine Learning

st.header("Lineare Regression: Preisvorhersage")

st.markdown("""Die folgenden Modelle der linearen Regression werden auf das Verkaufsjahr, das Modell und den Kilometerstand trainiert.""")

# Filtere den DataFrame, um nur die Top 5 Marken zu behalten
top_5_marken = ['Volkswagen', 'Ford', 'Skoda', 'Renault', 'Opel']
df_top_5 = data[data['make'].isin(top_5_marken)]

# Vorbereitung des Preprocessors und des Modells
preprocessor = ColumnTransformer(transformers=[
    ('model', OneHotEncoder(handle_unknown='ignore'), ['model'])
], remainder='passthrough')

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Erstelle Spalten für die Plots in Streamlit
col1, col2 = st.columns(2)

# Schleife über jede Marke
for i, marke in enumerate(top_5_marken, start=1):
    # Filtere nach Marke
    df_marke = df_top_5[df_top_5['make'] == marke]
    
    # Wähle Features und Ziel
    X = df_marke[['year', 'model', 'mileage']]
    y = df_marke['price']
    
    # Teile in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    # Trainiere das Modell
    model_pipeline.fit(X_train, y_train)
    
    # Vorhersage und Evaluation
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    with col1 if i % 2 != 0 else col2:
        # Plotte die Ergebnisse
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
        ax.set_xlabel('Wahrer Preis')
        ax.set_ylabel('Vorhergesagter Preis')
        st.pyplot(fig)

st.header("Lineare Regression: Preisvorhersage mit Datensatz bis 15.000€")

# Filtere den DataFrame für Autos unter 15.000€
df_filtered = data[data['price'] < 15000]

# Behalte nur die Top 5 Marken
df_filtered = df_filtered[df_filtered['make'].isin(top_5_marken)]

# Vorbereitung des Preprocessors und des Modells
preprocessor = ColumnTransformer(transformers=[
    ('model', OneHotEncoder(handle_unknown='ignore'), ['model'])
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Erstelle eine Spalte für die Plots in Streamlit
col1, col2 = st.columns(2)

# Schleife über jede Marke
for i, marke in enumerate(top_5_marken, start=1):
    # Filtere nach Marke
    df_marke = df_filtered[df_filtered['make'] == marke]
    
    # Wähle Features und Ziel
    X = df_marke[['year', 'model', 'mileage']]
    y = df_marke['price']
    
    # Teile in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    
    # Trainiere das Modell
    model.fit(X_train, y_train)
    
    # Vorhersage und Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Plotte die Ergebnisse
    with col1 if i % 2 != 0 else col2:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
        ax.set_xlabel('Tatsächlicher Preis')
        ax.set_ylabel('Vorhergesagter Preis')
        st.pyplot(fig)

st.header("Erweiterte Lineare Regression: Preisvorhersage mit HP")

# Vorbereitung des Preprocessors für numerische Features und One-Hot-Encoding für 'model'
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Füge den Imputer hinzu
        ('scaler', StandardScaler())  # Skaliere die numerischen Features
    ]), ['hp', 'year', 'mileage']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['model'])
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Erstelle Spalten für die Visualisierung der Ergebnisse in Streamlit
col1, col2 = st.columns(2)

# Trainiere und bewerte Modelle für jede Marke
for i, marke in enumerate(top_5_marken):
    df_marke = df_top_5[df_top_5['make'] == marke]
    X = df_marke[['hp', 'year', 'mileage', 'model']]
    y = df_marke['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trainiere das Modell
    model_pipeline.fit(X_train, y_train)

    # Vorhersage und Evaluation
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Visualisiere die Ergebnisse
    if i % 2 == 0:
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.3)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
            ax.set_xlabel('Wahrer Preis')
            ax.set_ylabel('Vorhergesagter Preis')
            st.pyplot(fig)
    else:
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.3)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
            ax.set_xlabel('Wahrer Preis')
            ax.set_ylabel('Vorhergesagter Preis')
            st.pyplot(fig)

st.header("Erweiterte Lineare Regression: Preisvorhersage mit HP für <15.000€")

# Filtere den DataFrame für Autos unter 15.000€
df_filtered = data[data['price'] <= 15000]

# Behalte nur die Top 5 Marken
df_filtered_top_5 = df_filtered[df_filtered['make'].isin(top_5_marken)]

# Vorbereitung des Preprocessors und des Modells mit SimpleImputer
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Füge den Imputer hinzu
        ('scaler', StandardScaler())  # Skaliere die numerischen Features
    ]), ['hp', 'year', 'mileage']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['model'])
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Erstelle Spalten für die Visualisierung der Ergebnisse in Streamlit
col1, col2 = st.columns(2)

# Trainiere und bewerte Modelle für jede Marke innerhalb der Preisgrenze
for i, marke in enumerate(top_5_marken):
    df_marke = df_filtered_top_5[df_filtered_top_5['make'] == marke]
    X = df_marke[['hp', 'year', 'mileage', 'model']]
    y = df_marke['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trainiere das Modell
    model_pipeline.fit(X_train, y_train)

    # Vorhersage und Evaluation
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Visualisiere die Ergebnisse
    if i % 2 == 0:
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.3)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
            ax.set_xlabel('Wahrer Preis')
            ax.set_ylabel('Vorhergesagter Preis')
            st.pyplot(fig)
    else:
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.3)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
            ax.set_xlabel('Wahrer Preis')
            ax.set_ylabel('Vorhergesagter Preis')
            st.pyplot(fig)

st.header("Random Forest: Preisvorhersage")

def train_and_plot_rf(df, marke, col):
    """Trainiert ein Random Forest Modell und visualisiert die Ergebnisse für eine gegebene Marke in einer bestimmten Streamlit-Spalte."""
    df_marke = df[df['make'] == marke]
    X = df_marke[['year', 'model', 'mileage', 'hp']]
    y = df_marke['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=[
            ('model', OneHotEncoder(handle_unknown='ignore'), ['model'])
        ], remainder='passthrough')),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=43))
    ])
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    with col:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Wahrer Preis')
        ax.set_ylabel('Vorhergesagter Preis')
        ax.set_title(f"{marke} (RMSE: {rmse:.2f}, MAE: {mae:.2f})")
        st.pyplot(fig)


# Erstelle Spalten für die Visualisierung in Streamlit
col1, col2 = st.columns(2)

# Trainiere Random Forest Modelle und visualisiere die Ergebnisse für jede Marke
for i, marke in enumerate(top_5_marken, start=1):
    if i % 2 != 0:
        train_and_plot_rf(data, marke, col1)  # `data` ist der gesamte ungefilterte Datensatz
    else:
        train_and_plot_rf(data, marke, col2)

st.header("PCA zur Reduktion der Dimensionalität der Features")

# Isoliere die numerischen Features und wende One-Hot-Encoding auf die kategoriale Variable 'model' an
df_encoded = pd.get_dummies(df_top_5[['year', 'mileage', 'model', 'hp']], columns=['model'], drop_first=True)

# Skaliere die Features vor der PCA, da PCA empfindlich auf die Skalierung der Features reagiert
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Wähle die Anzahl der Hauptkomponenten, z.B. 2 für die Visualisierung
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_scaled)

# Erstelle einen DataFrame für die Hauptkomponenten
df_pca = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Füge das Ziel (Preis) hinzu, um später nach ihm zu färben oder zu gruppieren
df_pca['price'] = df_top_5['price'].values

plt.figure(figsize=(10, 7))
sns.scatterplot(x='principal component 1', y='principal component 2', data=df_pca, hue='price', palette='rainbow')
plt.title('PCA der Automarktdaten')
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.show()
st.pyplot(plt)

st.header("Logistische Regression: Preisvorhersage (Preisschwelle: 15.000€)")

preis_schwellenwert = 15000  # Beispielwert

data=data.dropna()

# Binäre Zielvariable basierend auf dem Preis-Schwellenwert erstellen
data['hochpreisig'] = (data['price'] > preis_schwellenwert).astype(int)

# Filtere den DataFrame, um nur Zeilen mit den Top 5 Marken zu behalten
top_5_marken = data['make'].value_counts().head(5).index
df_top_5 = data[data['make'].isin(top_5_marken)]

# Wende One-Hot-Encoding auf 'model' und Label-Encoding auf 'make' an
df_encoded = pd.get_dummies(df_top_5, columns=['model'], drop_first=True)

# Features und Zielvariable auswählen
X = df_encoded[['hp', 'year', 'mileage'] + [col for col in df_encoded.columns if 'model_' in col]]
y = df_encoded['hochpreisig']

# Teile die Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Initialisiere und trainiere das logistische Regressionsmodell
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Mache Vorhersagen auf dem Testset
y_pred = logistic_model.predict(X_test)

# Berechne Gütekriterien
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualisiere die Ergebnisse
fig10, ax10 = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
ax10.set_xlabel('Vorhergesagt')
ax10.set_ylabel('Wahr')
ax10.set_title(f'Logistische Regression: Genauigkeit = {accuracy:.2f}')
st.pyplot(fig10)

st.header("Logistische Regression: Preisvorhersage in 10 Kategorien")

# Preis in Perzentile einteilen, um 10 Kategorien zu erstellen
data['preis_kategorie'] = pd.qcut(data['price'], q=3, labels=False)

data = data.dropna()

# Filtere den DataFrame, um nur Zeilen mit den Top 5 Marken zu behalten
top_5_marken = data['make'].value_counts().head(5).index
df_top_5 = data[data['make'].isin(top_5_marken)]

# Wende One-Hot-Encoding auf 'model' an
df_encoded = pd.get_dummies(df_top_5, columns=['model'], drop_first=True)

# Features und Zielvariable auswählen
X = df_encoded[['hp', 'year', 'mileage'] + [col for col in df_encoded.columns if 'model_' in col]]
y = df_encoded['preis_kategorie']

# Teile die Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Initialisiere und trainiere das logistische Regressionsmodell für multinomiale Klassifikation
logistic_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
logistic_model.fit(X_train, y_train)

# Mache Vorhersagen auf dem Testset
y_pred = logistic_model.predict(X_test)

# Berechne Gütekriterien
accuracy = accuracy_score(y_test, y_pred)

# Erstelle und visualisiere die Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)

fig11, ax11 = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
ax11.set_xlabel('Vorhergesagt')
ax11.set_ylabel('Wahr')
ax11.set_title(f'Multinomiale logistische Regression: Genauigkeit = {accuracy:.2f}')
st.pyplot(fig11)
