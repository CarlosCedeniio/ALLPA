#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[1]:


get_ipython().system('pip install cdsapi')
get_ipython().system('pip install xarray')


# In[ ]:


import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '2m_temperature', 'total_precipitation', '2m_dewpoint_temperature', '10m_wind_speed'
        ],
        'year': [str(year) for year in range(2010, 2022)],  # Ajusta el año final según sea necesario
        'month': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12',
        ],
        'day': [
            '01', '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23', '24', '25', '26', '27',
            '28', '29', '30', '31',
        ],
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        'area': [
            1.5, -92.0, -5.0, -75.0,  # Coordenadas aproximadas para Ecuador: Norte, Oeste, Sur, Este
        ],
    },
    'historical_data.nc')  # Nombre del archivo de salida


# In[ ]:


import xarray as xr

data = xr.open_dataset('historical_data.nc')
data


# In[ ]:


# Acceder a una variable específica, por ejemplo, la temperatura a 2 metros
temperature = data['2m_temperature']

# Mostrar los primeros valores
print(temperature.values)

# Acceder a otras variables de manera similar
precipitation = data['total_precipitation']
dewpoint_temperature = data['2m_dewpoint_temperature']
# Para la velocidad del viento, primero verifica cómo están nombradas las componentes U y V en tu dataset
# wind_speed_u = data['u_component_of_wind_at_10m']
# wind_speed_v = data['v_component_of_wind_at_10m']


# In[ ]:


import pandas as pd

# Suponiendo que deseas trabajar con la temperatura y la precipitación
df_temp = temperature.to_dataframe().reset_index()
df_precip = precipitation.to_dataframe().reset_index()

# Combina los DataFrames si es necesario, asegurándote de que coincidan en su índice temporal
df_combined = pd.merge(df_temp, df_precip, on=["time", "latitude", "longitude"], how="inner")
print(df_combined.head())


# In[ ]:


df['fecha'] = pd.to_datetime(df['fecha'])
df.set_index('fecha', inplace=True)

# Asumiendo que tu dataset tiene columnas nombradas como 'temperatura', 'precipitacion', 'humedad', 'velocidad_viento'
variables_de_interes = ['temperatura', 'precipitacion', 'humedad', 'velocidad_viento']
data = df[variables_de_interes]


# In[ ]:


# Preparar los datos para LSTM
def crear_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# In[ ]:


# Escalar datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# In[ ]:


time_steps = 10
X, y = crear_dataset(pd.DataFrame(data_scaled[:, 1:]), pd.DataFrame(data_scaled[:, 0]), time_steps)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # Dividir el temp en dos partes iguales


# In[ ]:


# LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')


# In[ ]:


# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)


# In[ ]:


# Predicción
y_pred = model.predict(X_test)


# In[ ]:


# Escalar inverso para obtener predicciones en la escala original
# Nota: Ajusta la lógica de inversión de escala según cómo hayas escalado tus datos.
y_test_inv = scaler.inverse_transform(np.concatenate((y_test, X_test[:, :, 1:]), axis=1))[:, 0]
y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, X_test[:, :, 1:]), axis=1))[:, 0]


# In[ ]:


# Rendimiento de tu modelo en el conjunto de prueba
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
rmse

