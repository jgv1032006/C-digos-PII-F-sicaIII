#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Practica 2: Adveccion de la temperatura y modelos de pronostico.
#*****************************************************************

# %% Bloque 1: Importando librerias

import xarray as xr
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# %% Bloque 2: Cargando los datos.
# Si trabajamos desde Google Drive (el archivo subido a la nube):

# Montar Google Drive para acceder al archivo de datos.
# Solo valido desde Notebook de google colab.
#from google.colab import drive
#drive.mount('/content/drive')
#file_path = '/content/drive/MyDrive/Meteo_UA/ECMWF_Jul2014.nc'

# Para trabajar desde spyder probar.
#url del archivo publico en Drive y ocupa menos de 100 megabytes.
#url = "https://drive.google.com/file/d/1x8iZltUbBa-1R174HMjfr_oobUchqy3T/view?usp=share_link"
#file_path='https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])

# Si trabajamos con el archivo descargado localmente en el ordenador:
# Poned la ruta que lleve al archivo de la practica en file_path.
#file_path = '.../Escritorio/ECMWF_Jul2014.nc'
#file_path = "/Users/alumno/Desktop/P2/ECMWF_Jul2014.nc"
file_path = file_path = "C:/Users/usuario/Desktop/P2 FÍSICA III/ECMWF_Jul2014.nc"

# Abrir el archivo NetCDF.
dataset = xr.open_dataset(file_path)

# %% Bloque 3: Extrayendo las variables principales.

# Mostrar las variables del dataset para entender su estructura.
#print(dataset)

# Mostrar los valores unicos de la variable de nivel.
levels = dataset['level'].values
print("Niveles de presion disponibles: \n", levels)

# Encontrar el indice del valor mas cercano a 500 hPa.
target_level = 500
index_n_hPa = abs(levels - target_level).argmin()
print(f"Indice del nivel de {target_level} hPa: {index_n_hPa}")
print(f"Valor del nivel mas cercano a {target_level} hPa: {levels[index_n_hPa]}")

# Las variables estaticas se llaman 'time', 'level', 'lat', 
# y 'lon'.
# Seleccionar el primer tiempo y el nivel de n hPa.
time = dataset['time'][0]
level = dataset['level'].sel(level=target_level)

levels = dataset['level']
#print(levels)

# Extraer la temperatura y el campo de velocidad del viento 
# para el primer tiempo y el nivel de n hPa.
temperature = dataset['t'].sel(time=time, level=level)
u_wind = dataset['u'].sel(time=time, level=level)
v_wind = dataset['v'].sel(time=time, level=level)

# Convertir a unidades de grados Celsius (si es necesario).
temperature_celsius = temperature.metpy.convert_units('degC')

# Extraer latitudes y longitudes.
latitudes = dataset['latitude']
longitudes = dataset['longitude']

# %% Bloque 4: Calculo numerico de la adveccion usando NumPy.

# Transformar el espacio lat/lon en espacio fisico x/y 
# para calcular dx y dy es el desafio critico
# de este apartado. En lugar de aproximarnos toscamente
# a un factor de conversion constante o ejecutar 
# una formula haversine potencialmente lenta y complicada, 
# aprovecharemos la funcion "transform_points" de Cartopy 
# para transformar los puntos cilindricos de
# latitud/longitud en un sistema de coordenadas que 
# utiliza coordenadas de distancia x/y.  Utilizando 
# la proyeccion LambertConformal para el dominio,
# podemos establecer el argumento de longitud central 
# en algo cercano al centro del dominio.

proj = ccrs.LambertConformal(central_longitude=-90)
lons,lats = np.meshgrid(longitudes,latitudes)
output = proj.transform_points(ccrs.PlateCarree(),lons,lats)
x_ref,y_ref = output[:,:,0], output[:,:,1]

# Calcular la derivada de x e y en ambas direcciones.
gradx_magnitude = np.gradient(x_ref,axis=1)
grady_magnitude = np.gradient(y_ref,axis=0)

#xr.DataArray(gradx_magnitude).plot()
gradx = xr.DataArray(gradx_magnitude)
grady = xr.DataArray(grady_magnitude)

# Calcular la derivada de la temperatura en x e y.
dtx_magnitude = np.gradient(temperature[:,:],axis=1)
dty_magnitude = np.gradient(temperature[:,:],axis=0)

dtx = xr.DataArray(dtx_magnitude)
dty = xr.DataArray(dty_magnitude)

advct_x = -u_wind[:,:].values * (dtx.values/gradx.values)
advct_y = -v_wind[:,:].values * (dty.values/grady.values)

# Calcular la adveccion.
Tadv_numpy = advct_x + advct_y


# %% Bloque 5: Calculo numerico de la adveccion usando MetPy.

# Calcula la derivada de temperatura en longitud y latitud.
dtdx = mpcalc.first_derivative(temperature,axis=1)
dtdy = mpcalc.first_derivative(temperature,axis=0)

advct_x_metpy = -u_wind*dtdx
advct_y_metpy = -v_wind*dtdy

# Calcular la adveccion.
Tadv_metpy = advct_x_metpy+advct_y_metpy
#print(Tadv_metpy)

# %% Bloque 6: Pronostico de la temperatura.

# Calculamos la temperatura en las siguientes n horas 
# teniendo en cuenta la temperatura inicial 
# y el campo de adveccion.
# n tiene que ser un múltiplo de 6 entre 6 y 24 ambos incluídos.

horas = 24
segundos = horas*60*60
temperature_advt = temperature.values \
                   + (advct_x + advct_y) * segundos
                   
temperature_advt_metpy = temperature \
+ Tadv_metpy.to_numpy() * segundos

# Se obtiene la temperatura del fichero para las
# siguientes n horas, y asi comparar con la prediccion realizada.
idx_hora = horas//6
time = dataset['time'][idx_hora] # tiempo a t + horas h
level = dataset['level'].sel(level=target_level)

temperature_next = dataset['t'].sel(time=time, level=level)
u_wind_next = dataset['u'].sel(time=time, level=level)
v_wind_next = dataset['v'].sel(time=time, level=level)

temperature_advt_celcius = temperature_advt - 273.15
temperature_advt_celcius_metpy = temperature_advt_metpy - 273.15
temperature_next_celcius = temperature_next - 273.15

# %% Bloque 7: Representacion de resultados de adveccion.
 
crs = ccrs.PlateCarree()
 
clevs_n_tmpc_diff = np.arange(-0.0010, 0.0010, 0.0001)
 
# Funcion usada  para crear las subfiguras del mapa.
def plot_background(ax):
     #ax.set_extent([-40., 40., 20., 60.])
     ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
     linewidth=0.5)
     ax.add_feature(cfeature.STATES, linewidth=0.5)
     ax.add_feature(cfeature.BORDERS, linewidth=0.5)
     return ax
 
# Crea la figura y representa el fondo en las distintas subfiguras.
fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 6.5),
constrained_layout=True,subplot_kw={'projection': crs})
 
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)
 
# Subfigura superior izquierda - Adveccion temperatura NumPy.
cf1 = axlist[0].contourf(lons, lats, Tadv_numpy, 
clevs_n_tmpc_diff, cmap=plt.cm.coolwarm, 
transform=ccrs.PlateCarree())
axlist[0].set_title('Adveccion de temperatura - NumPy',
fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[0], orientation='horizontal',
shrink=0.74, pad=0)
cb1.set_label(r'$^{\circ}$C/s', size=10)
cb1.ax.tick_params(labelsize=8)

# Subfigura superior derecha - Adveccion temperatura MetPy.
cf2 = axlist[1].contourf(lons, lats, Tadv_metpy,
clevs_n_tmpc_diff,cmap=plt.cm.coolwarm,
transform=ccrs.PlateCarree())
axlist[1].set_title('Adveccion de temperatura - Metpy',
fontsize=16)
cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal',
shrink=0.74, pad=0)
cb2.set_label(r'$^{\circ}$C/s', size=10)
cb2.ax.tick_params(labelsize=8)

# Espaciado vertical de las subfiguras.
fig.set_constrained_layout_pads(w_pad=0., h_pad=0.1, hspace=0.,
wspace=0.)

# Mostramos las figuras.
plt.savefig("Adveccion_{target_level}.png", dpi=300)
plt.show()

# %% Bloque 8: Representacion de resultados. Pronostico vs datos.

who="numpy"

crs = ccrs.PlateCarree()

clevs_n_tmpc_diff = np.arange(-0.0010, 0.0010, 0.0001)
if horas != 24:
    clevs_n_tmpc = np.arange(-30.0, 30.0, 5.0)
else:
    clevs_n_tmpc = np.arange(-70.0, 70.0, 5.0)

# Funcion para crear las figuras.
def plot_background(ax):
    #ax.set_extent([-40., 40., 20., 60.])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
    linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax
    
# Crea y representa el fondo en las distintas subfiguras.
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20, 13),
constrained_layout=True, subplot_kw={'projection': crs})
    
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

if who == "numpy":

    # Subfigura superior izquierda - Adveccion NumPy.
    cf1 = axlist[0].contourf(lons, lats, Tadv_numpy,
    clevs_n_tmpc_diff, cmap=plt.cm.coolwarm,
    transform=ccrs.PlateCarree())
    axlist[0].set_title('Adveccion de temperatura con NumPy', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=axlist[0],
    orientation='horizontal', shrink=0.74, pad=0)
    cb1.set_label(r'$^{\circ}$C/s', size=14)
    cb1.ax.tick_params(labelsize=12)
    
    # Subfigura superior derecha - Temperatura t=0
    cf2 = axlist[1].contourf(lons, lats, temperature - 273.15,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[1].set_title('Temperatura (t = 0)-'+ \
    dataset['time'][0].dt.strftime('%d %B %Y %H:%MZ').values,
    fontsize=16)
    cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal',
    shrink=0.74, pad=0)
    cb2.set_label(r'$^{\circ}$C/s', size=14)
    cb2.ax.tick_params(labelsize=12)
    
    # Figura inferior izquierda - Pronostico temperatura a t + n h
    cf3 = axlist[2].contourf(lons, lats, temperature_advt_celcius,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[2].set_title(f'Pronostico de temperatura (t + {horas} h)',
    fontsize=16)
    cb3 = fig.colorbar(cf3, ax=axlist[2],
    orientation='horizontal', shrink=0.74, pad=0)
    cb3.set_label(r'$^{\circ}$C/s', size=14)
    cb3.ax.tick_params(labelsize=12)
    
    # Figura inferior derecha - Temperatura a t + n h.
    cf4 = axlist[3].contourf(lons, lats, temperature_next_celcius,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[3].set_title(f'Temperatura (t + {horas} h)', fontsize=16)
    cb4 = fig.colorbar(cf4, ax=axlist[3], orientation='horizontal',
    shrink=0.74, pad=0)
    cb4.set_label(r'$^{\circ}$C/s', size=14)
    cb4.ax.tick_params(labelsize=12)
    
    # Espaciado verticical entre subfiguras.
    fig.set_constrained_layout_pads(w_pad=0., h_pad=0.1, hspace=0.,
    wspace=0.)
    
    # Muestra el grafico.
    plt.savefig("Pronostico+{horas}.png", dpi=300)
    plt.show()
    
else:
    
    # Subfigura superior izquierda - Adveccion MetPy.
    cf1 = axlist[0].contourf(lons, lats, Tadv_metpy,
    clevs_n_tmpc_diff, cmap=plt.cm.coolwarm,
    transform=ccrs.PlateCarree())
    axlist[0].set_title('Adveccion de temperatura con MetPy', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=axlist[0],
    orientation='horizontal', shrink=0.74, pad=0)
    cb1.set_label(r'$^{\circ}$C/s', size=14)
    cb1.ax.tick_params(labelsize=12)
    
    # Subfigura superior derecha - Temperatura t=0
    cf2 = axlist[1].contourf(lons, lats, temperature - 273.15,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[1].set_title('Temperatura (t = 0)-'+ \
    dataset['time'][0].dt.strftime('%d %B %Y %H:%MZ').values,
    fontsize=16)
    cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal',
    shrink=0.74, pad=0)
    cb2.set_label(r'$^{\circ}$C/s', size=14)
    cb2.ax.tick_params(labelsize=12)
    
    # Figura inferior izquierda - Pronostico temperatura a t + n h
    cf3 = axlist[2].contourf(lons, lats, temperature_advt_celcius_metpy,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[2].set_title(f'Pronostico de temperatura (t + {horas} h)',
    fontsize=16)
    cb3 = fig.colorbar(cf3, ax=axlist[2],
    orientation='horizontal', shrink=0.74, pad=0)
    cb3.set_label(r'$^{\circ}$C/s', size=14)
    cb3.ax.tick_params(labelsize=12)
    
    # Figura inferior derecha - Temperatura a t + n h.
    cf4 = axlist[3].contourf(lons, lats, temperature_next_celcius,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[3].set_title(f'Temperatura (t + {horas} h)', fontsize=16)
    cb4 = fig.colorbar(cf4, ax=axlist[3], orientation='horizontal',
    shrink=0.74, pad=0)
    cb4.set_label(r'$^{\circ}$C/s', size=14)
    cb4.ax.tick_params(labelsize=12)
    
    # Espaciado verticical entre subfiguras.
    fig.set_constrained_layout_pads(w_pad=0., h_pad=0.1, hspace=0.,
    wspace=0.)
    
    # Muestra el grafico.
    plt.savefig("Pronostico+{horas}.png", dpi=300)
    plt.show()
    

# %% Bloque 9: Representacion de resultados. Pronostico vs datos con vientos.

crs = ccrs.PlateCarree()

clevs_n_tmpc_diff = np.arange(-0.0010, 0.0010, 0.0001)
if horas != 24:
    clevs_n_tmpc = np.arange(-30.0, 30.0, 5.0)
else:
    clevs_n_tmpc = np.arange(-70.0, 70.0, 5.0)

# Funcion para crear las figuras.
def plot_background(ax):
    #ax.set_extent([-40., 40., 20., 60.])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
    linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax
    
# Crea y representa el fondo en las distintas subfiguras.
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20, 13),
constrained_layout=True, subplot_kw={'projection': crs})
    
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

if who == "numpy":

    # Subfigura superior izquierda - Adveccion NumPy.
    cf1 = axlist[0].contourf(lons, lats, Tadv_numpy,
    clevs_n_tmpc_diff, cmap=plt.cm.coolwarm,
    transform=ccrs.PlateCarree())
    axlist[0].set_title('Adveccion de temperatura con NumPy', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=axlist[0],
    orientation='horizontal', shrink=0.74, pad=0)
    cb1.set_label(r'$^{\circ}$C/s', size=14)
    cb1.ax.tick_params(labelsize=12)
    
    # Subfigura superior derecha - Temperatura t=0
    cf2 = axlist[1].contourf(lons, lats, temperature - 273.15,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[1].set_title('Temperatura (t = 0)-'+ \
    dataset['time'][0].dt.strftime('%d %B %Y %H:%MZ').values,
    fontsize=16)
    cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal',
    shrink=0.74, pad=0)
    cb2.set_label(r'$^{\circ}$C/s', size=14)
    cb2.ax.tick_params(labelsize=12)
    # Flechas direccion del viento.
    sknum = 5 # Separación entre flechas.
    skip=(slice(None,None,sknum),slice(None,None,sknum))
    # Dibujar flechas.
    axlist[1].barbs(lons[skip], lats[skip],
              u_wind[skip], v_wind[skip],
              pivot='middle', color='black', transform=ccrs.PlateCarree())
    
    # Figura inferior izquierda - Pronostico temperatura a tiempo t + h
    cf3 = axlist[2].contourf(lons, lats, temperature_advt_celcius,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[2].set_title(f'Pronostico de temperatura (t + {horas} h)',
    fontsize=16)
    cb3 = fig.colorbar(cf3, ax=axlist[2],
    orientation='horizontal', shrink=0.74, pad=0)
    cb3.set_label(r'$^{\circ}$C/s', size=14)
    cb3.ax.tick_params(labelsize=12)
    
    # Figura inferior derecha - Temperatura real a tiempo t + h.
    cf4 = axlist[3].contourf(lons, lats, temperature_next_celcius,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[3].set_title(f'Temperatura (t + {horas} h)', fontsize=16)
    cb4 = fig.colorbar(cf4, ax=axlist[3], orientation='horizontal',
    shrink=0.74, pad=0)
    cb4.set_label(r'$^{\circ}$C/s', size=14)
    cb4.ax.tick_params(labelsize=12)
    # Dibujar flechas direccion viento.
    axlist[3].barbs(lons[skip], lats[skip],
              u_wind_next[skip], v_wind_next[skip],
              pivot='middle', color='black', transform=ccrs.PlateCarree())
    
    # Espaciado verticical entre subfiguras.
    fig.set_constrained_layout_pads(w_pad=0., h_pad=0.1, hspace=0.,
    wspace=0.)
    
    # Muestra el grafico.
    plt.savefig("Pronostico_viento_{horas}.png", dpi=300)
    plt.show()

else:

    # Subfigura superior izquierda - Adveccion NumPy.
    cf1 = axlist[0].contourf(lons, lats, Tadv_metpy,
    clevs_n_tmpc_diff, cmap=plt.cm.coolwarm,
    transform=ccrs.PlateCarree())
    axlist[0].set_title('Adveccion de temperatura con MetPy', fontsize=16)
    cb1 = fig.colorbar(cf1, ax=axlist[0],
    orientation='horizontal', shrink=0.74, pad=0)
    cb1.set_label(r'$^{\circ}$C/s', size=14)
    cb1.ax.tick_params(labelsize=12)
    
    # Subfigura superior derecha - Temperatura t=0
    cf2 = axlist[1].contourf(lons, lats, temperature - 273.15,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[1].set_title('Temperatura (t = 0)-'+ \
    dataset['time'][0].dt.strftime('%d %B %Y %H:%MZ').values,
    fontsize=16)
    cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal',
    shrink=0.74, pad=0)
    cb2.set_label(r'$^{\circ}$C/s', size=14)
    cb2.ax.tick_params(labelsize=12)
    # Flechas direccion del viento.
    sknum = 5 # Separación entre flechas.
    skip=(slice(None,None,sknum),slice(None,None,sknum))
    # Dibujar flechas.
    axlist[1].barbs(lons[skip], lats[skip],
              u_wind[skip], v_wind[skip],
              pivot='middle', color='black', transform=ccrs.PlateCarree())
    
    # Figura inferior izquierda - Pronostico temperatura a tiempo t + h
    cf3 = axlist[2].contourf(lons, lats, temperature_advt_celcius_metpy,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[2].set_title(f'Pronostico de temperatura (t + {horas} h)',
    fontsize=16)
    cb3 = fig.colorbar(cf3, ax=axlist[2],
    orientation='horizontal', shrink=0.74, pad=0)
    cb3.set_label(r'$^{\circ}$C/s', size=14)
    cb3.ax.tick_params(labelsize=12)
    
    # Figura inferior derecha - Temperatura real a tiempo t + h.
    cf4 = axlist[3].contourf(lons, lats, temperature_next_celcius,
    clevs_n_tmpc,cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    axlist[3].set_title(f'Temperatura (t + {horas} h)', fontsize=16)
    cb4 = fig.colorbar(cf4, ax=axlist[3], orientation='horizontal',
    shrink=0.74, pad=0)
    cb4.set_label(r'$^{\circ}$C/s', size=14)
    cb4.ax.tick_params(labelsize=12)
    # Dibujar flechas direccion viento.
    axlist[3].barbs(lons[skip], lats[skip],
              u_wind_next[skip], v_wind_next[skip],
              pivot='middle', color='black', transform=ccrs.PlateCarree())
    
    # Espaciado verticical entre subfiguras.
    fig.set_constrained_layout_pads(w_pad=0., h_pad=0.1, hspace=0.,
    wspace=0.)
    
    # Muestra el grafico.
    plt.savefig("Pronostico_viento_{horas}.png", dpi=300)
    plt.show()