# Tratamiento de datos
import pandas as pd
import numpy as np

import requests
import json

from datetime import date
from datetime import datetime as dt
import calendar
import holidays
import locale
locale.setlocale(locale.LC_ALL, "es_ES.UTF-8")
import time

# Metricas
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Visualización/Gráficos
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# Configuración de alertas
import warnings
warnings.filterwarnings('ignore')

today=dt.now().isoformat()[:10]



def reeapi_by_hour(start_date, end_date):
    '''
    La función extrae datos de la API Red Eléctrica Española (REE). Extrae la demanda por hora en MWh.

    Parameters:
    -----------------
    start_date (str) : fecha de inicio, formato: 'YYYY-MM-DD'
    end_date (str) : fecha de final, formato: 'YYYY-MM-DD'

    Returns:
    -----------------
    df : pandas.core.frame.DataFrame
    
    '''
    api_url=f'https://apidatos.ree.es/es/datos/demanda/evolucion?start_date={start_date}T00:00&end_date={end_date}T23:00&time_trunc=hour&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741'
    response = requests.get(api_url)

    if response.status_code==200:
        content=response
        print(f'Status code: {response.status_code}. Request from {start_date} to {end_date}: ¡ACCEPTED!')
    else:
        print(f'Status code: {response.status_code}. Request from {start_date} to {end_date}: ¡DENIED!')

    date=[content.json()['included'][0]['attributes']['values'][i]['datetime'][:19] for i in range(len(content.json()['included'][0]['attributes']['values']))]
    value=[content.json()['included'][0]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]
    df=pd.DataFrame(list(zip(date,value)), columns=['Fecha','Demanda'])
    return df



def datos_serie_temporal(df):
    '''
    Crea distinos campos/columnas con información relativa a una fecha (año, mes, día del año, día de la semana, hora, semana del año, cuatrimestre)
    
    df.index.dayofweek : Monday:0 -> Sunday:6
    df.index : {1:Enero-Marzo, 2:Abril-Junio, 3:Julio-Septiembre, 4:Octubre-Noviembre}
    df.index.iscoalendar().week : Semana 1 a 52 (por año)

    Parameters:
    ---------------
    df : pandas.core.frame.DataFrame

    Returns:
    ---------------
    df : pandas.core.frame.DataFrame
    '''
    df['Año'] = df.index.year
    df['Mes'] = df.index.month
    df['Dia_año'] = df.index.dayofyear
    df['Dia_semana'] = df.index.dayofweek
    df['Hora'] = df.index.hour
    df['Cuartos'] = df.index.quarter
    df['Semana_año'] = df.index.isocalendar().week
    return df



def festivos_y_fin_de_semana(df):
    '''
    Crea dos campos/columnas donde se identifican los días correspondientes a festivos nacionales y a findes de semana.

    Parameters:
    ---------------
    df : pandas.core.frame.DataFrame

    Returns:
    ---------------
    df : pandas.core.frame.DataFrame

    '''
    df['datetime'] = [df.index[i] for i in range(0,df.shape[0])]
    df['datetime'] = df['datetime'].astype(str)
    df['datetime'] = [df.iloc[i]['datetime'][:10] for i in range(0,df.shape[0])]
    
    es_holidays=holidays.Spain(years=[2015,2016,2017,2018,2019,2020,2021,2022])
    
    festivos=[]
    fin_de_semana=[]

    festivos=[1 if df.iloc[i]['datetime'] in es_holidays else 0 for i in range(0,df.shape[0])]
    fin_de_semana=[1 if df.iloc[i]['Dia_semana']==5 or df.iloc[i]['Dia_semana']==6 else 0 for i in range(0,df.shape[0])]
    
    df.drop(['datetime'], axis=1, inplace=True)
    df['fest_nacional']=festivos
    df['fin_de_semana']=fin_de_semana
    return df



def api_produccion(fecha_inicio,fecha_fin):
    '''
    Extrae datos de la producción por tecnología por día en un intervalo de tiempo de un mes.
    Esta función es sirve hasta 2021, a partir de ese año debemos usar la función api_producción_v2.

    Parameters:
    -----------------
    fecha_inicio (str) : Formato de la fecha YYYY-MM-DD
    fecha_fin (str) : Formato de la fecha YYYY-MM-DD

    Returns:
    df : pandas.core.frame.DataFrame
    
    '''
    lista_tecnologias=['Fecha','Hidraulica','Turbina_bombeo','Nuclear','Carbon','Motores_diesel','Turbina_de_gas',
                   'Turbina_de_vapor','Ciclo_combinado','Hidroeolica','Eolica','Solar_fotovoltaica','Solar_termica',
                   'Otras_renovables','Cogeneracion','Residuos_no_renovables','Residuos_renovables','Generacion_total']

    api_url=f'https://apidatos.ree.es/es/datos/generacion/estructura-generacion?start_date={fecha_inicio}T00:00&end_date={fecha_fin}T00:00&time_trunc=day'
    response = requests.get(api_url)

    if response.status_code==200:
        content=response
        print(f'Código de respuesta: {response.status_code}. Datos desde {fecha_inicio} hasta {fecha_fin}. ¡REQUEST ACCEPTED!')
    else:
        print(f'Código de respuesta: {response.status_code}. Datos desde {fecha_inicio} hasta {fecha_fin}. ¡REQUEST DENIED!')

    Fechas=[content.json()['included'][0]['attributes']['values'][i]['datetime'][:10] for i in range(len(content.json()['included'][0]['attributes']['values']))]
    Hidraulica=[content.json()['included'][0]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]
    Turbina_bombeo=[content.json()['included'][1]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]      
    Nuclear=[content.json()['included'][2]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]      
    Carbon=[content.json()['included'][3]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Motores_diesel=[content.json()['included'][5]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Turbina_de_gas=[content.json()['included'][6]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Turbina_de_vapor=[content.json()['included'][7]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Ciclo_combinado=[content.json()['included'][8]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Hidroeolica=[IndexErrorExcept(i,9,content) for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Eolica=[content.json()['included'][10]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Solar_fotovoltaica=[content.json()['included'][11]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Solar_termica=[IndexErrorExcept(i,12,content) for i in range(len(content.json()['included'][0]['attributes']['values']))]
    Otras_renovables=[content.json()['included'][13]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Cogeneracion=[content.json()['included'][14]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Residuos_no_renovables=[content.json()['included'][15]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Residuos_renovables=[content.json()['included'][16]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Generacion_total=[content.json()['included'][17]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]    

    df=pd.DataFrame(zip(Fechas,Hidraulica,Turbina_bombeo,Nuclear,Carbon,Motores_diesel,Turbina_de_gas,Turbina_de_vapor,Ciclo_combinado,Hidroeolica,Eolica,Solar_fotovoltaica,Solar_termica,Otras_renovables,Cogeneracion,Residuos_no_renovables,Residuos_renovables,Generacion_total), columns=lista_tecnologias)
    return df




def api_produccion_v2(fecha_inicio,fecha_fin):
    '''
    Extrae datos de la producción por tecnología por día en un intervalo de tiempo de un mes.
    Esta función sirve a partir del año 2022 (2022 inclusive).
    
    Parameters:
    -----------------
    fecha_inicio (str) : Formato de la fecha YYYY-MM-DD
    fecha_fin (str) : Formato de la fecha YYYY-MM-DD

    Returns:
    ---------------
    df : pandas.core.frame.DataFrame
    
    '''
    lista_tecnologias=['Fecha','Hidraulica','Turbina_bombeo','Nuclear','Carbon','Motores_diesel','Turbina_de_gas',
                   'Turbina_de_vapor','Ciclo_combinado','Hidroeolica','Eolica','Solar_fotovoltaica','Solar_termica',
                   'Otras_renovables','Cogeneracion','Residuos_no_renovables','Residuos_renovables','Generacion_total']

    api_url=f'https://apidatos.ree.es/es/datos/generacion/estructura-generacion?start_date={fecha_inicio}T00:00&end_date={fecha_fin}T00:00&time_trunc=day'
    response = requests.get(api_url)

    if response.status_code==200:
        content=response
        print(f'Código de respuesta: {response.status_code}. Datos desde {fecha_inicio} hasta {fecha_fin}. ¡REQUEST ACCEPTED!')
    else:
        print(f'Código de respuesta: {response.status_code}. Datos desde {fecha_inicio} hasta {fecha_fin}. ¡REQUEST DENIED!')

    Fechas=[content.json()['included'][0]['attributes']['values'][i]['datetime'][:10] for i in range(len(content.json()['included'][0]['attributes']['values']))]
    Hidraulica=[content.json()['included'][0]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]
    Turbina_bombeo=[content.json()['included'][1]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]      
    Nuclear=[content.json()['included'][2]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]      
    Carbon=[content.json()['included'][3]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Motores_diesel=[content.json()['included'][4]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Turbina_de_gas=[content.json()['included'][5]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Turbina_de_vapor=[content.json()['included'][6]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Ciclo_combinado=[content.json()['included'][7]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Hidroeolica=[IndexErrorExcept(i,8,content) for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Eolica=[content.json()['included'][9]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Solar_fotovoltaica=[content.json()['included'][10]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Solar_termica=[IndexErrorExcept(i,11,content) for i in range(len(content.json()['included'][0]['attributes']['values']))]
    Otras_renovables=[content.json()['included'][12]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Cogeneracion=[content.json()['included'][13]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Residuos_no_renovables=[content.json()['included'][14]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Residuos_renovables=[content.json()['included'][15]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]        
    Generacion_total=[content.json()['included'][16]['attributes']['values'][i]['value'] for i in range(len(content.json()['included'][0]['attributes']['values']))]    

    df=pd.DataFrame(zip(Fechas,Hidraulica,Turbina_bombeo,Nuclear,Carbon,Motores_diesel,Turbina_de_gas,Turbina_de_vapor,Ciclo_combinado,Hidroeolica,Eolica,Solar_fotovoltaica,Solar_termica,Otras_renovables,Cogeneracion,Residuos_no_renovables,Residuos_renovables,Generacion_total), columns=lista_tecnologias)
    return df



def IndexErrorExcept(i,n,content):
    '''
    Mneja el error IndexError que nos aparece en algunas ocasiones al extraer información de la API.
    
    Parameters:
    ----------------
    i=i
    n(int) = tecnología (12:Solar_termica, 9/8:Hidroeolica)

    Returns:
    -----------------
    No devuelve nada, sólo aplica la función.
    '''
    try:
        return content.json()['included'][n]['attributes']['values'][i]['value']
    except IndexError:
        return 0



def produccion_energia(df,column,list_prod):
    '''
    Concatena todos los dataframes, es necesario pasarlos en una lista de forma ordenada.
    ------------------
    Parameters:
    df :  nombre del dataframe final
    column : str  Nombre de la columna con datos de fecha (datetime)
    list_prod (list): lista de los dataframes a concatenar (deben estar ordenados)

    Returns:
    ----------------
    df : pandas.core.frame.DataFrame
    '''
    df=pd.concat(list_prod, ignore_index=True)
    df.index=range(df.shape[0])
    df[column]=pd.to_datetime(df[column])
    df=df.set_index(column)
    return df



def modify_zero(df):
    '''
    Interpola los valores cero (0) que se han producido al extraer información de la API.

    Parameters
    -----------
    df : pandas.core.frame.DataFrame

    Returns
    -----------
    df : pandas.core.frame.DataFrame
    '''
    df_columns=list(df.columns)
    
    for column in df_columns:
        for row in range(len(df)):
            if df[str(column)].iloc[row]==0:
                df[str(column)].iloc[row]=(df[str(column)].iloc[row-1]+df[str(column)].iloc[row+1])/2
    return df



def stats_info(df,lista):
    '''
    Devuelve información estadistica de un conjunto de columnas de un DataFrame.
    Debe introducirse una lista con el nombre de las columnas a analizar.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    lista : list
    
    Returns
    -------
    string : str 
    '''
    for i in lista:
        Q1=df[i].quantile(0.25)
        Q3=df[i].quantile(0.75)
        IQR=round(Q3-Q1,3)
        outlier_sup=(Q3+1.5*IQR)
        outlier_inf=(Q1-1.5*IQR)
        Media=df[i].mean()
        Mediana=df[i].median()
        Mínimo=df[i].min()
        Máximo=df[i].max()
        std_dev=df[i].std()
        print(f'Datos estadísticos de la variable {i}:')
        print('Calculando...')
        print(f'Desviación estándard: {round(std_dev,3)}')
        print(f'Primer Cuartil -> Q1: {Q1}')
        print(f'Tercer Cuartil -> Q3: {Q3}')
        print(f'Rango intercuartilico (IQR): {IQR}')
        print(f'Se considera outlier valores superiores a {round(outlier_sup,3)} e inferior a {round(outlier_inf,3)}')
        print(f'Media: {round(Media,3)}')
        print(f'Mediana: {Mediana}')
        print(f'Mínimo: {Mínimo}')
        print(f'Máximo: {Máximo} \n')

def produccion_año(df,lista_años):
    '''
    Devuelve un DataFrame con la suma de producción de energía eléctrica por año y tecnología.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    lista_años : list
    
    Returns
    -------
    df : pandas.core.frame.DataFrame
    '''
    hidraulica_sum=[df[df['Hidraulica'].index.year==i]['Hidraulica'].sum() for i in lista_años]
    turbina_bombeo_sum=[df[df['Turbina_bombeo'].index.year==i]['Turbina_bombeo'].sum() for i in lista_años]
    nuclear_sum=[df[df['Nuclear'].index.year==i]['Nuclear'].sum() for i in lista_años]
    carbon_sum=[df[df['Carbon'].index.year==i]['Carbon'].sum() for i in lista_años]
    motores_diesel_sum=[df[df['Motores_diesel'].index.year==i]['Motores_diesel'].sum() for i in lista_años]
    turbina_gas_sum=[df[df['Turbina_de_gas'].index.year==i]['Turbina_de_gas'].sum() for i in lista_años]
    turbina_vapor_sum=[df[df['Turbina_de_vapor'].index.year==i]['Turbina_de_vapor'].sum() for i in lista_años]
    ciclo_combinado_sum=[df[df['Ciclo_combinado'].index.year==i]['Ciclo_combinado'].sum() for i in lista_años]
    hidroeolica_sum=[df[df['Hidroeolica'].index.year==i]['Hidroeolica'].sum() for i in lista_años]
    eolica_sum=[df[df['Eolica'].index.year==i]['Eolica'].sum() for i in lista_años]
    solar_fotovoltaica_sum=[df[df['Solar_fotovoltaica'].index.year==i]['Solar_fotovoltaica'].sum() for i in lista_años]
    solar_termica_sum=[df[df['Solar_termica'].index.year==i]['Solar_termica'].sum() for i in lista_años]
    otras_renovables_sum=[df[df['Otras_renovables'].index.year==i]['Otras_renovables'].sum() for i in lista_años]
    cogeneracion_sum=[df[df['Cogeneracion'].index.year==i]['Cogeneracion'].sum() for i in lista_años]
    residuos_no_renovables_sum=[df[df['Residuos_no_renovables'].index.year==i]['Residuos_no_renovables'].sum() for i in lista_años]
    residuos_renovales_sum=[df[df['Residuos_renovables'].index.year==i]['Residuos_renovables'].sum() for i in lista_años]

    lista_tecnologias_sum=['Año','Hidraulica','Turbina_bombeo','Nuclear','Carbon','Motores_diesel','Turbina_de_gas',
                   'Turbina_de_vapor','Ciclo_combinado','Hidroeolica','Eolica','Solar_fotovoltaica','Solar_termica',
                   'Otras_renovables','Cogeneracion','Residuos_no_renovables','Residuos_renovables']    

    df_sum=pd.DataFrame(zip(lista_años, hidraulica_sum, turbina_bombeo_sum, nuclear_sum, 
                        carbon_sum, motores_diesel_sum, turbina_gas_sum, turbina_vapor_sum, 
                        ciclo_combinado_sum, hidroeolica_sum, eolica_sum, solar_fotovoltaica_sum, 
                        solar_termica_sum, otras_renovables_sum, cogeneracion_sum, residuos_no_renovables_sum, 
                        residuos_renovales_sum), columns=lista_tecnologias_sum)

    df_sum=df_sum.set_index('Año')
    return df_sum



def metricas(y_train, y_test, y_train_pred, y_test_pred, tipo_modelo):
    '''
    Devuelve un DataFrame con información de distintos metodos de evaluación.
    Cada lista o array debe tener la misma longitud.
    
    Parameters
    ----------
    y_train : lista o array de nuestra variable target del set train
    y_test : lista o array de nuestra variable target del set test
    y_train_pred : lista o array de nuestra predicción de la variable target del set train
    y_test_pred : lista o array de nuestra predicción de la variable target del set test
    tipo_modelo : str 

    Returns
    -------
    df : pandas.core.frame.DataFrame
    '''
    resultados = {'MAE': [metrics.mean_absolute_error(y_test, y_test_pred), metrics.mean_absolute_error(y_train, y_train_pred)],
                'MSE': [metrics.mean_squared_error(y_test, y_test_pred), metrics.mean_squared_error(y_train, y_train_pred)],
                'RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))],
                'R2':  [metrics.r2_score(y_test, y_test_pred), metrics.r2_score(y_train, y_train_pred)],
                 "set": ["TEST", "TRAIN"]}
    df = pd.DataFrame(resultados)
    df["modelo"] = tipo_modelo
    return df