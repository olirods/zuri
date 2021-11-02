# -*- coding: utf-8 -*-
"""
Para cada uno de nuestros municipios que tenemos en el dataset, obtendremos el texto de entrenamiento de la página de El Confidencial, que es del tipo:

*Con un 100% de los votos escrutados, el PSOE triunfaría en las elecciones municipales de Utebo con siete concejales, lo cual se traduciría en uno más que los registrados en la región durante los pasados comicios. La formación socialista se habría hecho con el 33,2% de los votos, porcentaje que le serviría para estar por delante del PP, que se quedaría con la segunda plaza al obtener tres ediles; mientras que el tercer lugar correspondería a Cs, con tres; y le seguirían IU, con dos; FIA con uno; y Vox, con uno. Peor suerte correrían Podemos-Equo, que con un 4,75% no se sentaría en el Salón de Plenos; y el PAR, que nuevamente se quedaría a las puertas del concejo municipal al obtener el 4,06%.*
"""

import os
import pandas as pd
import numpy as np

file = 'train_data_asistente.csv'
data = pd.read_csv(file, index_col=0)

from googlesearch import search, get_random_user_agent
from random import randint
from bs4 import BeautifulSoup
import urllib.request
import re
import unidecode
import requests
import json

def google_results(keyword):
  """ Esta función realiza la búsqueda en Google con las palabras de 'keyword'
  y devuelve el enlace del primer resultado.

  Args:
      keyword (string): palabras clave a buscar

  Returns:
      string: enlace al primer resultado
  """
  i=1
  while i<3:
    try:
      query = keyword
      query = urllib.parse.quote_plus(query) # Format into URL encoding
      google_url = "http://api.scraperapi.com?api_key=[[[API-KEY]]]&url=https://www.google.com/search?q=" + query + "&autoparse=true"
      response = requests.get(google_url, {"User-Agent": get_random_user_agent})
      datosBusqueda = json.loads(response.text)
      return datosBusqueda['organic_results'][0]['link']
    except IndexError:
      i += 1
      print("IndexError.")
    except:
      print("ERROR GOOGLE DESCONOCIDO")

""" Este bucle irá municipio por municipio buscando el enlace al artículo de El Confidencial, cogiendo del HTML del artículo el párrafo que nos interesa y guardándolo en el DataFrame."""
for indice in range(0,8121):
  print("i = " + str(indice))
  municipioString = data.at[indice, 'municipio']

  # to search
  query = "elecciones municipales 2019 " + municipioString + " site:elconfidencial.com/elecciones-municipales-y-autonomicas/resultados/"
  
  try:
    enlace = google_results(query)
    print(municipioString + ": " + enlace)
  
    nombre = enlace
    nombre = re.sub("https://www.elconfidencial.com/elecciones-municipales-y-autonomicas/resultados/2019-05-../", '', nombre)
    nombre = re.sub("-escrutinio-recuento-26m_......./", '', nombre)

    municipioStringNORM = unidecode.unidecode(municipioString).lower().replace(" ", "-").replace("'", "-")

    if (municipioStringNORM == nombre):
      try:
        enlace = "http://api.scraperapi.com?api_key=6[[[API-KEY]]]&url=" + enlace
        print(enlace)
        my_request = urllib.request.urlopen(enlace)
        my_HTML = my_request.read()
      
        try:
          soup = BeautifulSoup(my_HTML)
          target = soup.find(id="news-body-center").contents[2].get_text()
          print(target)
          data.at[indice, "target_text"] = target
        except:
          print("Error con el contenido")
      
      except:
        print("Había link pero HTTP dio error")
        data.at[indice, "target_text"] = enlace
        
    else:
      print("Se equivocó al buscar")

  except TypeError:
    print("No resultado con " + municipioString)
  
  except:
    print("ERROR DESCONOCIDO")
  
  data.to_csv('train_data_prov.csv')