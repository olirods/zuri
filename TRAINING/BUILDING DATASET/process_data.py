# -*- coding: utf-8 -*-
"""Realizaremos el preprocesado de los datos electorales para obtenerlo en forma de la entrada 'input_text' que lee el generador de texto.
"""

import openpyxl
import os
import pandas as pd

"""Obtenemos la hoja de Excel, descargada de la página del Ministerio del Interior, con los datos de todos los municipios de las elecciones municipales de mayo de 2019."""

file = 'data2019.csv'
data = pd.read_csv(file)
data.info

data.head(10)

"""A continuación, calculamos el número de concejales a elegir correspondiente a cada municipio según la población censada, y tal como indica la Ley Orgánica del Régimen Electoral General."""

concejales = 0

for i in data.index:
  municipio = data.at[i, 'municipio_nom']
  poblacion = data.at[i, 'poblacion']

  if (poblacion <= 100): 
    concejales = 3
  elif (poblacion <= 250):
    concejales = 5
  elif (poblacion <= 1000):
    concejales = 7
  elif (poblacion <= 2000):
    concejales = 9
  elif (poblacion <= 5000):
    concejales = 11
  elif (poblacion <= 10000):
    concejales = 13
  elif (poblacion <= 20000):
    concejales = 17
  elif (poblacion <= 50000):
    concejales = 21
  elif (poblacion <= 100000):
    concejales = 25
  else:
    concejales = 25 + poblacion//100000
    if (concejales % 2 == 0):
      concejales += 1

  # ANADIMOS LA COLUMNA EN EL DATASET
  data.loc[i, 'concejales'] = concejales

  print("Municipio: " + str(municipio) + " - Poblacion: " + str(poblacion))
  print("   Concejales: " + str(concejales))

data.concejales = data.concejales.astype('Int64')
data.head(10)

"""Crearemos un nuevo DataFrame para guardar aquí los datos que realmente utilizaremos en el generador de textos. De primeras guardamos sólo el nombre del municipio para tener una referencia."""

data_util = data[['municipio_nom']].copy()
data_util.head(10)

"""Esta es una clase creada por Pedro Ferrer y Silvia Fuentes (https://github.com/vehrka/dhondt), para poder calcular los concejales (escaños) que le corresponden a cada partido según sus votos, aplicando la ley d'Hondt."""

import sys
from argparse import ArgumentParser


class dhondt():
    """Class to calculate d'Hondt statistics
    :Authors: Pedro Ferrer, Silvia Fuentes
    :Date: 2015-07-20
    :version: 1.1
    The minimum data to be providen is:
    + The number of seats [nseats]
    + The minimum percentage to get into the calculation [minper]
    + A dictionary with the votes of the candidatures [dcandi]
         dcandi = {'000001': 51000, '000002': 46000, '000007': 34000, '000006': 29000, 'others': 31000}
    CAVEAT LECTOR
    + It doesn't resolve seat ties
    + Always gets rid of a party called 'others'
    """
    def __init__(self, nseats, minper, dcandi, census=0, blankv=0, sploitv=0):
        self.nseats = nseats
        self.minper = minper
        self.census = census
        self.blankv = blankv
        self.sploitv = sploitv
        self.dcandi = dcandi.copy()
        self.calc()

    def __repr__(self):
        candidatures = sorted(self.dcandi.items(), key=lambda p: p[1], reverse=True)
        return '<dhondt nseats:{0} minper:{1} candi:{2}>'.format(self.nseats, self.minper, candidatures)

    @property
    def nseats(self):
        return self.__nseats

    @nseats.setter
    def nseats(self, nseats):
        if type(nseats) is int and nseats > 0:
            self.__nseats = nseats
        else:
            raise AttributeError('The number or seats value must be an integer greater than 0')

    @property
    def minper(self):
        return self.__minper

    @minper.setter
    def minper(self, minper):
        if type(minper) is float and minper > 0:
            self.__minper = minper
        else:
            raise AttributeError('The minimum percentage value must be a float greater than 0')

    @property
    def census(self):
        return self.__census

    @census.setter
    def census(self, census):
        if type(census) is int:
            self.__census = census
        else:
            raise AttributeError('The census value must be an integer')

    @property
    def blankv(self):
        return self.__blankv

    @blankv.setter
    def blankv(self, blankv):
        if type(blankv) is int:
            self.__blankv = blankv
        else:
            raise AttributeError('The blank votes value must be an integer')

    @property
    def sploitv(self):
        return self.__sploitv

    @sploitv.setter
    def sploitv(self, sploitv):
        if type(sploitv) is int:
            self.__sploitv = sploitv
        else:
            raise AttributeError('The sploit votes value must be an integer')

    @property
    def dcandi(self):
        return self.__dcandi

    @dcandi.setter
    def dcandi(self, dcandi):
        if type(dcandi) is dict:
            self.__dcandi = dcandi.copy()
            try:
                sum(dcandi.values())
            except TypeError:
                raise AttributeError('The candidatures votes values must be integers')
        else:
            raise AttributeError('The candidatures data must be a dictionary')

    def __mindata(self):
        if self.nseats and self.minper and self.dcandi:
            return True
        return False

    def calc(self):
        """Performs the calculation"""
        if not self.__mindata():
            sys.exit('Minimum data not set')
        vtot = sum(self.dcandi.values())
        # # TODO: Finish script with the RESULTS and PARTICIPATION sections
        # ncan = len(self.dcandi)
        # if self.census < (vtot + self.blankv + self.sploitv):
        #     bvcensus = False
        #     self.census = 0
        #     nabs = 0
        # else:
        #     bvcensus = True
        #     nabs = self.census - vtot - self.blankv - self.sploitv
        # Sort the candidatures in descending number of votes
        candidature = sorted(self.dcandi.items(), key=lambda p: p[1], reverse=True)
        minvot = (((vtot + self.blankv) * self.minper) / 100) - 1
        # Filter the candidatures that have not reached the minimum
        candismin = list(filter(lambda p: p[1] > minvot, candidature))
        candivali = list(filter(lambda p: p[0] != 'other', candismin))
        #candirest = list(filter(lambda p: p[1] < minvot + 1, candidatures))

        # Prepare the lists for the calculations
        candinames = [p[0] for p in candivali]
        candimaxis = [p[1] for p in candivali]
        canditrab = [(p[1], 0) for p in candivali]

        # Prepare the dictionaries for the results
        self.repre = dict(zip(candinames, [0 for name in candinames]))
        self.asigna = dict(zip(candinames, [[maxi] for maxi in candimaxis]))

        # Perform the seat calculation
        for i in range(self.nseats):
            # Find the party with the maximum nunber of votes in this round
            dic01 = dict(zip(candinames, canditrab))
            odic01 = sorted(dic01.items(), key=lambda p: p[1][0], reverse=True)
            parmax = odic01[0][0]
            inparmax = candinames.index(parmax)
            maxivotos = candimaxis[inparmax]
            nseatsre = canditrab[inparmax][1]
            # This line does the magic
            canditrab[inparmax] = (maxivotos / (nseatsre + 2), nseatsre + 1)
            self.repre[parmax] = nseatsre + 1
            # Fill the asignation table dictionary
            for j, trab in enumerate(canditrab):
                self.asigna[candinames[j]].append(int(trab[0]))
            # We need to know which was the party assigned with the seat before the last seat
            if i == self.nseats - 2:
                penparmax = parmax
            else:
                penparmax = parmax

        # Calculate the votes needed for another seat
        self.falta = {}
        votult = self.asigna[parmax][-2]

        for name in candinames:
            votu = self.dcandi[name]
            crep = self.repre[name]
            if name == parmax:
                # The last asigned seat gets the number differently
                crepp = self.repre[penparmax]
                votp = self.dcandi[penparmax]
                vfalta = int(votp / crepp * (crep + 1) - votu)
            else:
                cvot = self.asigna[name][-1]
                vfalta = int((votult - cvot) * (crep + 1))
            pfalta = (vfalta / votu) * 100.0
            # Stores the number of votes and the percentage over the actual votes
            self.falta[name] = (vfalta, pfalta)

for i in data.index:
  row_aux = data.loc[data.index == i]
  row_aux2 = row_aux.loc[:, "PSOE":data.columns[-2]]
  filtro = (row_aux2 != 0).any()
  row_format = row_aux2.loc[:, filtro]
  row_format = row_format.sort_values(by = i, axis=1, ascending = False)
  row_dict = row_format.to_dict('records')[0]

  candidature = sorted(row_dict.items(), key=lambda p: p[1], reverse=True)
  seats = int(row_aux.concejales.values[0])

  result = dhondt(seats, 5.0, row_dict)
  j = 1
  for partido in row_format.columns:
    stringPartido = 'partido' + str(j)
    stringConcejales = 'concejales' + str(j)
    data_util.at[i, stringPartido] = partido

    if result.repre.get(partido, -1) != -1:
      data_util.at[i, stringConcejales] = int(result.repre.get(partido))
    else:
      data_util.at[i, stringConcejales] = 0
    j += 1
    if j == 11:
      break
      #Ponemos un maximo de 10 partidos por municipio para que no sea locura

data_util.head(10)

"""Así, hemos añadido a nuestro nuevo DataFrame los datos de los partidos y sus concejales por municipio. Después, calculamos a partir de los votos de cada partido y los votos válidos totales, el porcentaje que le ha correspondido a cada uno."""

for z in range(1, 10+1):
  stringConcejales = 'concejales' + str(z)
  stringPartido = 'partido' + str(z)
  stringPorcentaje = 'porcentaje' + str(z)
  data_util[stringConcejales] = data_util[stringConcejales].astype('Int64')

  for i in data.index:
    if data_util.isnull().at[i, stringPartido] == False:
      data_util.at[i,stringPorcentaje] = data.at[i,data_util.at[i,stringPartido]]/data.at[i,'votos_validos']*100

pd.options.display.float_format = "{:,.2f}".format
data_util.head(10)

"""Ahora necesitamos calcular la diferencia de escaños que ha obtenido cada partido en cada municipio con respecto al año anterior, en este caso las elecciones municipales de mayo de 2015. Para ello, en primer lugar procesaremos los datos de estas elecciones como hemos hecho con las de 2019."""

file_old = 'data2015.csv'
data_old = pd.read_csv(file_old)
data_old

for i in data_old.index:
  municipio = data_old.at[i, 'municipio_nom']
  poblacion = data_old.at[i, 'poblacion']

  if (poblacion <= 100): 
    concejales = 3
  elif (poblacion <= 250):
    concejales = 5
  elif (poblacion <= 1000):
    concejales = 7
  elif (poblacion <= 2000):
    concejales = 9
  elif (poblacion <= 5000):
    concejales = 11
  elif (poblacion <= 10000):
    concejales = 13
  elif (poblacion <= 20000):
    concejales = 17
  elif (poblacion <= 50000):
    concejales = 21
  elif (poblacion <= 100000):
    concejales = 25
  else:
    concejales = 25 + poblacion//100000
    if (concejales % 2 == 0):
      concejales += 1

  # ANADIMOS LA COLUMNA EN EL DATASET
  data_old.loc[i, 'concejales'] = concejales

data_old.concejales = data_old.concejales.astype('Int64')

data_util_old = data_old[['municipio_nom']].copy()

data_old.head(10)

for i in data_old.index:
  row_aux = data_old.loc[data_old.index == i]
  row_aux2 = row_aux.loc[:, "PP":data_old.columns[-2]]
  filtro = (row_aux2 != 0).any()
  row_format = row_aux2.loc[:, filtro]
  row_format = row_format.sort_values(by = i, axis=1, ascending = False)

  try:
    row_dict = row_format.to_dict('records')[0]

    candidature = sorted(row_dict.items(), key=lambda p: p[1], reverse=True)
    seats = int(row_aux.concejales.values[0])

    result = dhondt(seats, 5.0, row_dict)
    j = 1
    for partido in row_format.columns:
      stringPartido = 'partido' + str(j)
      stringConcejales = 'concejales' + str(j)
      data_util_old.at[i, stringPartido] = partido

      if result.repre.get(partido, -1) != -1:
        data_util_old.at[i, stringConcejales] = int(result.repre.get(partido))
      else:
        data_util_old.at[i, stringConcejales] = 0
      j += 1

      if j == 11:
        break
    
  except:
    print("El municipio con indice " + str(i) + " dio error.")

for z in range(1, 10+1):
  stringConcejales = 'concejales' + str(z)
  stringPartido = 'partido' + str(z)
  
  data_util_old[stringConcejales] = data_util_old[stringConcejales].astype('Int64')
'''
  for i in data_old.index:
    if data_util_old.isnull().at[i, stringPartido] == False:
      data_util_old.at[i,stringPorcentaje] = data_old.at[i,data_util_old.at[i,stringPartido]]/data_old.at[i,'votos_validos']*100
    else:
      break'''
data_util_old.head(10)

data_util_old['municipio_nom'] = data_util_old['municipio_nom'].str.strip()
pd.options.display.float_format = "{:,.2f}".format
data_util_old.head(10)

"""Ahora que tenemos dos DataFrame de 2015 y 2019, los compararemos y sacaremos la diferencia de escaños que han obtenido los partidos."""

import re

for indice, municipio in data_util.iterrows():
  municipio_old = data_util_old.loc[data_util_old['municipio_nom'] == municipio['municipio_nom']]
 
  if municipio_old.empty == False:
    indice_old = municipio_old.index[0]
    
    for z in range(1, 10+1):
      stringPartido = 'partido' + str(z)
      stringConcejales = 'concejales' + str(z)
      stringDiferencia = 'diferencia' + str(z)

      if data_util.isnull().at[indice, stringPartido] == False:

        partidoN = ""
        for column in municipio_old.columns:
          if (data_util_old.isnull().at[indice_old, column] == False):
            if municipio_old[column].values[0] == data_util.at[indice, stringPartido]:
              partidoN_old = column
              cadena_N_aux = re.split('(\d+)', partidoN_old)
              N_old = cadena_N_aux[1]

              stringConcejales_old = 'concejales' + str(N_old)

              data_util.at[indice, stringDiferencia] = data_util.at[indice, stringConcejales] - data_util_old.at[indice_old, stringConcejales_old]
          else:
              break
      else:
        break


data_util.head(10)

for z in range(1, 10+1):
  stringDiferencia = 'diferencia' + str(z)
  stringPorcentaje = 'porcentaje' + str(z)
  data_util[stringDiferencia] = data_util[stringDiferencia].astype('Int64')
  data_util[stringPorcentaje] = data_util[stringPorcentaje].round(2)

data_util.to_csv('data_util.csv')
data_util

"""Ahora procedemos a transformar este dataset al formato que leerá el generador de textos, que deberá ser de la forma: "PSOE | 46.22 | 11 | 3 && PP | 16.92 | 4 | 0 && SI SE PUEDE | 10.51 | 2 | -1 && CCa-PNC | 10.03 | 2 | -3 && Cs | 6.7 | 1 | 1 && VECINOS POR CANDELARIA | 5.6 | 1 | 1 && IZQUIERDA UNIDA CANARIA | 1.74 | 0 | 0 && NCA-AMF | 0.99 | 0 | 0""""

import random, math
input_data_asistente = pd.DataFrame(columns=('municipio', 'escrutinio', 'input_text'))

for indice, municipio in data_util.iterrows():
  input = ""
  escrutinio = round(random.uniform(0.99, 100), 2)

  for z in range(1, 10+1):
    stringPartido = 'partido' + str(z)
    stringPorcentaje = 'porcentaje' + str(z)
    stringConcejales = 'concejales' + str(z)
    stringDiferencia = 'diferencia' + str(z)

    if data_util.isnull().at[indice, stringPartido] == False:
      nombreInput = data_util.at[indice, stringPartido]
      porcentajeInput = data_util.at[indice, stringPorcentaje]
      concejalesInput = data_util.at[indice, stringConcejales]
      diferenciaInput = ""

      if (data_util.isnull().at[indice, stringDiferencia] == False):  
        diferenciaDato = data_util.at[indice, stringDiferencia]                     
        if diferenciaDato > 0:
          diferenciaInput = str(diferenciaDato) + " más"
        elif diferenciaDato < 0:
          diferenciaInput = str(-diferenciaDato) + " menos"
        elif diferenciaDato == 0: 
          diferenciaInput = str(diferenciaDato)
      else:
        diferenciaInput = "<null>"
      partidoInput = "{nombre} | {porcentaje} | {concejales} | {diferencia} && ".format(nombre=nombreInput, porcentaje=porcentajeInput, concejales=concejalesInput, diferencia=diferenciaInput)
      input = input + partidoInput
      
  input = input[0:-4]
  input_data_asistente.loc[indice] = [data_util.at[indice, 'municipio_nom'], escrutinio, input]

input_data_asistente

input_data_asistente.to_csv('input_data_asistente.csv')

"""Tras ello, descargamos los textos de El Confidencial, para obtener la columna "target_text" con todos los textos. """

file = 'train_data_final.csv'
data_final = pd.read_csv(file, index_col=[0])
data_final = data_final['target_text']

data_final.head(10)

file = 'input_data_asistente.csv'
input_data_asistente = pd.read_csv(file, index_col=[0])

input_data_asistente

zuri_train_data = pd.DataFrame(columns=('municipio', 'escrutinio', 'input_text', 'target_text'))

for indice in data_final.index:
  target_text = data_final.loc[indice]
  target_text = target_text.replace("100", str(input_data_asistente.at[indice, "escrutinio"]))
  zuri_train_data.loc[indice] = [input_data_asistente.at[indice, 'municipio'], input_data_asistente.at[indice, 'escrutinio'], input_data_asistente.at[indice, 'input_text'], target_text]

zuri_train_data

zuri_train_data.to_csv('zuri_train_data.csv')
