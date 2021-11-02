from flask import Flask
from flask_assistant import Assistant, ask, tell, context_manager, event
import pandas as pd
import torch.nn as nn
import asyncio
import time
import requests
import threading

headers = {"Authorization": "Bearer api_OQGRQzPxYMHvPpFTuWbLUDZxNwQEaXkSFx"}
API_URL_FULL = "https://api-inference.huggingface.co/models/olirods/zuri"
API_URL_PRE = "https://api-inference.huggingface.co/models/LeoCordoba/mt5-small-mlsum"
API_URL_QUE = "https://api-inference.huggingface.co/models/mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"

app = Flask(__name__)
app.config['INTEGRATIONS'] = ['ACTIONS_ON_GOOGLE']
assist = Assistant(app, route='/', project_id="zuri-xnu9")

""" Peticiones HTTP a cada uno de los modelos """
# MODELO ZURI
def query_FULL(payload):
	response = requests.post(API_URL_FULL, headers=headers, json=payload)
	return response.json()

# MODELO SUMMARIZATION
def query_PRE(payload):
	response = requests.post(API_URL_PRE, headers=headers, json=payload)
	return response.json()

# MODELO SQUAD
def query_QUE(payload):
	response = requests.post(API_URL_QUE, headers=headers, json=payload)
	return response.json()

data = pd.read_csv('zuri_input_data.csv', index_col=[0])

async def generate(input, indice, action, question):
  """ Esta función se encarga de hacer las llamadas a los modelos de
  generación de texto en segundo plano y de guardar las respuestas en el 
  DataFrame para que puedan ser obtenidas más tarde por la aplicación.

  Args:
      input (string): datos de entrada de resultados electorales para el generador
      indice (integer): indice correspondiente del municipio dentro del DataFrame
      action (integer): indica que tipo de intent se ha solicitado primero:
                              0 - preview
                              1 - random
      question (string): [description]
  """
  query_full = query_FULL({"inputs": str(input), "parameters" : {"max_length": 1000}})

  gen_text = query_full[0]['generated_text']
  data.at[indice, 'full'] = gen_text

  if action == 0: 
    #PREVIEW
    query_pre = query_PRE({
    "inputs": gen_text, "parameters" : {"max_length": 200}})
    print(query_pre)
    summary = query_pre[0]['summary_text']
    data.at[indice, 'preview'] = summary
  elif action == 1:
    #RANDO
    query_que = query_QUE({
    "inputs": {
		"question": question,
		"context": gen_text,
	},
})
    answer = query_que['answer']
    data.at[indice, 'question'] = answer

  return 0

''' Gestión de la llamada a la función en segundo plano '''
def loop_in_thread(loop, input, indice, action, question):
  asyncio.set_event_loop(loop)
  loop.run_until_complete(generate(input,indice, action, question))

loop = asyncio.get_event_loop()

@assist.action('preview')
def google_preview(geocity):
    """ Esta función se ejecutará cuando el usuario le pida la intent 'preview' a Google
    Assistant. Realiza la petición a los modelos en segundo plano y mientras hace tiempo
    esperando 4 segundos y llamado a la intent auxiliar.

    Args:
      geocity (string): Parámetro de la intent. Nombre del municipio.

    Returns:
        Llamada al intent preview-results
    """
    action = 0
    question = ""

    municipio = data.loc[data['municipio'] == str(geocity)]
    indice = municipio.index[0]
    input = str(data.at[indice, 'input_text'])

    t = threading.Thread(target=loop_in_thread, args=(loop, input, indice, action, question))
    t.start()

    time.sleep(4)
    return event("preview-results")

@assist.action('preview-results')
def google_preview_results(geocity):
    """ Esta función se ejecutará tras la primera intent 'preview' y se
    encargará de devolver los resultados finales cuando los tenga, sino intentará
    seguir haciendo tiempo llamandose de nuevo a sí misma.

    Args:
      geocity (string): Parámetro de la intent. Nombre del municipio.

    Returns:
        o llamada al intent random-results
        o string con la respuesta
    """
    time.sleep(4)
    municipio = data.loc[data['municipio'] == str(geocity)]
    indice = municipio.index[0]
    preview = str(data.at[indice, 'preview'])
    print(preview)
    context_manager.add("preview-followup")

    if preview == "<null>":
        return event("preview-results")
    else:
        speech_text = str(preview) + ". ¿Quieres saber más?"
        resp = ask(speech_text).suggest("Sí", "No")
        return resp

@assist.action('random')
def google_random(geocity, question):
    """ Esta función se ejecutará cuando el usuario le pida la intent 'random' a Google
    Assistant. Realiza la petición a los modelos en segundo plano y mientras hace tiempo
    esperando 4 segundos y llamado a la intent auxiliar.

    Args:
      geocity (string): Parámetro de la intent. Nombre del municipio.
      question (string): Parámetro de la intent. Pregunta que realiza el usuario.

    Returns:
        Llamada al intent random-results
    """
    action = 1

    municipio = data.loc[data['municipio'] == str(geocity)]
    print(municipio)
    indice = municipio.index[0]
    input = str(data.at[indice, 'input_text'])

    t = threading.Thread(target=loop_in_thread, args=(loop, input, indice, action, question))
    t.start()

    return event("random-results")

@assist.action('random-results')
def google_random_results(geocity):
    """ Esta función se ejecutará tras la primera intent 'random' y se
    encargará de devolver los resultados finales cuando los tenga, sino intentará
    seguir haciendo tiempo llamandose de nuevo a sí misma.

    Args:
      geocity (string): Parámetro de la intent. Nombre del municipio.

    Returns:
        o llamada al intent random-results
        o string con la respuesta
    """
    time.sleep(4)
    municipio = data.loc[data['municipio'] == str(geocity)]
    indice = municipio.index[0]
    preview = str(data.at[indice, 'question'])
    context_manager.add("random-followup")

    if preview == "<null>":
        return event("random-results")
    else:
        speech_text = str(preview) + ". ¿Quieres saber más?"
        resp = ask(speech_text).suggest("Sí", "No")
        return resp

@assist.action('full')
def google_full(geocity):
    """ Esta función se ejecutará cuando el usuario le pida la intent 'preview' a Google
    Assistant. Realiza la petición a los modelos en segundo plano y mientras hace tiempo
    esperando 4 segundos y llamado a la intent auxiliar.

    Args:
      geocity (string): Parámetro de la intent. Nombre del municipio.
      question (string): Parámetro de la intent. Pregunta que realiza el usuario.

    Returns:
        Llamada al intent preview-results
    """
    municipio = data.loc[data['municipio'] == str(geocity)]
    indice = municipio.index[0]
    speech_text = str(data.at[indice, 'full'])
    resp = tell(speech_text)
    resp.card(
        title="Elecciones municipales 2023 en " + str(geocity),
        text="",
        img_url=data.at[indice, 'img'])
    return resp
    
if __name__ == '__main__':
    app.run('localhost', 8000)