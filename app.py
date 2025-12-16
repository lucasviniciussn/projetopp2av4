import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import datetime
import pandas as pd
import time

st.set_page_config(page_title="Verificação de Atenção", page_icon="🎓")

def get_color(classe):
    positivos = ['Engaged', 'Focused']
    neutros = ['Not Engaged', 'Looking Away']
    if classe in positivos: return "#d4edda", "#155724" 
    if classe in neutros: return "#fff3cd", "#856404"   
    return "#f8d7da", "#721c24"                         

# Banco de Dados
def initDB():
    conn = sqlite3.connect('dados_projeto.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            predicao TEXT,
            confianca REAL
        )
    ''')
    conn.commit()
    conn.close()

def saveDB(predicao, confianca):
    conn = sqlite3.connect('dados_projeto.db')
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    c.execute("INSERT INTO historico (data_hora, predicao, confianca) VALUES (?, ?, ?)",
              (agora, predicao, confianca))
    conn.commit()
    conn.close()

initDB()

# Modelo
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('student_attention_model.h5')
    except:
        return None

model = load_model()
class_names = ['Bored', 'Confused', 'Drowsy', 'Engaged', 
               'Focused', 'Frustrated', 'Looking Away', 'Not Engaged']

# Interface Streamlit

st.title("Monitor de Atenção")
st.write("Faça o upload da foto para analisar.")
file = st.file_uploader("", type=["jpg", "png", "jpeg"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Imagem do Aluno", use_column_width=True)

    if st.button("🔍 CLIQUE AQUI PARA ANALISAR", type="primary"):
        with st.spinner('Analisando...'):
            size = (128, 128)
            image_ops = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_ops)
            img_reshape = img_array[np.newaxis, ...]

            prediction = model.predict(img_reshape)
            index = np.argmax(prediction)
            classe_resultado = class_names[index]
            confianca = float(np.max(prediction))

            saveDB(classe_resultado, confianca)

            bg_color, text_color = get_color(classe_resultado)
                
            st.markdown(f"""
                <div style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 10px;">
                <h2 style="margin:0;">Estado: {classe_resultado}</h2>
                <p style="margin:0;">Confiança da IA: {confianca:.1%}</p>
                </div>
            """, unsafe_allow_html=True)

st.subheader("Histórico de Análises")

conn = sqlite3.connect('dados_projeto.db')
df = pd.read_sql("SELECT * FROM historico ORDER BY id DESC", conn)
conn.close()

if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.write("Nenhum registro ainda.")