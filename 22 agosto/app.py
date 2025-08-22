import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st


# Sidebar: input e pulsante
st.sidebar.header("Inserisci il tuo nome")
nome = st.sidebar.text_input("")
if st.sidebar.button("Manda"):
    if nome.strip():
        st.sidebar.success(f"Ciao {nome}!")
    else:
        st.sidebar.warning("Per favore inserisci un nome.")



# Counter
st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment():
    st.session_state.count += 1
def decrement():
    st.session_state.count -= 1

# Crea tre colonne: pulsante -, numero, pulsante +
col1, col2, col3 = st.columns([1, 1, 1])  # puoi cambiare i pesi per allargare il numero
with col1:
    st.button('Decrement', on_click=decrement)
with col2:
    st.write(f'Count = {st.session_state.count}')  # mostra il numero al centro
with col3:
    st.button('Increment', on_click=increment)
