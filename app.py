import streamlit as st
from rag import crear_chain

st.set_page_config(page_title='Ley Federal de Trabajo IA',layout='wide')

@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def get_chain():
    return crear_chain()


def main():
    chain = get_chain()

    st.title('Ley Federal de Trabajo')
    st.header('Chat de IA')
    st.subheader("El chat proporcionara informacion relacionada con la 'Ley Federal de Trabajo'")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
         st.write(msg["content"]) 

    if prompt := st.chat_input("Escribe tu pregunta..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})

        
        with st.chat_message("assistant"):
             with st.spinner("Pensando..."):
                respuesta = chain.invoke(prompt)
             st.write(respuesta)
        st.session_state.messages.append({"role":"assistant","content":respuesta})
        




if __name__=="__main__":
    main()         