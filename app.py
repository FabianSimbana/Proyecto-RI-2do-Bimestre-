import streamlit as st
import pandas as pd # IMPORTANTE: Para la tabla de ranking
from src.retrieval import Retriever
from src.reranker import Reranker
from src.rag_engine import RagEngine
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Amazon AI Shopper", layout="centered")
st.markdown("""<style>.stDeployButton {display:none;} .block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)

# --- ESTADO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_products" not in st.session_state:
    st.session_state.last_products = []
if "debug_ranking" not in st.session_state: # Nuevo estado para guardar la tabla
    st.session_state.debug_ranking = None

# --- CARGA ---
@st.cache_resource
def load_models():
    return Retriever(), Reranker()

try:
    retriever, reranker = load_models()
except Exception as e:
    st.error(f"Error cargando sistema: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    top_k = st.slider("Productos a mostrar", 1, 10, 3)
    st.divider()
    
    # MOSTRAR TABLA DE RANKING EN SIDEBAR (Opcional, también saldrá en el chat)
    if st.session_state.debug_ranking is not None:
        st.subheader("📊 Análisis de Re-ranking (Último)")
        st.dataframe(st.session_state.debug_ranking, hide_index=True)

rag = RagEngine(api_key=api_key if api_key else None)

st.title("🛍️ Amazon AI Shopper")
st.caption("Búsqueda con Contexto y Re-ranking")

# --- HISTORIAL ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_path"): st.image(msg["image_path"], width=200)
        if msg["content"]: st.markdown(msg["content"])
        
        # Mostrar productos si existen en este mensaje
        if msg.get("products"):
            with st.container():
                st.write("---")
                cols = st.columns(3)
                for i, prod in enumerate(msg["products"]):
                    meta = prod['metadata']
                    with cols[i % 3]:
                        if os.path.exists(meta['image_path']):
                            st.image(meta['image_path'], use_container_width=True)
                        st.caption(f"**{meta['title'][:40]}...**\n💲{meta['price']}")
            
            # Mostrar botón expandible con los datos del Ranking de ESA búsqueda
            if msg.get("ranking_data") is not None:
                with st.expander("📊 Ver Análisis de Ranking (Técnico)"):
                    st.dataframe(msg["ranking_data"], hide_index=True)

# --- INPUT ---
with st.expander("📷 Buscar por imagen", expanded=False):
    uploaded_file = st.file_uploader("Sube una foto", type=["jpg", "png", "jpeg"], key="img_uploader")

if prompt := st.chat_input("Escribe tu búsqueda..."):
    
    # 1. Manejo Input Usuario
    image_search_path = None
    if uploaded_file:
        image_search_path = f"temp_{uploaded_file.name}"
        with open(image_search_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.chat_message("user"):
            st.image(image_search_path, width=200)
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": f"[Imagen] {prompt}", "image_path": image_search_path})
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Procesamiento IA
    with st.chat_message("assistant"):
        with st.spinner("Procesando contexto y ranking..."):
            
            chat_history_text = [m["content"] for m in st.session_state.messages]
            
            # A. REESCRITURA DE QUERY (Contexto)
            effective_query = prompt
            if not image_search_path:
                effective_query = rag.rewrite_query(prompt, chat_history_text[:-1]) # No incluimos el actual para no confundir
                if effective_query != prompt:
                    st.caption(f" *Búsqueda contextual interpretada: '{effective_query}'*")
            
            # B. ANÁLISIS DE INTENCIÓN
            intent = rag.analyze_intent(effective_query, chat_history_text)
            
            final_products = []
            ranking_df = None
            
            if intent == "SEARCH" or image_search_path:
                # 1. Retrieval
                fetch_k = top_k * 4 # Traemos más para ver el efecto del re-ranking
                candidates = []
                if image_search_path:
                    candidates = retriever.search_by_image(image_search_path, k=fetch_k)
                else:
                    candidates = retriever.search_by_text(effective_query, k=fetch_k)
                
                # 2. Re-ranking
                if candidates and (effective_query or image_search_path):
                    # Texto para rerank: si es imagen, usamos el prompt o "producto similar"
                    text_for_rerank = effective_query if effective_query else "producto similar visualmente"
                    final_products = reranker.rerank(text_for_rerank, candidates, top_k=top_k)
                    
                    # --- CREAR TABLA DE DATOS PARA INFORME ---
                    # Combinamos los datos antes del corte top_k para ver el efecto
                    # Pero para mostrar en pantalla, mostramos los finales vs sus scores originales
                    ranking_data = []
                    for p in final_products:
                        ranking_data.append({
                            "Producto": p['metadata']['title'][:30],
                            "Score CLIP (Original)": f"{p['score']:.4f}",
                            "Score Re-Ranker": f"{p['rerank_score']:.4f}"
                        })
                    ranking_df = pd.DataFrame(ranking_data)
                    st.session_state.debug_ranking = ranking_df # Guardar en sidebar
                else:
                    final_products = candidates[:top_k]
                
                st.session_state.last_products = final_products
            
            else:
                # Intención DETAILS
                final_products = st.session_state.last_products
            
            # C. Respuesta Generada
            response_text = rag.generate_response(
                query=effective_query,
                top_products=final_products,
                history=chat_history_text,
                intent=intent
            )
            
            st.markdown(response_text)
            
            # D. Mostrar Resultados y Tabla
            products_to_save = []
            if (intent == "SEARCH" or image_search_path) and final_products:
                st.write("---")
                cols = st.columns(3)
                for i, prod in enumerate(final_products):
                    meta = prod['metadata']
                    with cols[i % 3]:
                        if os.path.exists(meta['image_path']):
                            st.image(meta['image_path'], use_container_width=True)
                        st.caption(f"**{meta['title'][:40]}...**\n💲{meta['price']}")
                products_to_save = final_products
                
                # Mostrar Tabla de Ranking AQUÍ MISMO
                if ranking_df is not None:
                    with st.expander("📊 Análisis de Re-ranking (Evidencia Técnica)"):
                        st.write("Comparativa de puntajes. El 'Score Re-Ranker' es el definitivo.")
                        st.dataframe(ranking_df, use_container_width=True)

            # Guardar en historial
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "products": products_to_save,
                "ranking_data": ranking_df
            })