# Sistema RAG Multimodal

Este proyecto implementa un **Sistema de Recuperación de Información Avanzado** para productos de e-commerce. Utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** multimodal, permitiendo búsquedas mediante texto e imágenes, refinamiento por re-ranking y generación de respuestas justificadas mediante Inteligencia Artificial Generativa.

---

## Características Principales

1.  **Búsqueda Multimodal (CLIP):** Permite buscar productos describiéndolos en lenguaje natural o subiendo una fotografía (Búsqueda Visual).
2.  **Re-Ranking Neuronal:** Implementa un paso de refinamiento usando un Cross-Encoder (`ms-marco-MiniLM`) que reordena los resultados para maximizar la relevancia semántica, corrigiendo los errores de la búsqueda vectorial pura.
3.  **Generación Aumentada (RAG):** Integra **Google Gemini 1.5 Flash** para actuar como un consultor experto, explicando las características técnicas del producto y justificando la recomendación.
4.  **Memoria Conversacional Inteligente:** El sistema recuerda el contexto. Si buscas "laptops" y luego escribes "que sean HP", el sistema interpreta "laptops HP".
5.  **Evidencia Técnica:** Interfaz que muestra métricas en tiempo real (Score Original vs. Score Re-ranker) para auditoría del sistema.

---

## Stack Tecnológico

* **Lenguaje:** Python 3.10+
* **Interfaz:** Streamlit
* **Base de Datos Vectorial:** ChromaDB
* **Embeddings & Visión:** OpenAI CLIP (`vit-base-patch32`)
* **Re-Ranking:** Sentence-Transformers (Cross-Encoder)
* **LLM:** Google Gemini API (`gemini-1.5-flash`)
* **Procesamiento de Datos:** Pandas, NumPy, PyTorch

---

## Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

### 1. Clonar el repositorio (o descomprimir carpeta)
```bash
cd Proy2RI
```
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    ## Ejecución
```bash
streamlit run app.py
    
