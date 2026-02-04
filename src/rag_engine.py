import google.generativeai as genai
import os

class RagEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Usamos el modelo flash por rapidez y buena ventana de contexto
                self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            except Exception as e:
                print(f"[ERROR] RAG: {e}")

    def rewrite_query(self, query, history):
        """
        Mejora la reescritura para mantener atributos (color, marca) 
        pero detectar cambios de tema.
        """
        if not self.model or not history:
            return query
            
        # Tomamos los últimos 3 mensajes para no saturar, pero tener contexto reciente
        recent_history = history[-3:]
        
        prompt = f"""
        Actúa como un experto en búsqueda semántica. Tu objetivo es reformular la consulta del usuario para que sea AUTÓNOMA y COMPLETA.
        
        HISTORIAL DE CHAT RECIENTE:
        {recent_history}
        
        INPUT DEL USUARIO ACTUAL: "{query}"
        
        REGLAS DE RAZONAMIENTO:
        1. PERSISTENCIA DE OBJETO: Si el usuario menciona solo un atributo (ej: "baratos", "en rojo", "de marca Sony"), APLÍCALO al objeto principal que se estaba discutiendo en el historial.
           - Ejemplo: Historial="Busco laptops", Usuario="que sea HP" -> Salida="laptops HP".
        2. CAMBIO DE TEMA: Si el usuario menciona un NUEVO objeto explícito (ej: "ahora muestra mouses"), IGNORA los atributos anteriores que no tengan sentido.
           - Ejemplo: Historial="zapatos rojos", Usuario="busco cables" -> Salida="cables" (No "cables rojos").
        3. SI NO HAY CONTEXTO: Si el historial no ayuda, mantén la consulta original.
        
        SALIDA:
        Responde ÚNICAMENTE con la frase de búsqueda reformulada. Sin explicaciones.
        """
        try:
            response = self.model.generate_content(prompt)
            clean_query = response.text.strip().replace('"', '').replace("Search query:", "")
            return clean_query
        except:
            return query

    def analyze_intent(self, query, history):
        if not self.model: return "SEARCH"
        if "[Imagen]" in query: return "SEARCH"

        prompt = f"""
        Analiza la intención del usuario.
        Usuario: "{query}"
        
        Responde SOLO una palabra:
        - SEARCH: Si el usuario quiere buscar productos, filtrarlos, ver más opciones o cambiar características (ej: "busca X", "más baratos", "en azul").
        - DETAILS: Si el usuario hace una pregunta específica sobre la información que YA se mostró (ej: "¿cuál tiene más RAM?", "¿por qué recomiendas ese?", "explícame el precio").
        """
        try:
            response = self.model.generate_content(prompt)
            intent = response.text.strip().upper()
            return "DETAILS" if "DETAILS" in intent else "SEARCH"
        except:
            return "SEARCH"

    def generate_response(self, query, top_products, history=[], intent="SEARCH"):
        """
        Genera una respuesta altamente detallada y estructurada.
        """
        if not self.model:
            return "⚠️ Error: Falta configurar la API Key."

        # Construir contexto rico con toda la metadata disponible
        context_text = ""
        for i, prod in enumerate(top_products):
            meta = prod['metadata']
            # Intentamos sacar toda la info posible
            desc = meta.get('text_content', 'Sin descripción detallada')
            price = meta.get('price', 'Consultar')
            title = meta['title']
            
            context_text += f"""
            --- PRODUCTO {i+1} ---
            ID: {prod['id']}
            Nombre: {title}
            Precio: {price}
            Detalles Técnicos y Descripción: {desc}
            -----------------------
            """

        system_instruction = """
        Eres un Consultor Técnico Experto en productos de Amazon.
        Tu objetivo no es solo vender, sino ASESORAR con datos profundos y comparativas.
        
        INSTRUCCIONES DE RESPUESTA:
        1. **Análisis Profundo:** No repitas solo el título. Lee la descripción técnica ("Detalles Técnicos") y extrae especificaciones clave (RAM, material, dimensiones, conectividad, etc.).
        2. **Formato Estructurado:** Usa Markdown.
           - Usa **Negritas** para características clave.
           - Usa Listas (bullets) para especificaciones.
        3. **Justificación:** Explica POR QUÉ este producto coincide con lo que el usuario pidió (ej: "Mencionaste 'barato' y este es la opción más económica a $15").
        4. **Comparativa (Si hay varios):** Si muestras varios productos, añade una pequeña frase comparándolos (ej: "El Producto 1 es más potente, pero el Producto 2 es mejor calidad-precio").
        5. **Honestidad:** Si la descripción es escasa, di "La información técnica es limitada", no inventes datos.
        """

        full_prompt = f"""
        {system_instruction}
        
        CONTEXTO DE LA CONVERSACIÓN:
        {history[-5:]}
        
        RESULTADOS DE LA BÚSQUEDA (RAG):
        {context_text}
        
        PREGUNTA DEL USUARIO: "{query}"
        
        Genera tu respuesta experta ahora:
        """

        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generando respuesta IA: {e}"