from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        # Usamos un modelo Cross-Encoder optimizado para MS MARCO
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"[INFO] Cargando modelo de Re-ranking: {model_name}...")
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar CrossEncoder: {e}")
            raise e

    def rerank(self, query_text, candidates, top_k=5):
        """
        Recibe una lista de candidatos (output de retrieval), 
        calcula score preciso y reordena.
        """
        if not candidates:
            return []

        # Preparar pares para el modelo: [[Query, Texto1], [Query, Texto2], ...]
        pairs = []
        for cand in candidates:
            # Concatenamos Titulo + Contenido para que el modelo tenga contexto completo
            # A veces el titulo solo no basta
            doc_text = f"{cand['metadata']['title']}. {cand['metadata'].get('text_content', '')}"
            pairs.append([query_text, doc_text])

        # Predecir scores (logits)
        scores = self.model.predict(pairs)

        # Asignar nuevos scores a los candidatos
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(scores[i])
            # Guardamos el score original para comparar en el informe
            cand['original_score'] = cand['score'] 

        # Ordenar descendente por el NUEVO score
        sorted_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        return sorted_candidates[:top_k]

# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__":
    # Datos dummy para probar sin necesitar la DB
    print("[TEST] Iniciando prueba de reranker...")
    ranker = Reranker()
    
    query = "zapatos deportivos nike"
    
    # Simulamos resultados que vendrian de ChromaDB (algunos buenos, otros malos)
    fake_results = [
        {"id": "1", "score": 0.85, "metadata": {"title": "Nike Air Max Running", "text_content": "Zapatos para correr"}},
        {"id": "2", "score": 0.82, "metadata": {"title": "Calcetines Nike", "text_content": "Algodón 100%"}}, # Irrelevante pero con palabra clave
        {"id": "3", "score": 0.80, "metadata": {"title": "Adidas Ultraboost", "text_content": "Competencia directa"}},
    ]
    
    reranked = ranker.rerank(query, fake_results)
    
    print(f"\nConsulta: {query}")
    print("-" * 30)
    for r in reranked:
        print(f"[{r['rerank_score']:.4f}] {r['metadata']['title']}")