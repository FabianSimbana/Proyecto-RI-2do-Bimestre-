import chromadb
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# --- CONFIGURACION ---
DB_PATH = 'db/chroma_db'
COLLECTION_NAME = 'amazon_products'
MODEL_ID = "openai/clip-vit-base-patch32"

class Retriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Iniciando Motor de Recuperación en {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar CLIP: {e}")
            raise e

        try:
            client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = client.get_collection(name=COLLECTION_NAME)
            count = self.collection.count()
            print(f"[INFO] Conectado a DB. Documentos: {count}")
        except Exception as e:
            print(f"[ERROR] Fallo ChromaDB: {e}")
            raise e

    def _safe_extract(self, features):
        """Extrae el tensor si CLIP devuelve un objeto"""
        if not isinstance(features, torch.Tensor):
            if hasattr(features, 'image_embeds'):
                return features.image_embeds
            elif hasattr(features, 'text_embeds'): # Para texto
                return features.text_embeds
            elif hasattr(features, 'pooler_output'):
                return features.pooler_output
            else:
                return features[0]
        return features

    def search_by_text(self, query_text, k=5):
        print(f"\n🔝 Buscando texto: '{query_text}'")
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = self._safe_extract(text_features) # SAFETY CHECK
        
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_emb = text_features.cpu().numpy().tolist()

        results = self.collection.query(
            query_embeddings=query_emb, n_results=k, include=["metadatas", "distances"]
        )
        return self._format_results(results)

    def search_by_image(self, image_path, k=5):
        print(f"\n🖼︝ Buscando imagen: '{image_path}'")
        if not os.path.exists(image_path):
            print("[ERROR] Imagen no existe.")
            return []

        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = self._safe_extract(image_features) # SAFETY CHECK
        
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        query_emb = image_features.cpu().numpy().tolist()

        results = self.collection.query(
            query_embeddings=query_emb, n_results=k, include=["metadatas", "distances"]
        )
        return self._format_results(results)

    def _format_results(self, results):
        formatted = []
        if not results['ids']: return []
        
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for i in range(len(ids)):
            formatted.append({
                "id": ids[i],
                "score": 1 - distances[i],
                "metadata": metadatas[i]
            })
        return formatted

if __name__ == "__main__":
    # Test rápido
    r = Retriever()
    res = r.search_by_text("kindle", k=1)
    if res:
        print(f"Top 1: {res[0]['metadata']['title']}")