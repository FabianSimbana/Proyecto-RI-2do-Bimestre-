import pandas as pd
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import sys

# --- CONFIGURACION ---
CSV_PATH = 'data/final_corpus.csv'
DB_PATH = 'db/chroma_db'            
COLLECTION_NAME = 'amazon_products'
BATCH_SIZE = 32                     

def main():
    print("INICIANDO INDEXADOR V5")
    
    # 1. Cargar CSV
    if not os.path.exists(CSV_PATH):
        print(f"Error: No existe {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f" Total ítems en CSV: {len(df)}")
    
    # --- DIAGNÓSTICO DE COLUMNAS ---
    # Vamos a asegurarnos de que el precio y descripción existan
    print(f"   Columnas detectadas: {list(df.columns)}")
    
    # Rellenar vacíos para que no rompa el código
    if 'price' not in df.columns:
        df['price'] = "N/A"
    if 'text_content' not in df.columns:
        # Si no hay text_content, intentamos crear uno con title + description si existen
        print("'text_content' no detectado. Intentando construirlo...")
        df['text_content'] = df['title'] # Fallback básico
    
    df['price'] = df['price'].fillna('Consultar')
    df['text_content'] = df['text_content'].fillna('')

    # 2. Cargar CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Cargando CLIP en: {device} ...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f" Error cargando modelo: {e}")
        return

    # 3. Iniciar DB
    print(f" Conectando a ChromaDB en: {DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print("   (Colección vieja borrada para actualizar datos)")
        except:
            pass
        collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        print(f" Error iniciando DB: {e}")
        return

    # 4. Procesamiento
    print(" Procesando imágenes e insertando...")
    
    batch_ids = []
    batch_embeddings = []
    batch_metadatas = []
    
    total_inserted = 0
    errors_files = 0
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            sys.stdout.write(f"\r   Procesando item {idx}/{len(df)}...")
            sys.stdout.flush()

        raw_path = str(row['image_path'])
        image_path = os.path.normpath(raw_path)
        
        if not os.path.exists(image_path):
            errors_files += 1
            continue
            
        try:
            # -- Embedding Visual --
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            
            # Safety check para tensores
            if not isinstance(features, torch.Tensor):
                features = features.image_embeds if hasattr(features, 'image_embeds') else features[0]

            features = features / features.norm(p=2, dim=-1, keepdim=True)
            emb = features.cpu().numpy().tolist()[0]
            
            description_text = str(row['text_content'])
            if len(description_text) > 800: # Recortar si es gigante para no saturar DB
                description_text = description_text[:800] + "..."

            meta = {
                "id": str(row['id']),
                "title": str(row['title']), 
                "parent_asin": str(row['parent_asin']),
                "image_path": raw_path,
                "price": str(row['price']),         # <--- AQUÍ ESTABA EL FALLO
                "text_content": description_text    # <--- AQUÍ ESTABA EL FALLO
            }

            batch_ids.append(str(row['id']))
            batch_embeddings.append(emb)
            batch_metadatas.append(meta)

            if len(batch_ids) >= BATCH_SIZE:
                collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)
                total_inserted += len(batch_ids)
                batch_ids = []
                batch_embeddings = []
                batch_metadatas = []

        except Exception as e_img:
            # print(f"Error silencioso en img: {e_img}")
            continue

    if batch_ids:
        collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)
        total_inserted += len(batch_ids)

    print(f"\n\n🏁 PROCESO TERMINADO")
    print(f"   Items actualizados en DB: {total_inserted}")
    print(" AHORA LA DB TIENE PRECIOS Y DESCRIPCIONES.")

if __name__ == "__main__":
    main()