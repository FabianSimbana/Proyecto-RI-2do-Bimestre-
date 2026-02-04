import pandas as pd
import requests
import os
import ast
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Pon aquí los nombres EXACTOS de tus archivos en la carpeta data/
FILES_TO_MERGE = [
    'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv',
    'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
]
OUTPUT_IMG_DIR = 'data/images'
FINAL_CSV_PATH = 'data/final_corpus.csv'
TARGET_ITEMS = 2000  # Tope para no llenar tu disco duro si hubiera demasiados

# --- FUNCIONES ---
def clean_url_list(url_string):
    """Intenta convertir el string de URLs en una lista limpia de Python"""
    try:
        # Caso 1: Es una lista real de python en string "['url1', 'url2']"
        if url_string.strip().startswith('['):
            return ast.literal_eval(url_string)
        # Caso 2: Es string separado por comas "url1,url2"
        else:
            return [x.strip() for x in url_string.split(',')]
    except:
        return []

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

# --- PROCESO PRINCIPAL ---
def main():
    print("🚀 Iniciando Fusión y Explosión de Imágenes...")
    
    # 1. Cargar y Fusionar Dataframes
    df_list = []
    for f in FILES_TO_MERGE:
        if os.path.exists(f):
            print(f"   Leyendo: {f}")
            df = pd.read_csv(f, low_memory=False)
            # Normalizamos nombres de columnas si varían
            if 'asins' not in df.columns and 'id' in df.columns:
                df['asins'] = df['id'] # Fallback
            df_list.append(df)
        else:
            print(f"⚠️ Alerta: No encuentro {f}")
    
    if not df_list:
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"   Filas totales combinadas (reviews): {len(full_df)}")

    # 2. Deduplicar Productos (Consolidar por ASIN)
    # Nos quedamos con el primer registro de cada ASIN que tenga imágenes
    products_df = full_df[full_df['imageURLs'].notna()]
    unique_products = products_df.drop_duplicates(subset=['asins'])
    
    print(f"   Productos ÚNICOS reales detectados: {len(unique_products)}")
    print("-" * 40)

    # 3. Preparar directorios
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    final_items = []
    
    # 4. Bucle de Explosión
    print(f"📸 Descargando imágenes (Max items: {TARGET_ITEMS})...")
    
    for _, row in tqdm(unique_products.iterrows(), total=len(unique_products)):
        if len(final_items) >= TARGET_ITEMS:
            break

        asin = str(row['asins'])
        name = str(row['name'])
        category = str(row.get('primaryCategories', 'Product'))
        
        # Obtener lista de URLs
        urls = clean_url_list(str(row['imageURLs']))
        
        # EXPLOSIÓN: Iterar cada URL de este producto
        for i, url in enumerate(urls):
            if len(final_items) >= TARGET_ITEMS:
                break
            
            # Filtrar basura
            if not isinstance(url, str) or not url.startswith('http'):
                continue

            # ID único para el sistema multimodal
            item_id = f"{asin}_{i}"
            filename = f"{item_id}.jpg"
            filepath = os.path.join(OUTPUT_IMG_DIR, filename)
            
            # Descargar si no existe
            is_valid = False
            if os.path.exists(filepath):
                is_valid = True
            else:
                is_valid = download_image(url, filepath)
            
            if is_valid:
                # Agregar al corpus final
                final_items.append({
                    'id': item_id,          # ID único (ej: B00X_0)
                    'parent_asin': asin,    # ID agrupador (ej: B00X)
                    'title': name,
                    # Contexto para el embedding de texto
                    'text_content': f"Category: {category}. Product: {name}. View {i+1}.",
                    'image_path': filepath
                })

    # 5. Guardar CSV Final
    result_df = pd.DataFrame(final_items)
    result_df.to_csv(FINAL_CSV_PATH, index=False)
    
    print("-" * 40)
    print(f"✅ ¡PROCESO TERMINADO!")
    print(f"📊 Total de ítems en tu corpus: {len(result_df)}")
    if len(result_df) < 500:
        print("⚠️ ADVERTENCIA: Aún son pocos ítems. Considera buscar un tercer CSV en Kaggle.")
    else:
        print("🎉 EXCELENTE: Tienes suficientes ítems para el proyecto.")
    print(f"📂 Archivo maestro guardado en: {FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()