import chromadb
import os
import shutil

DB_PATH = 'db/chroma_db_test'

def test():
    print("🧪 Probando ChromaDB...")
    
    # Limpieza previa
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.create_collection(name="test_collection")
        
        print("   Intentando insertar 1 dato de prueba...")
        collection.add(
            ids=["test_id_1"],
            embeddings=[[0.1] * 512], # Vector dummy de 512 dimensiones
            metadatas=[{"prueba": "funciona"}]
        )
        
        count = collection.count()
        print(f"   Conteo en DB: {count}")
        
        if count == 1:
            print("✅ ¡EXITO! Tu base de datos funciona correctamente.")
        else:
            print("❌ ERROR: Se ejecutó la inserción pero el conteo es 0.")
            
    except Exception as e:
        print(f"❌ ERROR CRITICO: {e}")

if __name__ == "__main__":
    test()