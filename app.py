import os
import sys
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRanker

print("Inicializando motores de IA y cargando modelos reales en memoria...")

#  CARGA DE DATOS 
ruta_carpeta = r"D:\Tipti\Data"
ruta_products = os.path.join(ruta_carpeta, "products.csv")
ruta_modelo = os.path.join(ruta_carpeta, "xgbranker_model.json")
ruta_features = os.path.join(ruta_carpeta, "features_ranking.parquet")

# Validar que existan todos los archivos necesarios
for archivo in [ruta_products, ruta_modelo, ruta_features]:
    if not os.path.exists(archivo):
        print(f"❌ Error crítico: No se encontró el archivo en la ruta: {archivo}")
        sys.exit(1)

df_products = pd.read_csv(ruta_products)

# Características Precalculadas de Entrenamiento (Etapa 1)
print("Cargando matriz de características del consumidor...")
df_features = pd.read_parquet(ruta_features)

# Cargar e Inicializar el Modelo XGBRanker real (Etapa 1)
print("Cargando modelo tabular predictivo XGBRanker...")
ranker_model = XGBRanker()
ranker_model.load_model(ruta_modelo)

# El orden exacto de las columnas que el modelo espera para hacer el .predict()
FEATURES_MODELO = [
    'user_product_frequency', 
    'user_product_reordered', 
    'product_global_popularity', 
    'department_id'
]

# Inicializar Sentence-Transformers y pre-calcular embeddings (Etapa 2)
print("Generando matriz de embeddings semánticos para el catálogo...")
print("Espere por favor...")
model_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
product_names = df_products['product_name'].tolist()
product_embeddings = model_embeddings.encode(product_names, show_progress_bar=False, convert_to_numpy=True)

print("Todos los motores inicializados y listos para ejecuciones en vivo.\n")

# FUNCIONES CONECTADAS (TOOLS)

def get_user_history(user_id: int) -> str:
    """Retorna los productos más comprados históricamente por un usuario específico usando datos reales."""
    # Filtrar las características correspondientes a este usuario
    df_user = df_features[df_features['user_id'] == user_id]
    
    if df_user.empty:
        return f"El usuario ID {user_id} es nuevo en la plataforma o no registra historial de compras previo en el dataset."
    
    # Cruzar con la tabla de productos para obtener nombres amigables
    df_user_names = df_user.merge(df_products, on='product_id', how='inner')
    
    # Tomar los 5 productos con mayor frecuencia histórica de compra
    top_compras = df_user_names.sort_values(by='user_product_frequency', ascending=False).head(5)
    
    respuesta = f"Historial de compras real extraído para el usuario {user_id}:\n"
    for _, fila in top_compras.iterrows():
        respuesta += f"- ID: {int(fila['product_id'])} | Nombre: {fila['product_name']} (Frecuencia de compra: {int(fila['user_product_frequency'])})\n"
        
    return respuesta


def get_similar_products_agent(product_id: int) -> str:
    """
    Busca de manera real los 3 productos semánticamente más similares en el catálogo
    utilizando la matriz de embeddings y la similitud coseno.
    """
    idx_query = df_products[df_products['product_id'] == product_id].index
    
    if len(idx_query) == 0:
        return f"Error: El product_id {product_id} no existe en el catálogo actual."
    
    idx_query = idx_query[0]
    nombre_query = df_products.loc[idx_query, 'product_name']
    
    embedding_query = product_embeddings[idx_query].reshape(1, -1)
    scores = cosine_similarity(embedding_query, product_embeddings)[0]
    
    # Obtener los 3 mejores excluyendo el producto consultado
    top_indices = np.argsort(scores)[::-1][1:4] 
    
    respuesta = f"Productos reales y disponibles similares a '{nombre_query}' (ID: {product_id}):\n"
    for idx in top_indices:
        p_id = int(df_products.loc[idx, 'product_id'])
        p_name = df_products.loc[idx, 'product_name']
        p_score = float(scores[idx])
        respuesta += f"- ID: {p_id} | Nombre: {p_name} | Similitud Semántica: {p_score:.2f}\n"
        
    return respuesta


def predict_reorder(user_id: int, product_id: int) -> str:
    """Calcula el score real de relevancia para un par usuario-producto usando el XGBRanker entrenado."""
    # Buscar el par exacto usuario-producto en nuestra matriz de entrenamiento
    registro = df_features[(df_features['user_id'] == user_id) & (df_features['product_id'] == product_id)]
    
    if registro.empty:
        return f"No hay datos de interacciones previas suficientes para calcular el ordenamiento del usuario {user_id} con el producto {product_id}. Score de ranking asignado por defecto: 0.0"
    
    # Extraer estrictamente los valores numéricos de los features que el modelo necesita
    X_pred = registro[FEATURES_MODELO]
    
    # Ejecutar la predicción  con el modelo XGBRanker
    score_real = float(ranker_model.predict(X_pred)[0])
    
    # Recuperar el nombre del producto para la respuesta del chatbot
    nombre_prod = df_products[df_products['product_id'] == product_id]['product_name'].values
    nombre_prod = nombre_prod[0] if len(nombre_prod) > 0 else "Producto Identificado"
    
    return f"El modelo XGBRanker analizó al usuario {user_id} con el producto '{nombre_prod}' (ID: {product_id}) y calculó un score real de relevancia de reorden de: {score_real:.4f}."



#  AGENTE CONVERSACIONAL
system_instruction = """
Eres un agente conversacional inteligente y experto en recomendaciones para el supermercado Tipti.
Tu objetivo es ayudar a los usuarios a armar carritos de compra eficientes utilizando las herramientas conectadas.

Reglas de uso de herramientas:
- Si el usuario te indica quién es (su ID) o pide ver qué compra habitualmente, usa de forma obligatoria 'get_user_history'.
- Si te pide un producto similar, alternativo o sustituto, debes usar 'get_similar_products_agent' con el ID.
- Si necesitas validar la relevancia o prioridad de un artículo para ese cliente específico, llama a 'predict_reorder'.

Responde siempre en español, con un tono amable, profesional, claro y estructurado con viñetas.
"""

def iniciar_demo_cli():
    print("====================================================================")
    print("BIENVENIDO A LA DEMO INTERACTIVA DE RECOMENDACIONES (TIPTI-BOT)")
    print("====================================================================")
    
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: No se detectó la variable de entorno GEMINI_API_KEY.")
        sys.exit(1)
        
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error al inicializar el cliente de Gemini: {e}")
        sys.exit(1)
        
    print("Bot inicializado en vivo con datos y modelos REALES. Escribe 'salir' para terminar.\n")
    
    while True:
        try:
            prompt_usuario = input("Tú: ")
            if prompt_usuario.lower() in ['salir', 'exit', 'quit']:
                print("\nBot: ¡Gracias por usar nuestro recomendador! Hasta luego. 🛒")
                break
                
            if not prompt_usuario.strip():
                continue
                
            print("Analizando catálogo y consultando modelos en tiempo real... ")
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[get_user_history, get_similar_products_agent, predict_reorder],
                    temperature=0.3
                )
            )
            
            print("\nBot:")
            print("-" * 50)
            print(response.text)
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"\nOcurrió un error durante la sesión: {e}\n")

if __name__ == "__main__":
    iniciar_demo_cli()