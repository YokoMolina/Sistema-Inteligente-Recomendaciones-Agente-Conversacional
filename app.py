import os
import sys
from google import genai
from google.genai import types

# =====================================================================
# 1. HERRAMIENTAS REALES (Simulación de tus funciones importadas)
# =====================================================================
# NOTA: En tu proyecto real, asegúrate de que estas funciones accedan 
# a tus dataframes df_products o df_dataset_final correspondientes.

def get_user_history(user_id: int) -> str:
    """Retorna los productos más comprados por el usuario."""
    # Aquí va tu lógica de la Etapa 1
    if user_id == 99989:
        return """Historial de compras del usuario 99989:
        - ID: 31720 | Nombre: Organic Whole Milk (Veces comprado: 8)
        - ID: 3957  | Nombre: 100% Raw Coconut Water (Veces comprado: 8)
        - ID: 691   | Nombre: Organic Promise Strawberry Fields Cereal (Veces comprado: 5)"""
    return f"Historial cargado para el usuario {user_id}."

def get_similar_products_agent(product_id: int, k: int = 3) -> str:
    """Llama al modelo de embeddings de la Etapa 2."""
    return f"""Productos similares al ID {product_id}:
    - ID: 18076 | Nombre: Orange Peach Mango Juice | Similitud: 0.96
    - ID: 31690 | Nombre: Peach Mangosteen Juice Drink | Similitud: 0.91"""

def predict_reorder(user_id: int, product_id: int) -> str:
    """Llama al clasificador XGBRanker de la Etapa 1."""
    return f"El modelo XGBRanker calculó un score de relevancia de reorden de: 5.3898 para el usuario y el producto {product_id}."

# =====================================================================
# 2. CONFIGURACIÓN DEL AGENTE CONVERSACIONAL
# =====================================================================
system_instruction = """
Eres un agente conversacional inteligente y experto en recomendaciones para un supermercado e-commerce.
Tu objetivo es ayudar a los usuarios a armar carritos de compra eficientes utilizando las tres herramientas conectadas.
Responde siempre en español, con un tono profesional, claro y estructurado con viñetas.
"""

def iniciar_demo_cli():
    print("====================================================================")
    print("🛒 BIENVENIDO A LA DEMO INTERACTIVA DE RECOMENDACIONES (TIPTI-BOT) 🛒")
    print("====================================================================")
    
    # Verificar si la API Key está en el entorno
    if "GEMINI_API_KEY" not in os.environ:
        print("❌ Error: No se detectó la variable de entorno GEMINI_API_KEY.")
        print("Por favor, ejecuta en tu terminal: export GEMINI_API_KEY='tu_clave'")
        sys.exit(1)
        
    try:
        client = genai.Client()
    except Exception as e:
        print(f"❌ Error al inicializar el cliente de Gemini: {e}")
        sys.exit(1)
        
    print("🤖 Bot inicializado en vivo. Escribe 'salir' para terminar el chat.\n")
    
    while True:
        try:
            prompt_usuario = input("👤 Tú: ")
            if prompt_usuario.lower() in ['salir', 'exit', 'quit']:
                print("\n🤖 Bot: ¡Gracias por usar nuestro recomendador! Hasta luego. 🛒")
                break
                
            if not prompt_usuario.strip():
                continue
                
            print("🤖 Pensando y coordinando herramientas en vivo... ⏳")
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[get_user_history, get_similar_products_agent, predict_reorder],
                    temperature=0.2
                )
            )
            
            print("\n🤖 Bot:")
            print("-" * 50)
            print(response.text)
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"\n❌ Ocurrió un error durante la inferencia: {e}\n")

if __name__ == "__main__":
    iniciar_demo_cli()