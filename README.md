# Sistema Inteligente de Recomendaciones & Agente Conversacional (Prueba Técnica)

Este repositorio contiene el desarrollo integral de un sistema de recomendación e-commerce de tres etapas, diseñado para optimizar la experiencia de compra de los usuarios mediante modelos predictivos tabulares, búsqueda semántica y agentes basados en Modelos de Lenguaje Grande (LLMs).

---

## Instrucciones de Instalación y Uso

### 1. Clonar el repositorio 
* Desarrollado y testeado en Python 3.10

### 2. Instalar Dependencias
Instalar los paquetes requeridos utilizando el archivo: requirements.txt

```bash
pip install -r requirements.txt
```

### 3. Configurar las Variables de Entorno
Para interactuar con el agente conversacional, se debe configurar tu API Key de Google AI Studio:

```powershell
# En Windows (PowerShell)
$env:GEMINI_API_KEY="TU_API_KEY_REAL"

# En Windows (CMD)
set GEMINI_API_KEY=TU_API_KEY_REAL

# En Mac/Linux
export GEMINI_API_KEY="TU_API_KEY_REAL"
```

### 4. Ejecutar la Demo Funcional (CLI)
Levanta la interfaz interactiva en tiempo real ejecutando:

```bash
python app.py
```

## Criterios de Diseño:

### Etapa 1: Modelo Tabular de Ranking (`XGBRanker`)
* Se descartó el uso de una clasificación binaria tradicional (`XGBClassifier`) en favor de un enfoque de aprendizaje para ordenamiento (*Learning to Rank*) utilizando **`XGBRanker`** con la optimización de la métrica **$NDCG@10$** (Normalized Discounted Cumulative Gain).
* **Justificación:** En e-commerce, el orden de los factores sí altera el producto. No basta con predecir la probabilidad matemática de si un usuario comprará algo, sino en qué posición exacta de la pantalla se le debe priorizar. 
* **Muestreo por Limitaciones de Hardware (Sampling):** Debido al volumen masivo del catálogo histórico y con el fin de mitigar restricciones severas de memoria RAM en el entorno de computación local, se implementó una estrategia de muestreo estratificado. Esto permitió preservar la distribución real de la variable objetivo (*reordered*) reduciendo drásticamente la huella de memoria sin degradar la capacidad de generalización del algoritmo.
* **Procesamiento y Carga de Datos por Partes (*Chunking*):** Para la ingesta y el *Feature Engineering* (cómputo de tasas de reorden por usuario y producto), los datasets crudos se procesaron secuencialmente en bloques o *chunks* de memoria controlados empleando `pandas`. Esta decisión de arquitectura evitó desbordamientos del sistema (*Out-of-Memory*) y garantizó un pipeline de preparación de datos escalable y reproducible.


### Etapa 3: Agente Conversacional & Optimización de Recursos
* **Decisión del Modelo:** El agente conversacional utiliza **`gemini-2.5-flash`**. Se eligió este modelo específico porque ofrece el mejor balance del mercado entre velocidad de respuesta, costo cero en su capa gratuita y una ventana de contexto lo suficientemente amplia para procesar el catálogo de productos recomendado sin cortes.
* **Decisión de la Interfaz (CLI):** Para la demo funcional se optó por una interfaz de línea de comandos (**CLI** en la terminal). Esta decisión permitió concentrar el 100% del tiempo de desarrollo en la robustez de la lógica de negocio, el procesamiento de datos y la inteligencia del bot, evitando la complejidad innecesaria de diseñar interfaces gráficas web.
* **Manejo de Casos Especiales (Edge Cases):** El sistema identifica automáticamente si un usuario es nuevo en la plataforma (sin historial de compras). Para estos casos, se programó una respuesta controlada que evita que el bot invente datos, asegurando una atención al cliente fluida, cordial y realista.




