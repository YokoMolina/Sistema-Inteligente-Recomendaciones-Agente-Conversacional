# Sistema Inteligente de Recomendaciones & Agente Conversacional (Prueba Técnica)

Este repositorio contiene el desarrollo integral de un sistema de recomendación e-commerce de tres etapas, diseñado para optimizar la experiencia de compra de los usuarios mediante modelos predictivos tabulares, búsqueda semántica y agentes basados en Modelos de Lenguaje Grande (LLMs).

---

## Instrucciones de Instalación y Uso

### 1. Clonar el repositorio y Requisitos
* Desarrollado y testeado en Python 3.10

```bash
git clone [https://github.com/YokoMolina/Tipti.git](https://github.com/YokoMolina/Tipti.git)
cd Tipti
```

### 2. Configurar el Entorno Virtual
Se recomienda utilizar un entorno virtual limpio para evitar conflictos de dependencias:

```powershell
# En Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate

# En Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
Instala los paquetes requeridos utilizando el archivo auto-generado:

```bash
pip install -r requirements.txt
```

### 4. Configurar las Variables de Entorno
Para interactuar con el agente conversacional, es mandatorio configurar tu API Key de Google AI Studio:

```powershell
# En Windows (PowerShell)
$env:GEMINI_API_KEY="TU_API_KEY_REAL"

# En Windows (CMD)
set GEMINI_API_KEY=TU_API_KEY_REAL

# En Mac/Linux
export GEMINI_API_KEY="TU_API_KEY_REAL"
```

### 5. Ejecutar la Demo Funcional (CLI)
Levanta la interfaz interactiva en tiempo real ejecutando:

```bash
python app.py
```









