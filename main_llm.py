# ##### Categorizador de Noticias 🇦🇷 — Atraso Cambiario  
# Flask + Gemini 2.5 Flash‑Lite
# Este micro‑sitio web permite **subir un Excel con noticias**, enviarlas a Gemini para
# etiquetar si contienen (1) / no contienen (0) interpretaciones de "atraso cambiario",
# y descargar el mismo archivo con dos columnas nuevas:
# * `label_llm` → 0 ó 1  
# * `texto_completo` → razonamiento estructurado del LLM
# ---
# ## 1 · Requisitos rápidos
# | Paquete | Versión probada |
# |---------|-----------------|
# | Python ≥ 3.10 |   |
# | `flask` | 3.0.x |
# | `pandas` | 2.2.x |
# | `openpyxl` | 3.1.x |
# | `google‑generativeai` | 0.4.x |
# | `replit` | 3.x (sólo para clave‑valor simple) |
# Instalación 👇
# ```bash
# python -m venv venv
# source venv/bin/activate        # en Windows: venv\\Scripts\\activate
# pip install -r requirements.txt

import pandas as pd
import google.generativeai as genai
import time
import os
import re
from typing import List, Tuple
from replit import db
import json
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request, send_file, session
import io
import threading
from werkzeug.utils import secure_filename

# Configurar tu API key de Google
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"

# Configurar Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Variable global para tracking del progreso
progress_data = {
    'processing': False,
    'current': 0,
    'total': 0,
    'status': 'Esperando archivo...',
    'completed': False,
    'filename': None,
    'error': None
}

def categorizar_noticia_individual(noticia: str) -> Tuple[int, str]:
    """
    Categoriza una noticia individual con razonamiento completo
    """
    prompt = f"""Eres un analista especializado en macroeconomía argentina. Debes analizar si la siguiente noticia contiene una interpretación de que el dólar oficial argentino está atrasado.

NOTICIA A ANALIZAR:
"{noticia}"

INSTRUCCIONES:
Sigue estos pasos de análisis de forma ESTRICTA:

PASO 1: Identifica las frases clave
- Busca menciones EXPLÍCITAS al tipo de cambio oficial
- Identifica si hay palabras clave como: "atraso cambiario", "dólar atrasado", "tipo de cambio pisado", "dólar artificialmente bajo", "pérdida de competitividad cambiaria", "urgen devaluar", "presionan por devaluación"

PASO 2: Identifica quién lo dice
- ¿Es el periodista quien interpreta el atraso?
- ¿Es un agente citado (empresario, economista, etc.)?
- ¿O nadie lo dice explícitamente?

PASO 3: Evalúa el contexto
- ¿Se menciona el dólar oficial específicamente?
- ¿Se relaciona explícitamente con inflación, competitividad o precios?
- ¿Es una interpretación actual o una expectativa futura?

PASO 4: Aplica criterios estrictos
MARCA 1 SOLO SI:
- Hay una mención EXPLÍCITA de atraso del dólar oficial
- Alguien (periodista o fuente citada) INTERPRETA que el dólar está atrasado
- La interpretación es sobre la situación ACTUAL (no expectativas futuras)

NO MARQUES 1 SI:
- Solo se menciona dólar blue o brecha (sin decir que el oficial está atrasado)
- Solo se habla de inflación (sin vincularla al atraso cambiario)
- Solo se mencionan problemas económicos (sin culpar al tipo de cambio)
- Solo se habla de expectativas futuras de devaluación
- No hay mención explícita del concepto de atraso cambiario

FORMATO DE RESPUESTA:
Paso 1 - Frases clave encontradas: [lista las frases relevantes o indica "ninguna"]
Paso 2 - Quién lo dice: [identifica la fuente o indica "nadie"]
Paso 3 - Contexto: [explica brevemente el contexto]
Paso 4 - Evaluación final: [explica tu decisión]

Respuesta: [0 o 1]"""

    try:
        response = model.generate_content(prompt)
        texto_completo = response.text.strip()

        # Extraer el número de la respuesta
        match = re.search(r'Respuesta:\s*[(\[]?([01])[)\]]?', texto_completo, re.IGNORECASE)
        if match:
            label = int(match.group(1))
        else:
            # Si no encuentra el patrón, buscar cualquier 0 o 1 al final
            numeros = re.findall(r'[01]', texto_completo[-20:])
            label = int(numeros[-1]) if numeros else 0

        return label, texto_completo

    except Exception as e:
        print(f"Error al procesar noticia: {e}")
        return 0, f"Error: {str(e)}"

def procesar_excel_background(df, filename):
    """
    Procesa el Excel en background y actualiza el progreso
    """
    global progress_data
    
    try:
        progress_data['processing'] = True
        progress_data['total'] = len(df)
        progress_data['current'] = 0
        progress_data['status'] = 'Iniciando procesamiento...'
        progress_data['error'] = None
        
        # Verificar que existe la columna 'text'
        if 'text' not in df.columns:
            progress_data['error'] = "No se encontró la columna 'text' en el Excel"
            progress_data['processing'] = False
            return
        
        # Inicializar columnas
        df['label_llm'] = None
        df['texto_completo'] = None
        
        # Procesar cada noticia
        for i in range(len(df)):
            if not progress_data['processing']:  # Check if stopped
                break
                
            noticia = df.loc[i, 'text']
            progress_data['current'] = i + 1
            progress_data['status'] = f'Procesando noticia {i+1}/{len(df)}: {noticia[:50]}...'
            
            # Categorizar
            label, razonamiento = categorizar_noticia_individual(noticia)
            
            # Guardar resultados
            df.loc[i, 'label_llm'] = label
            df.loc[i, 'texto_completo'] = razonamiento
            
            # Pequeña pausa
            time.sleep(0.5)
        
        # Guardar archivo procesado
        if progress_data['processing']:  # Solo si no fue cancelado
            output_filename = f"procesado_{filename}"
            output_path = os.path.join('uploads', output_filename)
            df.to_excel(output_path, index=False)
            
            progress_data['status'] = 'Procesamiento completado!'
            progress_data['completed'] = True
            progress_data['filename'] = output_filename
        
        progress_data['processing'] = False
        
    except Exception as e:
        progress_data['error'] = str(e)
        progress_data['processing'] = False
        progress_data['status'] = f'Error: {str(e)}'

# Create Flask app
app = Flask(__name__)
app.secret_key = 'mi_clave_secreta_para_sesiones'

# Crear directorio de uploads si no existe
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def home():
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Categorizador de Noticias</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .container { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center; 
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #3498db;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #ecf0f1;
            }
            .upload-area.dragover {
                background-color: #d5dbdb;
                border-color: #2980b9;
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                width: 100%;
                max-width: 300px;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px 5px;
            }
            button:hover { background-color: #2980b9; }
            button:disabled { 
                background-color: #95a5a6; 
                cursor: not-allowed; 
            }
            .progress-container {
                margin: 20px 0;
                display: none;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                background-color: #27ae60;
                width: 0%;
                transition: width 0.3s ease;
            }
            .status {
                margin: 10px 0;
                padding: 10px;
                background-color: #e8f6f3;
                border-radius: 5px;
                font-weight: bold;
            }
            .error {
                background-color: #fadbd8;
                color: #c0392b;
            }
            .success {
                background-color: #d5f4e6;
                color: #27ae60;
            }
            .logs {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                font-family: monospace;
                height: 200px;
                overflow-y: auto;
                margin: 20px 0;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Categorizador de Noticias</h1>
            <p>Sube un archivo Excel con una columna llamada 'text' y obten las noticias categorizadas por atraso cambiario.</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area" id="uploadArea">
                    <p>📁 Arrastra tu archivo Excel aquí o haz clic para seleccionar</p>
                    <input type="file" name="file" id="fileInput" accept=".xlsx,.xls" required>
                </div>
                <div style="text-align: center;">
                    <button type="submit" id="uploadBtn">🚀 Procesar Excel</button>
                    <button type="button" id="cancelBtn" style="display: none;" onclick="cancelProcessing()">❌ Cancelar</button>
                </div>
            </form>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="status" id="statusDiv">Esperando archivo...</div>
                <div id="downloadSection" style="display: none; text-align: center; margin: 20px 0;">
                    <button onclick="downloadFile()" style="background-color: #27ae60;">⬇️ Descargar Excel Procesado</button>
                </div>
            </div>
            
            <div class="logs" id="logs"></div>
        </div>

        <script>
            let processingInterval;
            
            // Drag and drop functionality
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                fileInput.files = e.dataTransfer.files;
            });
            
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('fileInput');
                
                if (!fileInput.files[0]) {
                    alert('Por favor selecciona un archivo');
                    return;
                }
                
                formData.append('file', fileInput.files[0]);
                
                // Show progress container and logs
                document.getElementById('progressContainer').style.display = 'block';
                document.getElementById('logs').style.display = 'block';
                
                // Disable upload button, enable cancel button
                document.getElementById('uploadBtn').disabled = true;
                document.getElementById('cancelBtn').style.display = 'inline-block';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // Start progress monitoring
                        startProgressMonitoring();
                    } else {
                        showError(result.error);
                    }
                } catch (error) {
                    showError('Error al subir archivo: ' + error.message);
                }
            });
            
            function startProgressMonitoring() {
                processingInterval = setInterval(async () => {
                    try {
                        const response = await fetch('/progress');
                        const data = await response.json();
                        
                        updateProgress(data);
                        
                        if (!data.processing && (data.completed || data.error)) {
                            clearInterval(processingInterval);
                            document.getElementById('uploadBtn').disabled = false;
                            document.getElementById('cancelBtn').style.display = 'none';
                            
                            if (data.completed) {
                                document.getElementById('downloadSection').style.display = 'block';
                            }
                        }
                    } catch (error) {
                        console.error('Error checking progress:', error);
                    }
                }, 1000);
            }
            
            function updateProgress(data) {
                const progressFill = document.getElementById('progressFill');
                const statusDiv = document.getElementById('statusDiv');
                const logs = document.getElementById('logs');
                
                const percentage = data.total > 0 ? (data.current / data.total * 100) : 0;
                progressFill.style.width = percentage + '%';
                
                statusDiv.textContent = data.status;
                statusDiv.className = 'status';
                
                if (data.error) {
                    statusDiv.classList.add('error');
                } else if (data.completed) {
                    statusDiv.classList.add('success');
                }
                
                // Add to logs
                const now = new Date().toLocaleTimeString();
                logs.innerHTML += `[${now}] ${data.status}\\n`;
                logs.scrollTop = logs.scrollHeight;
            }
            
            function showError(message) {
                const statusDiv = document.getElementById('statusDiv');
                statusDiv.textContent = message;
                statusDiv.className = 'status error';
                
                document.getElementById('uploadBtn').disabled = false;
                document.getElementById('cancelBtn').style.display = 'none';
            }
            
            async function cancelProcessing() {
                try {
                    await fetch('/cancel', { method: 'POST' });
                    clearInterval(processingInterval);
                    document.getElementById('uploadBtn').disabled = false;
                    document.getElementById('cancelBtn').style.display = 'none';
                } catch (error) {
                    console.error('Error canceling:', error);
                }
            }
            
            function downloadFile() {
                window.location.href = '/download';
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress_data
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'})
        
        if file and file.filename.endswith(('.xlsx', '.xls')):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            
            # Leer el Excel
            df = pd.read_excel(filepath)
            
            # Reset progress data
            progress_data = {
                'processing': False,
                'current': 0,
                'total': 0,
                'status': 'Archivo cargado, iniciando procesamiento...',
                'completed': False,
                'filename': None,
                'error': None
            }
            
            # Start background processing
            thread = threading.Thread(target=procesar_excel_background, args=(df, filename))
            thread.daemon = True
            thread.start()
            
            return jsonify({'success': True, 'message': 'Archivo cargado, procesamiento iniciado'})
        else:
            return jsonify({'success': False, 'error': 'Solo se permiten archivos Excel (.xlsx, .xls)'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/progress')
def get_progress():
    return jsonify(progress_data)

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    global progress_data
    progress_data['processing'] = False
    progress_data['status'] = 'Procesamiento cancelado por el usuario'
    return jsonify({'success': True})

@app.route('/download')
def download_file():
    global progress_data
    
    if progress_data['completed'] and progress_data['filename']:
        filepath = os.path.join('uploads', progress_data['filename'])
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, 
                           download_name=progress_data['filename'])
    
    return jsonify({'error': 'Archivo no disponible'}), 404

if __name__ == "__main__":
    print("=== CATEGORIZADOR DE NOTICIAS WEB ===")
    print("Iniciando servidor web...")
    print("Ve a tu URL de deployment para usar la aplicación")
    app.run(host='0.0.0.0', port=5000, debug=False)
