#!/usr/bin/env python3
"""Web UI server for Sejm Whiz monitoring dashboard."""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI(
    title="Sejm Whiz Web UI", 
    description="Web interface for monitoring Sejm Whiz data processing pipeline",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Redirect to home page for better user experience."""
    return RedirectResponse(url="/home")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "0.1.0"}

@app.get("/home", response_class=HTMLResponse)
async def home():
    """Serve the home page."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sejm Whiz - Home</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        .top-nav {
            position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            padding: 0 20px; height: 60px; display: flex; align-items: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .nav-brand {
            font-weight: 600; font-size: 1.2em; color: #2d3748;
            margin-right: 30px; text-decoration: none;
        }
        .nav-links {
            display: flex; gap: 20px; align-items: center;
        }
        .nav-link {
            text-decoration: none; color: #4a5568; font-weight: 500;
            padding: 8px 16px; border-radius: 6px; transition: all 0.3s ease;
        }
        .nav-link:hover {
            background: rgba(102, 126, 234, 0.1); color: #667eea;
        }
        .nav-link.active {
            background: #667eea; color: white;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 70px 20px 20px 20px; margin: 0;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .hero {
            background: white; border-radius: 12px; padding: 40px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); text-align: center;
        }
        .features {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; margin-top: 30px;
        }
        .feature-card {
            background: white; border-radius: 12px; padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .feature-card h3 { color: #2d3748; margin-bottom: 15px; }
        .feature-card p { color: #718096; line-height: 1.6; }
        .btn {
            display: inline-block; background: #667eea; color: white;
            padding: 12px 24px; border-radius: 6px; text-decoration: none;
            font-weight: 500; margin-top: 20px; transition: all 0.3s ease;
        }
        .btn:hover { background: #5a67d8; }
    </style>
</head>
<body>
    <nav class="top-nav">
        <a href="/" class="nav-brand">🚀 Sejm Whiz</a>
        <div class="nav-links">
            <a href="/dashboard" class="nav-link">📊 Dashboard</a>
            <a href="/docs" class="nav-link">📚 API Docs</a>
            <a href="/health" class="nav-link">❤️ Health</a>
            <a href="/home" class="nav-link active">🏠 Home</a>
        </div>
    </nav>
    <div class="container">
        <div class="hero">
            <h1>🚀 Sejm Whiz</h1>
            <p style="font-size: 1.2em; color: #718096; margin-top: 15px;">
                AI-driven legal prediction system using Polish Parliament data
            </p>
            <p style="color: #4a5568; margin-top: 20px;">
                Monitor parliamentary proceedings and legal documents to predict future law changes 
                with multi-act amendment detection and cross-reference analysis.
            </p>
            <a href="/dashboard" class="btn">📊 View Dashboard</a>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h3>🏛️ Sejm Data Processing</h3>
                <p>Real-time ingestion and processing of Polish Parliament proceedings, 
                   debates, and voting records using advanced NLP techniques.</p>
            </div>
            <div class="feature-card">
                <h3>⚖️ Legal Document Analysis</h3>
                <p>Integration with ELI API for effective law data, analyzing legal 
                   documents and tracking amendments across multiple acts.</p>
            </div>
            <div class="feature-card">
                <h3>🤖 AI Predictions</h3>
                <p>Machine learning models using bag of embeddings with HerBERT 
                   for semantic similarity and legal change prediction.</p>
            </div>
            <div class="feature-card">
                <h3>🔍 Semantic Search</h3>
                <p>Vector-based search using PostgreSQL + pgvector for finding 
                   related legal documents and cross-references.</p>
            </div>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sejm Whiz Data Processor Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        .top-nav {
            position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            padding: 0 20px; height: 60px; display: flex; align-items: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .nav-brand {
            font-weight: 600; font-size: 1.2em; color: #2d3748;
            margin-right: 30px; text-decoration: none;
        }
        .nav-links {
            display: flex; gap: 20px; align-items: center;
        }
        .nav-link {
            text-decoration: none; color: #4a5568; font-weight: 500;
            padding: 8px 16px; border-radius: 6px; transition: all 0.3s ease;
        }
        .nav-link:hover {
            background: rgba(102, 126, 234, 0.1); color: #667eea;
        }
        .nav-link.active {
            background: #667eea; color: white;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 70px 20px 20px 20px; margin: 0;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .header h1 { color: #2d3748; margin-bottom: 10px; font-size: 2em; }
        .header p { color: #718096; font-size: 1.1em; }
        .status-bar { display: flex; gap: 20px; margin-bottom: 30px; }
        .status-card {
            flex: 1; background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .status-card h3 {
            color: #718096; font-size: 0.9em; text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 10px;
        }
        .status-value { font-size: 1.8em; font-weight: bold; color: #2d3748; }
        .log-container {
            background: #1a202c; border-radius: 12px; padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); 
            height: 500px; display: flex; flex-direction: column;
        }
        .log-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #2d3748;
            flex-shrink: 0;
        }
        .log-title { color: #e2e8f0; font-size: 1.2em; font-weight: 600; }
        .log-controls { display: flex; gap: 10px; }
        .btn {
            padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer;
            font-size: 0.9em; font-weight: 500; transition: all 0.3s ease;
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-secondary { background: #4a5568; color: white; }
        .btn-autoscroll-on { background: #48bb78; color: white; }
        .btn-autoscroll-off { background: #ed8936; color: white; }
        #logContent {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em; line-height: 1.6; color: #e2e8f0; white-space: pre-wrap;
            flex: 1; overflow-y: auto; scroll-behavior: smooth;
            border: 1px solid #2d3748; border-radius: 6px; padding: 10px;
        }
        .log-line { padding: 4px 0; margin-bottom: 2px; }
        .log-info { color: #63b3ed; }
        .log-success { color: #68d391; }
        .connection-status {
            position: fixed; bottom: 20px; right: 20px; padding: 10px 20px;
            border-radius: 20px; font-size: 0.9em; font-weight: 500;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .connected { background: #48bb78; color: white; }
    </style>
</head>
<body>
    <nav class="top-nav">
        <a href="/" class="nav-brand">🚀 Sejm Whiz</a>
        <div class="nav-links">
            <a href="/dashboard" class="nav-link active">📊 Dashboard</a>
            <a href="/docs" class="nav-link">📚 API Docs</a>
            <a href="/health" class="nav-link">❤️ Health</a>
            <a href="/home" class="nav-link">🏠 Home</a>
        </div>
    </nav>
    <div class="container">
        <div class="header">
            <h1>🚀 Sejm Whiz Data Processor Monitor</h1>
            <p>Real-time monitoring of data ingestion and processing pipeline</p>
        </div>

        <div class="status-bar">
            <div class="status-card">
                <h3>Processor Status</h3>
                <div class="status-value">✅ Web UI Running</div>
            </div>
            <div class="status-card">
                <h3>Documents Processed</h3>
                <div class="status-value" id="docsProcessed">0</div>
            </div>
            <div class="status-card">
                <h3>Current Pipeline</h3>
                <div class="status-value" id="currentPipeline">Demo Mode</div>
            </div>
            <div class="status-card">
                <h3>Last Update</h3>
                <div class="status-value" id="lastUpdate">-</div>
            </div>
        </div>

        <div class="log-container">
            <div class="log-header">
                <div class="log-title">📊 Live Logs</div>
                <div class="log-controls">
                    <button class="btn btn-secondary" onclick="clearLogs()">Clear</button>
                    <button class="btn btn-autoscroll-on" id="autoScrollBtn" onclick="toggleAutoScroll()">
                        <span id="autoScrollText">🔄 Auto-scroll: ON</span>
                    </button>
                    <button class="btn btn-primary" onclick="toggleConnection()">
                        <span id="connectionToggle">Pause</span>
                    </button>
                </div>
            </div>
            <div id="logContent">Connecting to log stream...</div>
        </div>
    </div>

    <div id="connectionStatus" class="connection-status connected">
        ✅ Connected
    </div>

    <script>
        let eventSource = null;
        let autoScroll = true;
        let isConnected = false;
        let stats = { docsProcessed: 0, currentPipeline: 'Demo Mode', lastUpdate: new Date() };

        function connectToLogStream() {
            if (eventSource) eventSource.close();
            eventSource = new EventSource('/api/logs/stream');
            
            eventSource.onopen = function() {
                isConnected = true;
                document.getElementById('connectionStatus').innerHTML = '✅ Connected';
                addLogLine('✅ Connected to log stream', 'success');
            };

            eventSource.onmessage = function(event) {
                addLogLine(event.data);
                stats.docsProcessed++;
                updateStats();
            };

            eventSource.onerror = function() {
                isConnected = false;
                document.getElementById('connectionStatus').innerHTML = '❌ Disconnected';
                addLogLine('❌ Connection lost. Attempting to reconnect...', 'error');
                setTimeout(() => { if (!isConnected) connectToLogStream(); }, 5000);
            };
        }

        function addLogLine(line, type = '') {
            const logContent = document.getElementById('logContent');
            const logLine = document.createElement('div');
            logLine.className = 'log-line';
            if (line.includes('INFO') || type === 'success') logLine.className += ' log-success';
            if (line.includes('ERROR') || type === 'error') logLine.style.color = '#fc8181';
            logLine.textContent = line;
            logContent.appendChild(logLine);
            
            // Ensure auto-scroll works reliably
            if (autoScroll) {
                // Use requestAnimationFrame for smoother scrolling
                requestAnimationFrame(() => {
                    logContent.scrollTop = logContent.scrollHeight;
                });
            }
        }

        function updateStats() {
            document.getElementById('docsProcessed').textContent = stats.docsProcessed;
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }

        function clearLogs() {
            document.getElementById('logContent').innerHTML = '';
            addLogLine('🧹 Logs cleared', 'success');
        }

        function toggleAutoScroll() {
            autoScroll = !autoScroll;
            const btn = document.getElementById('autoScrollBtn');
            const text = document.getElementById('autoScrollText');
            
            if (autoScroll) {
                text.textContent = '🔄 Auto-scroll: ON';
                btn.className = 'btn btn-autoscroll-on';
                // Scroll to bottom when re-enabled
                requestAnimationFrame(() => {
                    document.getElementById('logContent').scrollTop = document.getElementById('logContent').scrollHeight;
                });
            } else {
                text.textContent = '⏸️ Auto-scroll: OFF';
                btn.className = 'btn btn-autoscroll-off';
            }
        }

        function toggleConnection() {
            if (isConnected) {
                eventSource.close();
                isConnected = false;
                document.getElementById('connectionToggle').textContent = 'Resume';
                addLogLine('⏸️ Paused log streaming', 'info');
            } else {
                connectToLogStream();
                document.getElementById('connectionToggle').textContent = 'Pause';
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            connectToLogStream();
            setInterval(updateStats, 1000);
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.get("/api/logs/stream")
async def stream_logs():
    """Stream demo logs for the dashboard."""
    async def log_generator() -> AsyncGenerator[str, None]:
        yield f"data: {datetime.utcnow().isoformat()} - web_ui - INFO - Demo log stream started\\n\\n"
        
        batch_num = 1
        while True:
            logs = [
                f"{datetime.utcnow().isoformat()} - data_processor - INFO - Starting batch {batch_num}",
                f"{datetime.utcnow().isoformat()} - sejm_ingestion - INFO - Fetching new proceedings",
                f"{datetime.utcnow().isoformat()} - text_processing - INFO - Processing batch {batch_num}",
                f"{datetime.utcnow().isoformat()} - embedding_generation - INFO - Generating embeddings",
                f"{datetime.utcnow().isoformat()} - database_storage - INFO - Storing results",
                f"{datetime.utcnow().isoformat()} - data_processor - INFO - Batch {batch_num} completed",
            ]
            for log in logs:
                yield f"data: {log}\\n\\n"
                await asyncio.sleep(1)
            batch_num += 1
            await asyncio.sleep(3)
    
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)