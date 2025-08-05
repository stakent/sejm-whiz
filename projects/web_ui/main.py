#!/usr/bin/env python3
"""Web UI server for Sejm Whiz monitoring dashboard."""

import asyncio
import subprocess
from datetime import datetime, UTC
from typing import AsyncGenerator, Optional
import requests

import uvicorn
from fastapi import FastAPI, HTTPException
from sejm_whiz import __version__
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    RedirectResponse,
    JSONResponse,
)

app = FastAPI(
    title="Sejm Whiz Web UI",
    description="Web interface for monitoring Sejm Whiz data processing pipeline",
    version=__version__,
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
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": __version__,
        "hot_reload_test": "✅ Hot reload working!",
    }


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
            <a href="/search" class="nav-link">🔍 Search</a>
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
            <p style="color: #48bb78; margin-top: 10px; font-weight: bold;">
                🔥 Hot reload is working perfectly! No more Docker builds!
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
            <a href="/search" class="nav-link">🔍 Search</a>
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
                document.getElementById('logContent').innerHTML = ''; // Clear initial text
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
    """Stream real processor logs from Kubernetes."""

    async def log_generator() -> AsyncGenerator[str, None]:
        kubectl_available = False
        pod_selector = None

        # Try GPU processor first, then CPU processor
        processor_labels = [
            "app=sejm-whiz-processor-gpu",
            "app=sejm-whiz-processor-cpu",
            "app=data-processor",
        ]

        # Check which processor pods are available
        for label in processor_labels:
            try:
                # Try with kubectl first
                check_process = await asyncio.create_subprocess_exec(
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    "sejm-whiz",
                    "-l",
                    label,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await check_process.communicate()
                if (
                    check_process.returncode == 0
                    and b"No resources found" not in stdout
                    and stdout.strip()
                ):
                    kubectl_available = True
                    pod_selector = label
                    yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Found processor pods with label: {label}\n\n"
                    break
            except (FileNotFoundError, OSError):
                # If kubectl not found, try with full path
                try:
                    check_process = await asyncio.create_subprocess_exec(
                        "/usr/local/bin/kubectl",
                        "get",
                        "pods",
                        "-n",
                        "sejm-whiz",
                        "-l",
                        label,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await check_process.communicate()
                    if (
                        check_process.returncode == 0
                        and b"No resources found" not in stdout
                        and stdout.strip()
                    ):
                        kubectl_available = True
                        pod_selector = label
                        yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - Found processor pods with label: {label}\n\n"
                        break
                except (FileNotFoundError, OSError):
                    continue

        if kubectl_available and pod_selector:
            try:
                # Stream logs from Kubernetes pod
                process = await asyncio.create_subprocess_exec(
                    "kubectl",
                    "logs",
                    "-f",
                    "-n",
                    "sejm-whiz",
                    "-l",
                    pod_selector,
                    "--tail=50",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                if process.stdout:
                    async for line in process.stdout:
                        yield f"data: {line.decode('utf-8')}\n\n"
                    return
            except Exception as e:
                yield f"data: {datetime.now(UTC).isoformat()} - dashboard - ERROR - Failed to stream logs: {e}\n\n"

        # Fallback: Generate demo logs
        yield f"data: {datetime.now(UTC).isoformat()} - dashboard - INFO - No live processor found, showing demo logs\n\n"

        batch_num = 1
        while True:
            logs = [
                f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Starting batch {batch_num}",
                f"{datetime.now(UTC).isoformat()} - sejm_ingestion - INFO - [SAMPLE] Fetching new proceedings",
                f"{datetime.now(UTC).isoformat()} - text_processing - INFO - [SAMPLE] Processing batch {batch_num}",
                f"{datetime.now(UTC).isoformat()} - embedding_generation - INFO - [SAMPLE] Generating embeddings",
                f"{datetime.now(UTC).isoformat()} - database_storage - INFO - [SAMPLE] Storing results",
                f"{datetime.now(UTC).isoformat()} - data_processor - INFO - [SAMPLE] Batch {batch_num} completed",
            ]
            for log in logs:
                yield f"data: {log}\n\n"
                await asyncio.sleep(1)
            batch_num += 1
            await asyncio.sleep(3)

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/search", response_class=HTMLResponse)
async def search_page():
    """Serve the semantic search interface."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sejm Whiz - Semantic Search</title>
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
        .container { max-width: 1200px; margin: 0 auto; }
        .search-header {
            background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); text-align: center;
        }
        .search-form {
            background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .search-input {
            width: 100%; padding: 15px; font-size: 1.1em; border: 2px solid #e2e8f0;
            border-radius: 8px; margin-bottom: 20px; transition: border-color 0.3s ease;
        }
        .search-input:focus {
            outline: none; border-color: #667eea;
        }
        .search-options {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-bottom: 20px;
        }
        .option-group {
            display: flex; flex-direction: column; gap: 8px;
        }
        .option-group label {
            font-weight: 500; color: #2d3748; font-size: 0.9em;
        }
        .option-group select, .option-group input {
            padding: 8px; border: 1px solid #e2e8f0; border-radius: 6px;
        }
        .search-button {
            background: #667eea; color: white; padding: 12px 30px;
            border: none; border-radius: 8px; font-size: 1.1em; font-weight: 500;
            cursor: pointer; transition: background 0.3s ease;
        }
        .search-button:hover { background: #5a67d8; }
        .search-button:disabled { background: #a0aec0; cursor: not-allowed; }
        .results-container {
            background: white; border-radius: 12px; padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); display: none;
        }
        .results-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #e2e8f0;
        }
        .result-item {
            border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px;
            margin-bottom: 15px; transition: box-shadow 0.3s ease;
        }
        .result-item:hover {
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .result-title {
            font-weight: 600; color: #2d3748; margin-bottom: 8px; font-size: 1.1em;
        }
        .result-content {
            color: #4a5568; line-height: 1.6; margin-bottom: 10px;
        }
        .result-metadata {
            display: flex; gap: 15px; font-size: 0.9em; color: #718096;
        }
        .result-score {
            background: #667eea; color: white; padding: 2px 8px; border-radius: 12px;
            font-size: 0.8em; font-weight: 500;
        }
        .loading {
            text-align: center; padding: 40px; color: #718096;
        }
        .query-analysis {
            background: #f7fafc; border-radius: 8px; padding: 15px; margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }
        .analysis-title {
            font-weight: 600; color: #2d3748; margin-bottom: 8px;
        }
        .analysis-content {
            font-size: 0.9em; color: #4a5568;
        }
    </style>
</head>
<body>
    <nav class="top-nav">
        <a href="/" class="nav-brand">🚀 Sejm Whiz</a>
        <div class="nav-links">
            <a href="/search" class="nav-link active">🔍 Search</a>
            <a href="/dashboard" class="nav-link">📊 Dashboard</a>
            <a href="/docs" class="nav-link">📚 API Docs</a>
            <a href="/health" class="nav-link">❤️ Health</a>
            <a href="/home" class="nav-link">🏠 Home</a>
        </div>
    </nav>
    <div class="container">
        <div class="search-header">
            <h1>🔍 Semantic Search</h1>
            <p style="font-size: 1.1em; color: #718096; margin-top: 10px;">
                Search through Polish parliamentary proceedings and legal documents using AI-powered semantic search
            </p>
        </div>

        <div class="search-form">
            <form id="searchForm">
                <input
                    type="text"
                    id="searchQuery"
                    class="search-input"
                    placeholder="Enter your search query in Polish (e.g., 'prawo pracy', 'ustawa o ochronie danych')..."
                    required
                >

                <div class="search-options">
                    <div class="option-group">
                        <label for="searchMode">Search Mode</label>
                        <select id="searchMode">
                            <option value="hybrid">Hybrid (Recommended)</option>
                            <option value="semantic_only">Semantic Only</option>
                            <option value="cross_register">Cross-Register</option>
                            <option value="legal_focused">Legal Focused</option>
                        </select>
                    </div>

                    <div class="option-group">
                        <label for="documentType">Document Type</label>
                        <select id="documentType">
                            <option value="">All Documents</option>
                            <option value="sejm_proceeding">Sejm Proceedings</option>
                            <option value="legal_act">Legal Acts</option>
                            <option value="amendment">Amendments</option>
                        </select>
                    </div>

                    <div class="option-group">
                        <label for="maxResults">Max Results</label>
                        <select id="maxResults">
                            <option value="10">10</option>
                            <option value="25">25</option>
                            <option value="50">50</option>
                        </select>
                    </div>

                    <div class="option-group">
                        <label for="threshold">Similarity Threshold</label>
                        <input type="range" id="threshold" min="0.3" max="0.9" step="0.1" value="0.5">
                        <span id="thresholdValue">0.5</span>
                    </div>
                </div>

                <div style="display: flex; gap: 15px; align-items: center;">
                    <label><input type="checkbox" id="queryExpansion" checked> Query Expansion</label>
                    <label><input type="checkbox" id="crossRegister" checked> Cross-Register Matching</label>
                </div>

                <div style="text-align: center; margin-top: 20px;">
                    <button type="submit" class="search-button" id="searchButton">
                        🔍 Search Documents
                    </button>
                </div>
            </form>
        </div>

        <div class="results-container" id="resultsContainer">
            <div class="results-header">
                <h2 id="resultsTitle">Search Results</h2>
                <div id="resultsInfo"></div>
            </div>

            <div id="queryAnalysis" class="query-analysis" style="display: none;">
                <div class="analysis-title">Query Analysis</div>
                <div class="analysis-content" id="analysisContent"></div>
            </div>

            <div id="resultsContent"></div>
        </div>
    </div>

    <script>
        let isSearching = false;

        // Update threshold display
        document.getElementById('threshold').addEventListener('input', function() {
            document.getElementById('thresholdValue').textContent = this.value;
        });

        // Handle search form submission
        document.getElementById('searchForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await performSearch();
        });

        async function performSearch() {
            if (isSearching) return;

            const query = document.getElementById('searchQuery').value.trim();
            if (!query) return;

            isSearching = true;
            const searchButton = document.getElementById('searchButton');
            const resultsContainer = document.getElementById('resultsContainer');
            const resultsContent = document.getElementById('resultsContent');

            // Update UI state
            searchButton.disabled = true;
            searchButton.textContent = '🔄 Searching...';
            resultsContainer.style.display = 'block';
            resultsContent.innerHTML = '<div class="loading">🔍 Searching through legal documents...</div>';

            try {
                // Prepare search parameters
                const searchParams = {
                    query: query,
                    limit: parseInt(document.getElementById('maxResults').value),
                    threshold: parseFloat(document.getElementById('threshold').value),
                    document_type: document.getElementById('documentType').value || null,
                    enable_query_expansion: document.getElementById('queryExpansion').checked,
                    enable_cross_register: document.getElementById('crossRegister').checked,
                    search_mode: document.getElementById('searchMode').value
                };

                // Perform search
                const response = await fetch('/api/v1/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(searchParams)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Search failed');
                }

                const data = await response.json();
                displayResults(data);

            } catch (error) {
                console.error('Search error:', error);
                resultsContent.innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #e53e3e;">
                        ❌ Search failed: ${error.message}
                        <br><br>
                        <small>Make sure the API server is running and accessible.</small>
                    </div>
                `;
            } finally {
                isSearching = false;
                searchButton.disabled = false;
                searchButton.textContent = '🔍 Search Documents';
            }
        }

        function displayResults(data) {
            const resultsInfo = document.getElementById('resultsInfo');
            const resultsContent = document.getElementById('resultsContent');
            const queryAnalysis = document.getElementById('queryAnalysis');
            const analysisContent = document.getElementById('analysisContent');

            // Update results info
            resultsInfo.innerHTML = `
                <div style="font-size: 0.9em; color: #718096;">
                    ${data.total_results} results in ${data.processing_time_ms.toFixed(1)}ms
                </div>
            `;

            // Display query analysis if available
            if (data.processed_query) {
                const analysis = data.processed_query;
                analysisContent.innerHTML = `
                    <strong>Normalized Query:</strong> ${analysis.normalized_query}<br>
                    <strong>Query Type:</strong> ${analysis.query_type}<br>
                    ${analysis.legal_terms && analysis.legal_terms.length > 0 ?
                        `<strong>Legal Terms:</strong> ${analysis.legal_terms.join(', ')}<br>` : ''}
                    ${analysis.expanded_terms && analysis.expanded_terms.length > 0 ?
                        `<strong>Expanded Terms:</strong> ${analysis.expanded_terms.join(', ')}<br>` : ''}
                    ${analysis.legal_references && analysis.legal_references.length > 0 ?
                        `<strong>Legal References:</strong> ${analysis.legal_references.join(', ')}<br>` : ''}
                `;
                queryAnalysis.style.display = 'block';
            } else {
                queryAnalysis.style.display = 'none';
            }

            // Display results
            if (data.results && data.results.length > 0) {
                resultsContent.innerHTML = data.results.map(result => `
                    <div class="result-item">
                        <div class="result-title">${escapeHtml(result.title)}</div>
                        <div class="result-content">${escapeHtml(result.content)}</div>
                        <div class="result-metadata">
                            <span class="result-score">${(result.similarity_score * 100).toFixed(1)}%</span>
                            <span>Type: ${result.document_type}</span>
                            <span>ID: ${result.document_id}</span>
                            ${result.metadata && result.metadata.cross_register_score ?
                                `<span>Cross-Register: ${(result.metadata.cross_register_score * 100).toFixed(1)}%</span>` : ''}
                        </div>
                    </div>
                `).join('');
            } else {
                resultsContent.innerHTML = `
                    <div style="text-align: center; padding: 40px; color: #718096;">
                        🔍 No results found for your query.
                        <br><br>
                        Try adjusting your search terms or lowering the similarity threshold.
                    </div>
                `;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Focus search input on page load
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('searchQuery').focus();
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


# API proxy endpoints for semantic search
@app.post("/api/v1/search")
async def proxy_search(request_data: dict):
    """Proxy search requests to the main API server."""
    try:
        # Forward to main API server running on port 8000
        response = requests.post(
            "http://localhost:8000/api/v1/search",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API server error: {response.text}",
            )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to API server. Make sure it's running on port 8000.",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504, detail="API server timeout. The search is taking too long."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.get("/api/v1/search")
async def proxy_search_get(
    q: str,
    limit: int = 10,
    threshold: float = 0.5,
    document_type: Optional[str] = None,
    expand_query: bool = True,
    cross_register: bool = True,
    search_mode: str = "hybrid",
):
    """Proxy GET search requests to the main API server."""
    params = {
        "q": q,
        "limit": limit,
        "threshold": threshold,
        "expand_query": expand_query,
        "cross_register": cross_register,
        "search_mode": search_mode,
    }
    if document_type:
        params["document_type"] = document_type

    try:
        response = requests.get(
            "http://localhost:8000/api/v1/search",
            params={k: v for k, v in params.items() if v is not None},
            timeout=30,
        )

        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API server error: {response.text}",
            )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to API server. Make sure it's running on port 8000.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.get("/api/processor/status")
async def processor_status():
    """Get the current status of the data processor."""
    try:
        # Try GPU processor first, then CPU processor
        processor_labels = [
            "app=sejm-whiz-processor-gpu",
            "app=sejm-whiz-processor-cpu",
            "app=data-processor",
        ]

        for label in processor_labels:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    "sejm-whiz",
                    "-l",
                    label,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                import json

                pods_data = json.loads(result.stdout)
                if pods_data.get("items"):
                    pod = pods_data["items"][0]
                    processor_type = (
                        "GPU"
                        if "gpu" in label
                        else "CPU"
                        if "cpu" in label
                        else "Unknown"
                    )
                    return {
                        "status": pod["status"]["phase"],
                        "processor_type": processor_type,
                        "pod_name": pod["metadata"]["name"],
                        "started_at": pod["status"].get("startTime"),
                        "container_statuses": pod["status"].get(
                            "containerStatuses", []
                        ),
                        "label_selector": label,
                    }
    except Exception:
        pass

    # Fallback status
    return {
        "status": "unknown",
        "message": "Unable to determine processor status",
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
