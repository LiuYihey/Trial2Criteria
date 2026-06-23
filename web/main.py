import sys
import os
import json
import asyncio
import glob
import numpy as np
import math
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Adjust path to import from the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from clinical_rag.search import EnhancedMedicalSearch, load_config
from clinical_rag.utils import make_json_serializable as _make_json_serializable

app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Load config and initialize the search system globally
# This is more efficient than reloading for every request
try:
    config = load_config()
    
    # --- Configuration Validation ---
    api_key = config.get("api_keys", {}).get("deepseek_api_key")
    if not api_key:
        raise ValueError("CRITICAL: 'deepseek_api_key' is missing or empty in config.json.")
        
    search_system = EnhancedMedicalSearch(config)
    print("[OK] EnhancedMedicalSearch system initialized successfully.")
except Exception as e:
    search_system = None
    print(f"[ERROR] Failed to initialize EnhancedMedicalSearch system: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_title(request: Request):
    """Process the research title and stream results back."""
    data = await request.json()
    title = data.get("title")
    until_year_str = data.get("until_year")

    if not search_system:
        async def error_generator():
            yield f"data: {json.dumps({'error': 'Medical search system is not available.'})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    until_year = None
    if until_year_str:
        try:
            until_year = int(until_year_str)
        except (ValueError, TypeError):
            # Handle the case where until_year is not a valid integer
            async def error_generator():
                yield f"data: {json.dumps({'error': 'Invalid year format provided.'})}\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")

    async def event_generator():
        try:
            # The process_research_title function is now a generator
            for result in search_system.process_research_title(title=title, until_year=until_year):
                yield f"data: {json.dumps(result)}\n\n"
                await asyncio.sleep(0.1)  # Small delay to allow message to be sent
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/generate_criteria")
async def generate_criteria_with_mode(request: Request):
    """Generate final criteria with the specified mode."""
    data = await request.json()
    title = data.get("title")
    mode = data.get("mode", "Standard")
    context_data = data.get("context", {})
    # 从前端获取当前步骤编号，而不是硬编码
    current_step = data.get("current_step", 0)
    
    # Remove debug output
    
    if not search_system:
        return JSONResponse(status_code=500, content={"error": "Medical search system is not available."})
    
    try:
        # Extract context data from the request
        disease_info = context_data.get("disease_info")
        drug_info = context_data.get("drug_info", [])
        papers_data = context_data.get("papers_data")
        trials_data = context_data.get("trials_data", {})
        description = context_data.get("description", "")
        primary_outcome = context_data.get("primary_outcome", "")
        
        # Generate criteria with the specified mode
        criteria_result = search_system.generate_trial_criteria(
            title=title,
            drug_info=drug_info,
            papers=papers_data,
            trials_by_phase=trials_data,
            mode=mode,
            disease_info=disease_info,
            description=description,
            primary_outcome=primary_outcome
        )
        
        # Return the result
        return {
            "step": current_step + 1,  # 使用当前步骤+1作为标准生成步骤的编号
            "name": "Generating Final Inclusion/Exclusion Criteria",
            "data": _make_json_serializable(criteria_result),
            "next_step_name": "Analysis Complete!"
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"--- [ERROR] Failed to generate criteria: {e} ---")
        print(f"--- [ERROR] Traceback: {error_trace} ---")
        
        # 尝试提供更有用的错误消息
        error_msg = str(e)
        if "JSONDecodeError" in error_trace:
            error_msg = "Failed to parse criteria JSON from API response. Please try again."
        elif "AuthenticationError" in error_trace:
            error_msg = "API authentication failed. Please check API key configuration."
        elif "TimeoutError" in error_trace:
            error_msg = "API request timed out. Please try again later."
        
        # 返回结构化的错误响应，包含错误信息但保持与成功响应相似的结构
        return {
            "step": current_step + 1,  # 同样使用当前步骤+1
            "name": "Generating Final Inclusion/Exclusion Criteria",
            "data": {
                "InclusionCriteria": ["Error: Failed to generate criteria."],
                "ExclusionCriteria": ["Error: Failed to generate criteria."],
                "error": error_msg
            },
            "error": error_msg,
            "next_step_name": "Analysis Complete (with errors)"
        }

@app.get("/download-latest-log")
async def download_latest_log():
    log_dir = os.path.join(parent_dir, "generation_logs")
    list_of_files = glob.glob(os.path.join(log_dir, '*.txt'))
    if not list_of_files:
        return JSONResponse(status_code=404, content={"error": "No log files found."})
    
    latest_file = max(list_of_files, key=os.path.getctime)
    return FileResponse(path=latest_file, media_type='application/octet-stream', filename=os.path.basename(latest_file))

if __name__ == "__main__":
    import uvicorn
    # This allows running the app directly for debugging
    # We fetch the host and port from the global config object loaded on startup
    app_config = config.get("app", {})
    host = app_config.get("host", "127.0.0.1")
    port = app_config.get("port", 8000)

    uvicorn.run(app, host=host, port=port) 