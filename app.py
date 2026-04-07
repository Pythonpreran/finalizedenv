"""
FastAPI Application for ICU Drug Titration Environment.

Exposes OpenEnv-compliant REST endpoints for interacting with the
ICU drug titration RL environment. Supports multiple concurrent sessions.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from grader import grade_trajectory
from icu_env import ICUDrugTitrationEnv
from models import (
    Action,
    BaselineResponse,
    GradeResponse,
    HealthResponse,
    Observation,
    ResetRequest,
    StepResponse,
    TaskInfo,
)
from pharmacology_constants import TASK_DEFINITIONS

# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="ICU Drug Titration Environment",
    description=(
        "An OpenEnv-compliant RL environment for simulating ICU clinical "
        "pharmacology. An AI agent acts as a clinical pharmacist managing "
        "drug titration over a 24-hour simulated patient episode."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = ICUDrugTitrationEnv()

# =============================================================================
# ENDPOINTS
# =============================================================================


@app.head("/")
async def head_root():
    return ""

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/reset", response_model=dict, tags=["Environment"])
async def reset_environment(
    request: Optional[ResetRequest] = None,
    session_id: Optional[str] = Query(None, description="Session ID (auto-generated if not provided)"),
):
    """
    Reset the environment for a new episode.

    Returns the initial observation and session ID for subsequent calls.
    """
    try:
        # ✅ FIX: handle missing body (OpenEnv requirement)
        if request is None:
            task_id = "easy"
            seed = None
        else:
            task_id = request.task_id
            seed = request.seed

        observation, sid = env.reset(
            task_id=task_id,
            session_id=session_id,
            seed=seed,
        )
        return {
            "observation": observation.model_dump(),
            "session_id": sid,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step_environment(
    action: Action,
    session_id: str = Query("default", description="Session ID from reset"),
):
    """
    Execute one step in the environment.

    The Action model is the direct request body (not wrapped).
    Returns observation, reward, done, terminated, truncated, and info.
    """
    try:
        observation, reward, done, info = env.step(
            action=action,
            session_id=session_id,
        )
        terminated = info.get("terminated", False)
        truncated = info.get("truncated", False)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", tags=["Environment"])
async def get_state(
    session_id: str = Query(..., description="Session ID"),
):
    """Get the full environment state for a session."""
    try:
        state = env.state(session_id)
        return state.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks", response_model=list[TaskInfo], tags=["Environment"])
async def get_tasks():
    """List all available tasks with their configurations."""
    tasks = []
    for task_id, task in TASK_DEFINITIONS.items():
        tasks.append(TaskInfo(
            task_id=task_id,
            name=task["name"],
            description=task["description"],
            disease=task["disease"],
            horizon=task["horizon"],
            difficulty=task["difficulty"],
            allowed_drugs=task["allowed_drugs"],
        ))
    return tasks


@app.get("/grader", response_model=GradeResponse, tags=["Evaluation"])
async def grade_episode(
    session_id: str = Query(..., description="Session ID to grade"),
):
    """
    Grade a completed episode trajectory.

    Returns a deterministic score between 0.0 and 1.0.
    """
    try:
        state = env.state(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not done yet. Complete the episode before grading.",
        )

    score, breakdown = grade_trajectory(state)

    return GradeResponse(
        task_id=state.task_id,
        score=score,
        breakdown=breakdown,
        total_steps=state.current_step,
        done_reason=state.done_reason,
    )


@app.get("/baseline", response_model=BaselineResponse, tags=["Evaluation"])
async def get_baseline():
    """
    Run a simple heuristic baseline and return scores.

    The baseline uses a fixed strategy for each task:
    - Easy: Add norepinephrine at 0.1 mcg/kg/min
    - Medium: Add norepinephrine + propofol at moderate doses
    - Hard: Conservative multi-drug approach
    """
    from models import ActionType

    scores = {}
    details = {}

    # --- Easy baseline ---
    obs, sid = env.reset(task_id="easy", seed=42)
    # Add norepinephrine at moderate dose
    action = Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.1)
    obs, reward, done, info = env.step(action, sid)
    # Maintain for remaining steps
    while not done:
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.HOLD),
            sid,
        )
    state = env.state(sid)
    easy_score, easy_breakdown = grade_trajectory(state)
    scores["easy"] = easy_score
    details["easy"] = easy_breakdown

    # --- Medium baseline ---
    obs, sid = env.reset(task_id="medium", seed=42)
    action = Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.05)
    obs, reward, done, info = env.step(action, sid)
    action = Action(action_type=ActionType.ADD_DRUG, drug="propofol", dose=30.0)
    obs, reward, done, info = env.step(action, sid)
    while not done:
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.HOLD),
            sid,
        )
    state = env.state(sid)
    med_score, med_breakdown = grade_trajectory(state)
    scores["medium"] = med_score
    details["medium"] = med_breakdown

    # --- Hard baseline ---
    obs, sid = env.reset(task_id="hard", seed=42)
    action = Action(action_type=ActionType.ADD_DRUG, drug="norepinephrine", dose=0.15)
    obs, reward, done, info = env.step(action, sid)
    action = Action(action_type=ActionType.ADD_DRUG, drug="insulin", dose=5.0)
    obs, reward, done, info = env.step(action, sid)
    while not done:
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.HOLD),
            sid,
        )
    state = env.state(sid)
    hard_score, hard_breakdown = grade_trajectory(state)
    scores["hard"] = hard_score
    details["hard"] = hard_breakdown

    return BaselineResponse(scores=scores, details=details)


# =============================================================================
# STATIC FILE SERVING (Frontend Dashboard)
# =============================================================================

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", tags=["UI"])
    async def serve_dashboard():
        """Serve the ICU monitoring dashboard."""
        return FileResponse(os.path.join(static_dir, "index.html"))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
