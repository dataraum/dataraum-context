"""File upload endpoint with auto-pipeline execution."""

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import select

from dataraum.api.deps import SessionDep
from dataraum.api.schemas import FileUploadResponse
from dataraum.core.connections import get_connection_manager
from dataraum.pipeline.db_models import PipelineRun
from dataraum.pipeline.orchestrator import PipelineConfig, run_pipeline
from dataraum.core.config import load_phase_config, load_pipeline_config
from dataraum.storage import Source

router = APIRouter()


def _run_pipeline_sync(
    source_id: str,
    run_id: str,
) -> dict:
    """Run the pipeline synchronously."""
    manager = get_connection_manager()

    # Load per-phase configs
    pipeline_yaml = load_pipeline_config()
    active_phases = pipeline_yaml.get("phases", [])
    phase_configs = {name: load_phase_config(name) for name in active_phases}

    config = PipelineConfig(skip_completed=True)

    return run_pipeline(
        manager=manager,
        source_id=source_id,
        target_phase=None,  # Run all phases
        config=config,
        phase_configs=phase_configs,
        run_id=run_id,
    )


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    session: SessionDep,
    file: UploadFile = File(..., description="CSV file to analyze"),
    auto_run: bool = True,
) -> FileUploadResponse:
    """Upload a CSV file and optionally run the pipeline.

    This endpoint:
    1. Saves the uploaded file to the data directory
    2. Creates a source record
    3. Optionally runs the full pipeline (default: yes)
    4. Returns the source_id for subsequent queries

    Use auto_run=false to upload without processing (you can trigger pipeline later).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Only accept CSV for now
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Only CSV files are supported. Please upload a .csv file."
        )

    manager = get_connection_manager()
    output_dir = manager.output_dir

    # Create data directory if needed
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = data_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create source name from filename
    source_name = Path(file.filename).stem

    # Check if source already exists
    existing = session.execute(
        select(Source).where(Source.name == source_name)
    ).scalar_one_or_none()

    if existing:
        source_id = existing.source_id
        # Check if pipeline already completed
        last_run = session.execute(
            select(PipelineRun)
            .where(PipelineRun.source_id == source_id)
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if last_run and last_run.status == "completed":
            return FileUploadResponse(
                source_id=source_id,
                source_name=source_name,
                file_name=file.filename,
                message="File already processed. Use existing source_id for queries.",
                pipeline_status="completed",
                run_id=last_run.run_id,
            )
    else:
        # Create new source
        source_id = str(uuid4())
        source = Source(
            source_id=source_id,
            name=source_name,
            source_type="csv",
            connection_config={"path": str(file_path)},
        )
        session.add(source)
        session.flush()

    if not auto_run:
        return FileUploadResponse(
            source_id=source_id,
            source_name=source_name,
            file_name=file.filename,
            message="File uploaded. Use POST /api/v1/sources/{source_id}/run to start pipeline.",
            pipeline_status="pending",
            run_id=None,
        )

    # Run the pipeline synchronously
    run_id = str(uuid4())
    try:
        result = _run_pipeline_sync(source_id, run_id)
        status = result.get("status", "unknown")

        if status == "completed":
            return FileUploadResponse(
                source_id=source_id,
                source_name=source_name,
                file_name=file.filename,
                message="Pipeline completed successfully. Data is ready for analysis.",
                pipeline_status="completed",
                run_id=run_id,
            )
        else:
            return FileUploadResponse(
                source_id=source_id,
                source_name=source_name,
                file_name=file.filename,
                message=f"Pipeline finished with status: {status}",
                pipeline_status=status,
                run_id=run_id,
            )
    except Exception as e:
        return FileUploadResponse(
            source_id=source_id,
            source_name=source_name,
            file_name=file.filename,
            message=f"Pipeline failed: {str(e)}",
            pipeline_status="failed",
            run_id=run_id,
        )
