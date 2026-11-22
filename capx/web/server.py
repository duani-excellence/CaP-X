"""FastAPI server for CaP-X interactive web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request as UrlRequest

import tyro
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from capx.envs.configs.instantiate import instantiate
from capx.utils.launch_utils import _load_config
from capx.web.async_trial_runner import LaunchArgsCompat
from capx.web.models import (
    ConfigListResponse,
    InjectPromptCommand,
    LoadConfigRequest,
    LoadConfigResponse,
    SessionState,
    SessionStatusResponse,
    StartTrialRequest,
    StartTrialResponse,
    StateUpdateEvent,
    StopCommand,
    StopTrialRequest,
    StopTrialResponse,
)
from capx.web.session_manager import Session, get_session_manager
from capx.web.viser_ports import find_viser_port, invalidate_viser_port_cache, set_active_viser_port

logging.basicConfig(level=logging.INFO)
from capx.web.process_runner import run_trial_process as run_trial_async

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CaP-X Interactive Web UI",
        description="Real-time interactive interface for CaP-X robot code execution",
        version="1.0.0",
    )

    from capx.web.multitask import router as multitask_router
    app.include_router(multitask_router)

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def stop_workers():
        manager = get_session_manager()
        for row in manager.list_sessions():
            await manager.remove_session(row["session_id"])

    app.add_event_handler("shutdown", stop_workers)

    # ========================================================================
    # REST API Endpoints
    # ========================================================================

    @app.get("/api/default-config")
    async def get_default_config():
        """Return the default config path / model settings from launch.py.

        Loading a page never starts a paid model request automatically.
        """
        default_path = getattr(app.state, "default_config_path", None)
        return {
            "config_path": default_path,
            "auto_start": False,
            "server_url": getattr(
                app.state, "default_server_url", "http://127.0.0.1:8110/chat/completions"
            ),
            "model": getattr(app.state, "default_model", "google/gemini-3.1-pro-preview"),
            "temperature": getattr(app.state, "default_temperature", 1.0),
            "max_tokens": getattr(app.state, "default_max_tokens", 20480),
        }

    @app.get("/api/trial-options")
    async def trial_options():
        from capx.web.selection import MODELS, OBSERVATIONS
        return {"models": MODELS, "observations": OBSERVATIONS}

    @app.get("/api/configs", response_model=ConfigListResponse)
    async def list_configs():
        """List available YAML config files from all environment directories."""
        configs_root = Path("env_configs")
        if not configs_root.exists():
            return ConfigListResponse(configs=[])

        configs = []
        for yaml_file in configs_root.rglob("*.yaml"):
            # Skip hillclimb subdirectories (internal experiment configs)
            if "hillclimb" in yaml_file.parts:
                continue
            configs.append(str(yaml_file.relative_to(".")))

        configs.sort()
        return ConfigListResponse(configs=configs)

    @app.post("/api/load-config", response_model=LoadConfigResponse)
    async def load_config(request: LoadConfigRequest):
        """Load and validate a YAML config file."""
        config_path = request.config_path

        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        try:
            # Create a minimal args object for _load_config
            @dataclass
            class MinimalArgs:
                config_path: str
                server_url: str = "http://127.0.0.1:8110/chat/completions"
                model: str = "google/gemini-3.1-pro-preview"
                temperature: float = 1.0
                max_tokens: int = 20480
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = "google/gemini-3.1-pro-preview"
                visual_differencing_model_server_url: str | None = "http://127.0.0.1:8110/chat/completions"
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = None
                num_workers: int | None = None
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            args = MinimalArgs(config_path=config_path)
            env_factory, config, _ = await asyncio.to_thread(_load_config, args)

            # Extract task prompt (no length limit - UI handles display)
            task_prompt = env_factory.get("cfg", {}).get("prompt", "")

            return LoadConfigResponse(
                status="loaded",
                config_summary={
                    "record_video": config.get("record_video"),
                    "output_dir": config.get("output_dir"),
                    "use_visual_feedback": config.get("use_visual_feedback"),
                    "use_img_differencing": config.get("use_img_differencing"),
                    "total_trials": config.get("total_trials"),
                },
                task_prompt=task_prompt,
            )
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/start-trial", response_model=StartTrialResponse)
    async def start_trial(request: StartTrialRequest):
        """Start a new trial execution."""
        selected_key = None
        from capx.web.multitask import processes as multitask_processes
        if any(p.poll() is None for p in multitask_processes.values()):
            raise HTTPException(409, '多任务作业正在运行，请等待或停止')
        if request.provider is not None or request.observation_mode is not None:
            from capx.web.selection import resolve_selection
            try:
                url, model, selected_key, selected_config = resolve_selection(request.provider, request.observation_mode)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            request = request.model_copy(update={
                "config_path": selected_config, "server_url": url, "model": model,
                "temperature": 0.2, "max_tokens": getattr(app.state, "default_max_tokens", 32768),
                "execution_timeout": 90, "use_visual_feedback": False,
                "use_img_differencing": False,
            })
        config_path = request.config_path
        max_tokens = (request.max_tokens if request.max_tokens is not None
                      else getattr(app.state, "default_max_tokens", 20480))

        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        manager = get_session_manager()

        # Create new session
        session = await manager.create_session()

        try:
            # Build args for config loading
            @dataclass
            class LoadArgs:
                config_path: str
                server_url: str
                model: str
                temperature: float
                max_tokens: int
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = None
                visual_differencing_model_server_url: str | None = None
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = 1
                num_workers: int | None = 1
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            load_args = LoadArgs(
                config_path=config_path,
                server_url=request.server_url,
                model=request.model,
                temperature=request.temperature,
                max_tokens=max_tokens,
                use_visual_feedback=request.use_visual_feedback,
                use_img_differencing=request.use_img_differencing,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
            )
            env_factory, config, _ = await asyncio.to_thread(_load_config, load_args)

            # Process output_dir like launch.py does - add model name to path and create directory
            if config.get("output_dir"):
                from datetime import datetime
                # Add model name and timestamp to output path
                base_dir = config["output_dir"]
                model_slug = request.model.replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + session.session_id[:8]
                new_out_dir = f"{base_dir}/{model_slug}/{timestamp}"
                Path(new_out_dir).mkdir(parents=True, exist_ok=True)
                config["output_dir"] = new_out_dir
                logger.info(f"Output directory set to: {new_out_dir}")

            # Store in session
            session.config_path = config_path
            session.config = config
            session.env_factory = env_factory
            session.state = SessionState.LOADING_CONFIG

            # Prefer request key, then launch.py default, then env.
            api_key = selected_key or (
                getattr(request, "api_key", None)
                or getattr(app.state, "default_api_key", None)
                or os.environ.get("CAPX_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

            # Build launch args for trial runner
            trial_args = LaunchArgsCompat(
                model=request.model,
                server_url=request.server_url,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=request.temperature,
                reasoning_effort="medium",
                debug=False,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
                visual_differencing_model_api_key=api_key,
            )

            # Set the initial settings on session (can be changed during trial)
            session.await_user_input_each_turn = request.await_user_input_each_turn
            session.execution_timeout = request.execution_timeout

            # Start trial in background task
            session.task = asyncio.create_task(
                run_trial_async(
                    session=session,
                    args=trial_args,
                )
            )

            return StartTrialResponse(
                session_id=session.session_id,
                status="started",
            )

        except Exception as e:
            logger.exception(f"Failed to start trial: {e}")
            await manager.remove_session(session.session_id)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/stop", response_model=StopTrialResponse)
    async def stop_trial(request: StopTrialRequest):
        """Stop a running trial."""
        manager = get_session_manager()
        success = await manager.stop_session(request.session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found or not running")

        return StopTrialResponse(status="stopped")

    @app.post("/api/inject-prompt")
    async def inject_prompt(session_id: str, text: str):
        """Inject user prompt text into a running session."""
        manager = get_session_manager()
        success = await manager.inject_prompt(session_id, text)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot inject: session not awaiting input")

        return {"status": "injected"}

    @app.get("/api/session/{session_id}", response_model=SessionStatusResponse)
    async def get_session_status(session_id: str):
        """Get the status of a session."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionStatusResponse(
            session_id=session.session_id,
            state=session.state,
            current_block_index=session.current_block_index,
            total_code_blocks=session.total_code_blocks,
            num_regenerations=session.num_regenerations,
        )

    @app.get("/api/sessions")
    async def list_sessions():
        """List all active sessions."""
        manager = get_session_manager()
        return {"sessions": manager.list_sessions()}

    @app.get("/api/active-session")
    async def get_active_session():
        """Get the currently active session (if any).

        Useful for reconnecting after page refresh.
        """
        manager = get_session_manager()
        session = manager.get_active_session()
        if session:
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "config_path": session.config_path,
            }
        return {"session_id": None}

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket connection for real-time trial updates."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        session.websockets.append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

        # Replay event history so reconnecting clients see all past messages
        if session.event_history:
            logger.info(f"Replaying {len(session.event_history)} events for session {session_id}")
            for event_json in session.event_history:
                try:
                    await websocket.send_text(event_json)
                except Exception:
                    break

        # Send current state
        await websocket.send_text(
            StateUpdateEvent(
                session_id=session_id,
                state=session.state,
            ).model_dump_json()
        )

        try:
            while True:
                # Receive commands from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    msg_type = message.get("type")

                    if msg_type == "stop":
                        logger.info(f"Stop command received for session {session_id}")
                        await manager.stop_session(session_id)

                    elif msg_type == "inject_prompt":
                        text = message.get("text", "")
                        if text:
                            await manager.inject_prompt(session_id, text)
                            logger.info(f"Injected prompt: {text[:50]}...")

                    elif msg_type == "resume":
                        # Put empty string to unblock the queue wait
                        if session.state == SessionState.AWAITING_USER_INPUT:
                            await session.user_injection_queue.put("")

                    elif msg_type == "update_settings":
                        # Update session settings dynamically during a trial
                        if "await_user_input_each_turn" in message:
                            session.await_user_input_each_turn = message["await_user_input_each_turn"]
                            logger.info(f"Updated await_user_input_each_turn to {session.await_user_input_each_turn}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            if websocket in session.websockets:
                session.websockets.remove(websocket)
            # Check if session should be cleaned up
            await manager.on_websocket_disconnect(session_id)

    # ========================================================================
    # Viser reverse proxy  (avoids port-forwarding issues)
    # ========================================================================

    async def _proxy_viser_http(path: str = "", query: str = "") -> Response:
        """Forward an HTTP request to the local Viser server."""
        port = await asyncio.to_thread(find_viser_port)
        if port is None:
            return Response(
                content="Viser server not available — is the trial running?",
                status_code=503,
            )
        url = f"http://localhost:{port}/{path}"
        if query:
            url += f"?{query}"
        try:
            req = UrlRequest(url)
            resp = await asyncio.to_thread(urlopen, req, None, 5)
            content = await asyncio.to_thread(resp.read)
            content_type = resp.headers.get(
                "Content-Type", "application/octet-stream"
            )
            return Response(content=content, media_type=content_type)
        except Exception as exc:
            return Response(content=f"Proxy error: {exc}", status_code=502)

    @app.api_route("/viser-proxy", methods=["GET", "HEAD"])
    async def proxy_viser_root(request: Request):
        """Proxy the Viser root page (no trailing slash)."""
        return await _proxy_viser_http("", str(request.url.query))

    @app.api_route("/viser-proxy/{path:path}", methods=["GET", "HEAD"])
    async def proxy_viser_path(request: Request, path: str):
        """Proxy Viser sub-paths (assets, hdri, etc.)."""
        return await _proxy_viser_http(path, str(request.url.query))

    @app.websocket("/viser-proxy")
    async def proxy_viser_ws(websocket: WebSocket):
        """Reverse-proxy WebSocket connections to the local Viser server.

        The Viser client constructs the WS URL from ``window.location.href``
        (replacing http→ws and stripping the trailing slash), so the iframe at
        ``/viser-proxy/`` connects to ``ws://host/viser-proxy``.
        """
        port = await asyncio.to_thread(find_viser_port)
        if port is None:
            await websocket.close(code=1013, reason="Viser not running")
            return

        # Forward subprotocols (Viser expects "viser-v<VERSION>")
        proto_header = websocket.headers.get("sec-websocket-protocol", "")
        client_protocols = [
            s.strip() for s in proto_header.split(",") if s.strip()
        ]

        import websockets
        from websockets.asyncio.client import connect as ws_connect

        try:
            viser_ws = await ws_connect(
                f"ws://localhost:{port}/",
                subprotocols=(
                    [websockets.Subprotocol(p) for p in client_protocols]
                    if client_protocols
                    else None
                ),
                compression=None,
                max_size=2**24,  # 16 MiB — Viser can send large scene blobs
            )
        except Exception as exc:
            logger.warning(f"Could not connect to Viser WS: {exc}")
            await websocket.close(code=1013, reason=str(exc))
            return

        # Accept the browser connection with Viser's selected subprotocol
        await websocket.accept(subprotocol=viser_ws.subprotocol)

        async def _client_to_viser() -> None:
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        await viser_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await viser_ws.send(msg["text"])
            except (WebSocketDisconnect, Exception):
                pass

        async def _viser_to_client() -> None:
            try:
                async for data in viser_ws:
                    if isinstance(data, bytes):
                        await websocket.send_bytes(data)
                    else:
                        await websocket.send_text(data)
            except Exception:
                pass

        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_client_to_viser()),
                    asyncio.create_task(_viser_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception as exc:
            logger.warning(f"Viser WS proxy error: {exc}")
        finally:
            await viser_ws.close()
            try:
                await websocket.close()
            except Exception:
                pass

    # ========================================================================
    # Static file serving for frontend (production)
    # ========================================================================

    # Check if built frontend exists
    frontend_dist = Path(__file__).parent.parent.parent / "web-ui" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{path:path}")
        async def serve_frontend_routes(path: str):
            # Try to serve static file, otherwise serve index.html for SPA routing
            file_path = frontend_dist / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app


# ============================================================================
# CLI Entry Point
# ============================================================================


@dataclass
class ServerArgs:
    """Command-line arguments for the web server."""

    host: str = "0.0.0.0"
    """Host to bind the server to."""

    port: int = 8200
    """Port to run the server on."""

    reload: bool = False
    """Enable auto-reload for development."""


def main(args: ServerArgs | None = None) -> None:
    """Run the web server."""
    if args is None:
        args = tyro.cli(ServerArgs)

    app = create_app()

    logger.info(f"Starting CaP-X Interactive Web UI on http://{args.host}:{args.port}")
    logger.info("Frontend dev server should be running on http://localhost:5173")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
