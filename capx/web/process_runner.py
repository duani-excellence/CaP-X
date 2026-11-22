"""Trusted Web orchestration in a spawned worker; generated code stays in P0 bwrap.

Only JSON events cross back to the Web process. Environment calls execute on the
worker main thread. The parent watchdog remains responsive during native calls.
"""
import asyncio
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

from capx.web.models import SessionState, ErrorEvent, StateUpdateEvent


def physical_task_success(info, reward, truncated):
    return bool(info.get('physical_success') and info.get('sandbox_rc') == 0
                and reward == 1 and not truncated)


class WireEvent:
    def __init__(self, data):
        self.data = data
        self.type = data['type']

    def model_dump_json(self):
        return json.dumps(self.data, ensure_ascii=False)


def _worker(payload, events, commands):
    from capx.web.session_manager import Session
    from capx.web.async_trial_runner import run_trial_async, LaunchArgsCompat

    async def main():
        session = Session(session_id=payload['session_id'], config=payload['config'],
                          config_path=payload['config_path'], env_factory=payload['env_factory'],
                          execution_timeout=payload['execution_timeout'],
                          await_user_input_each_turn=payload['await_user_input_each_turn'])
        session.process_worker = True
        session.websockets = [True]  # Parent owns actual sockets and reconnect history.

        def publish(event):
            events.put(json.dumps({'kind':'event','event':event.model_dump(mode='json')}))
        async def emit(event):
            publish(event)
        session.emit = emit
        session.publish_event = publish
        session.publish_viser_port = lambda port: events.put(json.dumps({'kind':'viser','port':port}))

        async def receive():
            while True:
                try:
                    data = json.loads(commands.get_nowait())
                except queue.Empty:
                    await asyncio.sleep(.05)
                    continue
                if data['kind'] == 'prompt':
                    await session.user_injection_queue.put(data['text'])
                elif data['kind'] == 'settings':
                    session.await_user_input_each_turn = data['await_user_input_each_turn']
        receiver = asyncio.create_task(receive())
        try:
            await run_trial_async(session, LaunchArgsCompat(**payload['args']))
            events.put(json.dumps({'kind':'finished'}))
            # Preserve the final Viser view until stop/replacement shuts down this worker.
            while True:
                await asyncio.sleep(1)
        finally:
            receiver.cancel()
            if session.env is not None:
                session.env.close()
    asyncio.run(main())


def _stop(process):
    if process.is_alive():
        process.terminate()
    process.join(.5)
    if process.is_alive():
        process.kill()
        process.join(.5)


async def run_trial_process(session, args, *, _target=_worker):
    ctx = mp.get_context('spawn')
    events, commands = ctx.Queue(), ctx.Queue()
    payload = dict(session_id=session.session_id, config=session.config,
                   config_path=session.config_path, env_factory=session.env_factory,
                   args=vars(args), execution_timeout=session.execution_timeout,
                   await_user_input_each_turn=session.await_user_input_each_turn)
    process = ctx.Process(target=_target, args=(payload, events, commands), daemon=False)
    process.start()
    session.worker_pid = process.pid
    deadline = time.monotonic() + 180
    phase = 'environment initialization'
    finished = False
    settings = None
    audit = None
    try:
        if session.config.get('output_dir'):
            path = Path(session.config['output_dir']) / 'web-worker.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            audit = path.open('x')
            audit.write(json.dumps({'event':'start','pid':process.pid,'session_id':session.session_id})+'\n')
            audit.flush()
        while True:
            if session.cancel_event.is_set():
                raise asyncio.CancelledError()
            if not process.is_alive():
                raise RuntimeError(f'Web worker exited unexpectedly (exit code {process.exitcode})')
            # Bounded draining keeps timeout and cancellation responsive even during streaming.
            for _ in range(128):
                try:
                    row = json.loads(events.get_nowait())
                except queue.Empty:
                    break
                if row['kind'] == 'viser':
                    from capx.web.viser_ports import set_active_viser_port
                    set_active_viser_port(row['port'])
                elif row['kind'] == 'finished':
                    finished = True
                    deadline = None
                elif row['kind'] == 'event':
                    data = row['event']; kind = data['type']
                    if kind == 'state_update':
                        session.state = SessionState(data['state'])
                        if session.state == SessionState.AWAITING_USER_INPUT:
                            deadline = None
                        elif session.state == SessionState.RUNNING:
                            deadline = time.monotonic()+240
                    elif kind == 'code_execution_start':
                        phase = 'code execution'
                        deadline = time.monotonic()+min(session.execution_timeout,90)
                        session.current_block_index = data['block_index']
                    elif kind == 'code_execution_result':
                        phase = 'post-execution processing'
                        deadline = time.monotonic()+180
                    elif kind == 'model_streaming_start':
                        phase = 'model generation'
                        deadline = time.monotonic()+240
                    elif kind == 'trial_complete':
                        session.total_code_blocks = data['num_code_blocks']
                        session.num_regenerations = data['num_regenerations']
                    await session.emit(WireEvent(data))
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f'Web worker timeout during {phase}; worker terminated, next trial creates a fresh environment')
            if not finished:
                current = session.await_user_input_each_turn
                if settings != current:
                    commands.put(json.dumps({'kind':'settings','await_user_input_each_turn':current}))
                    settings = current
                while not session.user_injection_queue.empty():
                    commands.put(json.dumps({'kind':'prompt','text':session.user_injection_queue.get_nowait()}))
            await asyncio.sleep(.025)
    except asyncio.CancelledError:
        session.state = SessionState.IDLE
        raise
    except Exception as exc:
        session.state = SessionState.ERROR
        await session.emit(ErrorEvent(session_id=session.session_id, message=str(exc), recoverable=True))
        await session.emit(StateUpdateEvent(session_id=session.session_id,state=SessionState.ERROR))
    finally:
        _stop(process)
        session.worker_pid = None
        from capx.web.viser_ports import invalidate_viser_port_cache
        invalidate_viser_port_cache()
        for channel in (events, commands):
            channel.cancel_join_thread()
            channel.close()
        if audit is not None:
            audit.write(json.dumps({'event':'stop','exitcode':process.exitcode,'state':session.state.value})+'\n')
            audit.close()
