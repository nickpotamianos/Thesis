#swarm_control/bridge.py
from __future__ import annotations
import asyncio, json, time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

try:
    # Optional dependency; only required in mavsdk mode
    import mavsdk
    from mavsdk.offboard import OffboardError, VelocityNedYaw
except Exception:
    mavsdk = None

@dataclass
class VehicleBinding:
    sys_addr: str            # e.g. "udp://:14540" or "serial:///dev/ttyUSB0:57600"
    frame: str = "ned"       # "ned" (world-aligned in our sim) or "local"
    yaw_deg: float = 0.0

class ControlBridge:
    """
    Publish/simulate vantage moves at a fixed rate.
    Modes:
      - none: no-op
      - sim: mutate an internal state (for quick end-to-end smoke testing)
      - mavsdk: send velocity-NED setpoints to each bound vehicle
    """
    def __init__(self, mode: str = "none", rate_hz: int = 5,
                 bindings: Optional[Dict[str, VehicleBinding]] = None):
        self.mode = mode
        self.rate_hz = int(rate_hz)
        self.bindings = bindings or {}
        self._sim_state: Dict[str, np.ndarray] = {}   # id -> p(3)

        # MAVSDK session
        self._mav_tasks_started = False
        self._mav_drones: Dict[str, any] = {}  # id -> System

    # ---------- SIM BACKEND ----------
    def set_sim_pose(self, robot_id: str, p_xyz: np.ndarray):
        self._sim_state[robot_id] = np.asarray(p_xyz, float).reshape(3)

    def get_sim_pose(self, robot_id: str) -> Optional[np.ndarray]:
        return self._sim_state.get(robot_id, None)

    # ---------- PUBLIC API ----------
    def send_vantage_moves(self, t: float, moves: Dict[str, np.ndarray]):
        if self.mode == "none":
            return
        if self.mode == "sim":
            # Apply displacement directly to internal sim state; print JSON for logs
            log = {}
            for rid, dv in moves.items():
                dv = np.asarray(dv, float).reshape(3)
                p = self._sim_state.get(rid, np.zeros(3))
                self._sim_state[rid] = p + dv
                log[rid] = [float(dv[0]), float(dv[1]), float(dv[2])]
            print(f"[CTRL sim] t={t:.2f}, moves={json.dumps(log)}")
            return
        if self.mode == "mavsdk":
            if mavsdk is None:
                print("[CTRL] MAVSDK not installed; cannot send commands.")
                return
            # Fire-and-forget asyncio task to stream one-shot velocities toward the desired Δx
            asyncio.get_event_loop().create_task(self._send_mavsdk_once(moves))

    # ---------- MAVSDK BACKEND ----------
    async def _ensure_connected(self):
        if self._mav_tasks_started:
            return
        for rid, bind in self.bindings.items():
            sys = mavsdk.System()
            await sys.connect(system_address=bind.sys_addr)
            self._mav_drones[rid] = sys
            async for state in sys.core.connection_state():
                if state.is_connected:
                    break
            # Enter offboard with a neutral setpoint
            try:
                await sys.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, bind.yaw_deg))
                await sys.offboard.start()
            except OffboardError as e:
                print(f"[CTRL] Offboard for {rid} failed: {e._result.result}")
        self._mav_tasks_started = True

    async def _send_mavsdk_once(self, moves: Dict[str, np.ndarray]):
        await self._ensure_connected()
        dt = 1.0 / max(1, self.rate_hz)
        for rid, dv in moves.items():
            if rid not in self._mav_drones:
                continue
            v_ned = dv / dt  # naive mapping: Δx per tick -> velocity
            v_ned = np.clip(v_ned, -2.0, 2.0)  # quick cap; tighten via CLI later
            vx, vy, vz = float(v_ned[0]), float(v_ned[1]), float(-v_ned[2])  # ENU->NED z sign
            try:
                await self._mav_drones[rid].offboard.set_velocity_ned(
                    VelocityNedYaw(vx, vy, vz, self.bindings[rid].yaw_deg)
                )
            except OffboardError as e:
                print(f"[CTRL] Velocity set failed for {rid}: {e._result.result}")
