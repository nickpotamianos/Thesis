class ControlBridge:
    def __init__(self, mode: str = "none", rate_hz: int = 5):
        self.mode = mode
        self.rate_hz = int(rate_hz)

    def send_vantage_moves(self, t: float, moves: dict):
        """
        Publish or simulate control commands for vantage moves.
        - none: no-op
        - sim: stub; print or log moves
        - mavsdk: placeholder for MAVSDK publisher
        """
        if self.mode == "none":
            return
        if self.mode == "sim":
            # Minimal stub: could be extended to perturb state inside caller
            try:
                print(f"[CTRL sim] t={t:.2f}, moves={{{k: [float(v[0]), float(v[1]), float(v[2])] for k,v in moves.items()}}}")
            except Exception:
                pass
        elif self.mode == "mavsdk":
            # Placeholder for MAVSDK backend; intentionally a no-op here
            pass

