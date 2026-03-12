from __future__ import annotations

import numpy as np

from .base import SimulationBackend, SimulationResult


class NumPyReferenceBackend(SimulationBackend):
    name = "numpy"

    def run_forward(
        self,
        velocity: np.ndarray,
        source: tuple[int, int, int],
        receivers: list[tuple[int, int, int]],
        wavelet: np.ndarray,
        save_wavefield: bool,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> SimulationResult:
        velocity = np.asarray(velocity, dtype=np.float32)
        wavelet = np.asarray(wavelet, dtype=np.float32)
        nt = int(wavelet.shape[0])
        recorded = np.zeros((nt, len(receivers)), dtype=np.float32)
        wavefield = np.zeros((nt, *velocity.shape), dtype=np.float32) if save_wavefield else None
        prev = np.zeros_like(velocity, dtype=np.float32)
        curr = np.zeros_like(velocity, dtype=np.float32)
        coeff = (velocity * dt) ** 2
        inv_dx2 = 1.0 / max(dx * dx, 1e-6)
        inv_dy2 = 1.0 / max(dy * dy, 1e-6)
        inv_dz2 = 1.0 / max(dz * dz, 1e-6)

        sx, sy, sz = source
        for it in range(nt):
            lap = np.zeros_like(curr)
            lap[1:-1, 1:-1, 1:-1] = (
                (curr[1:-1, 1:-1, 2:] - 2.0 * curr[1:-1, 1:-1, 1:-1] + curr[1:-1, 1:-1, :-2]) * inv_dx2
                + (curr[1:-1, 2:, 1:-1] - 2.0 * curr[1:-1, 1:-1, 1:-1] + curr[1:-1, :-2, 1:-1]) * inv_dy2
                + (curr[2:, 1:-1, 1:-1] - 2.0 * curr[1:-1, 1:-1, 1:-1] + curr[:-2, 1:-1, 1:-1]) * inv_dz2
            )
            nxt = 2.0 * curr - prev + coeff * lap
            nxt[sz, sy, sx] += wavelet[it]
            nxt[0, :, :] = 0.0
            nxt[-1, :, :] = 0.0
            nxt[:, 0, :] = 0.0
            nxt[:, -1, :] = 0.0
            nxt[:, :, 0] = 0.0
            nxt[:, :, -1] = 0.0
            for ir, (rx, ry, rz) in enumerate(receivers):
                recorded[it, ir] = nxt[rz, ry, rx]
            if wavefield is not None:
                wavefield[it] = nxt
            prev, curr = curr, nxt

        diagnostics = {
            "backend": self.name,
            "nt": nt,
            "receiver_count": len(receivers),
            "max_amplitude": float(np.max(np.abs(recorded))) if recorded.size else 0.0,
        }
        return SimulationResult(recorded_data=recorded, forward_wavefield=wavefield, diagnostics=diagnostics)

    def run_rtm(
        self,
        velocity: np.ndarray,
        source: tuple[int, int, int],
        receivers: list[tuple[int, int, int]],
        wavelet: np.ndarray,
        observed_data: np.ndarray,
        forward_wavefield: np.ndarray | None,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> SimulationResult:
        velocity = np.asarray(velocity, dtype=np.float32)
        observed = np.asarray(observed_data, dtype=np.float32)
        nt = int(observed.shape[0])
        if forward_wavefield is None:
            forward_wavefield = self.run_forward(
                velocity=velocity,
                source=source,
                receivers=receivers,
                wavelet=np.asarray(wavelet, dtype=np.float32),
                save_wavefield=True,
                dt=dt,
                dx=dx,
                dy=dy,
                dz=dz,
            ).forward_wavefield
        assert forward_wavefield is not None
        image = np.zeros_like(velocity, dtype=np.float32)
        prev = np.zeros_like(velocity, dtype=np.float32)
        curr = np.zeros_like(velocity, dtype=np.float32)
        coeff = (velocity * dt) ** 2
        inv_dx2 = 1.0 / max(dx * dx, 1e-6)
        inv_dy2 = 1.0 / max(dy * dy, 1e-6)
        inv_dz2 = 1.0 / max(dz * dz, 1e-6)

        reverse_steps = 0
        for it in range(nt - 1, -1, -1):
            lap = np.zeros_like(curr)
            lap[1:-1, 1:-1, 1:-1] = (
                (curr[1:-1, 1:-1, 2:] - 2.0 * curr[1:-1, 1:-1, 1:-1] + curr[1:-1, 1:-1, :-2]) * inv_dx2
                + (curr[1:-1, 2:, 1:-1] - 2.0 * curr[1:-1, 1:-1, 1:-1] + curr[1:-1, :-2, 1:-1]) * inv_dy2
                + (curr[2:, 1:-1, 1:-1] - 2.0 * curr[1:-1, 1:-1, 1:-1] + curr[:-2, 1:-1, 1:-1]) * inv_dz2
            )
            nxt = 2.0 * curr - prev + coeff * lap
            for ir, (rx, ry, rz) in enumerate(receivers):
                nxt[rz, ry, rx] += observed[it, ir]
            nxt[0, :, :] = 0.0
            nxt[-1, :, :] = 0.0
            nxt[:, 0, :] = 0.0
            nxt[:, -1, :] = 0.0
            nxt[:, :, 0] = 0.0
            nxt[:, :, -1] = 0.0
            image += nxt * forward_wavefield[it]
            prev, curr = curr, nxt
            reverse_steps += 1

        diagnostics = {
            "backend": self.name,
            "nt": nt,
            "image_norm": float(np.linalg.norm(image)),
            "reverse_steps": reverse_steps,
        }
        return SimulationResult(image=image, forward_wavefield=forward_wavefield, diagnostics=diagnostics)
