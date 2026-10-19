"""
Market Twin • Conector Nativo Rust de Altíssima Performance (C-ABI / ctypes)
Executa a simulação estocástica em threads paralelas Rayon sem GIL e sem overhead de IPC.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("market_twin.native_engine")

_DLL_INSTANCE: Optional[ctypes.CDLL] = None
_NATIVE_LOADED: bool = False


def _load_native_library() -> Optional[ctypes.CDLL]:
    global _DLL_INSTANCE, _NATIVE_LOADED
    if _NATIVE_LOADED:
        return _DLL_INSTANCE

    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "market_twin_engine.dll"),
        os.path.join(os.path.dirname(current_dir), "engine_rust", "target", "release", "market_twin_engine.dll"),
        os.path.join(os.path.dirname(current_dir), "engine_rust", "target", "debug", "market_twin_engine.dll"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                dll = ctypes.CDLL(path)
                # Configurar assinaturas de tipos C-ABI
                dll.run_native_simulation.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
                dll.run_native_simulation.restype = ctypes.c_void_p

                dll.free_native_string.argtypes = [ctypes.c_void_p]
                dll.free_native_string.restype = None

                _DLL_INSTANCE = dll
                _NATIVE_LOADED = True
                logger.info(f"✔ Motor Nativo Rust carregado com sucesso: {path}")
                return dll
            except Exception as e:
                logger.warning(f"Falha ao carregar DLL em {path}: {e}")

    _NATIVE_LOADED = True
    return None


def is_native_available() -> bool:
    """Verifica se o motor nativo compilado está ativo."""
    return _load_native_library() is not None


def run_native_monte_carlo(
    pitch_dict: Dict[str, Any],
    population_size: int,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Executa a simulação estocástica em código de máquina nativo paralelo (Rust + Rayon).
    Retorna None se a biblioteca nativa não estiver compilada.
    """
    dll = _load_native_library()
    if not dll:
        return None

    try:
        input_json_bytes = json.dumps(pitch_dict).encode("utf-8")
        raw_ptr = dll.run_native_simulation(
            ctypes.c_char_p(input_json_bytes),
            ctypes.c_uint64(population_size),
            ctypes.c_uint64(seed),
        )

        if not raw_ptr:
            return None

        try:
            c_str = ctypes.cast(raw_ptr, ctypes.c_char_p).value
            if c_str:
                result_dict = json.loads(c_str.decode("utf-8"))
                return result_dict
            return None
        finally:
            dll.free_native_string(raw_ptr)

    except Exception as e:
        logger.error(f"Erro na execução da biblioteca nativa: {e}")
        return None
