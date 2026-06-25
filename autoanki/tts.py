"""TTS 音频生成模块

直接调用 kitten-tts CLI 生成 MP3。
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path

_log = logging.getLogger("autoanki.tts")

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

KITTEN_TTS_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "kitten-tts-x86_64-linux",
    "kitten-tts",
)

MODEL_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--KittenML--kitten-tts-mini-0.8",
    "snapshots",
    "c02725660cea441db4c383af69f1f26f5cd00947",
)


# ---------------------------------------------------------------------------
# 音频生成
# ---------------------------------------------------------------------------


def _safe_filename(text: str, max_len: int = 60) -> str:
    short = text.strip().lower()[:max_len]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in short)
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{safe}_{h}"


def _generate_kitten(text: str, out_path: Path) -> bool:
    """通过 kitten-tts CLI 生成 MP3。"""
    try:
        subprocess.run(
            [KITTEN_TTS_BIN, MODEL_DIR, text, "-o", str(out_path)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return True
    except Exception as e:
        _log.warning("kitten-tts 失败 (%s): %s", text[:30], e)
        return False


def generate_audio(
    text: str,
    *,
    out_dir: Path,
) -> Path | None:
    """生成 MP3。

    Args:
        text: 待合成文本
        out_dir: 输出目录

    Returns:
        MP3 文件路径, 失败返回 None
    """
    fname = _safe_filename(text) + ".mp3"
    out_path = out_dir / fname
    if out_path.exists():
        return out_path

    ok = _generate_kitten(text, out_path)

    if ok:
        _log.debug("TTS OK: %s <- %r", out_path.name, text[:50])
        return out_path
    return None


def generate_audio_batch(
    texts: list[str],
    *,
    out_dir: Path,
) -> dict[str, Path]:
    """批量生成音频, {原始文本 -> MP3路径}. 已存在跳过."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    for text in texts:
        mp3 = generate_audio(text, out_dir=out_dir)
        if mp3:
            results[text] = mp3
    return results


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


def check_tts_available() -> str | None:
    """检查 TTS 是否可用, None=OK, str=错误描述"""
    if not os.path.isfile(KITTEN_TTS_BIN):
        return f"kitten-tts 未找到: {KITTEN_TTS_BIN}"
    if not os.path.isdir(MODEL_DIR):
        return f"模型目录未找到: {MODEL_DIR}"
    return None