#!/usr/bin/env python3
"""
autoanki CLI — 5 阶段词汇牌组 Pipeline

Usage:
  uv run autoanki run                      # 全量跑 (无 TTS)
  uv run autoanki run --tts                # 全量跑 + 音频 (stage 4)
  uv run autoanki run --limit 5            # 只跑前 5 个 chunk
  uv run autoanki run --chunks 10-20       # 只跑 chunk 10-20
  uv run autoanki run --skip-preprocess    # 跳过清洗
  uv run autoanki run --dry-run            # 只看分片

  uv run autoanki chunk                    # 只做 stage1+2
  uv run autoanki infer                    # 只做 stage3
  uv run autoanki tts                      # 只做 stage4 (TTS 音频)
  uv run autoanki package                  # 只做 stage5 (打包)

  uv run autoanki inspect                  # 查看已提取词汇
  uv run autoanki tts-check                # 检查 TTS 就绪状态
"""

from __future__ import annotations
import datetime
import shutil
import time
import tomllib

import json
import logging
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import typer
from genanki import Deck, Model, Note, Package

_log = logging.getLogger("autoanki")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # paths
    ass_dir: Path = Path("ass")
    epub_file: Path = Path("ass") / "diary Of A Wimpy Kid 08] .epub"
    md_file: Path = Path("ass") / "diary_of_a_wimpy_kid.md"
    output_apkg: str = ""
    run_state_dir: Path = Path("_run_state")
    raw_dir: Path = Path("_raw")
    session_id: str = ""  # set at runtime

    # llm
    api_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model_name: str = "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL"
    temperature: float = 0.3
    max_tokens: int = 8064
    llm_timeout: int = 300

    # chunking
    chunk_size: int = 2000
    chunk_overlap: int = 80

    # anki
    anki_model_id: int = 16381234567
    anki_deck_id: int = 20593847561

    # run options (set by CLI)
    limit: int | None = None
    chunk_range: tuple[int, int] | None = None
    skip_preprocess: bool = False
    dry_run: bool = False
    resume: bool = True

    # tts
    tts_enabled: bool = False
    tts_audio_dir: Path = field(default_factory=lambda: Path("_audio"))

    @property
    def raw_file(self) -> Path:
        return self.raw_dir / f"{self.session_id}.txt"


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

def _load_config_file(config_path: Path | None = None) -> dict:
    """加载 autoanki.toml 配置文件, 返回 dict"""
    if config_path is None:
        config_path = Path("autoanki.toml")

    if not config_path.exists():
        return {}

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("autoanki", {})


def _build_config(
    config_file: dict,
    cli_args: dict,
) -> Config:
    """合并配置: CLI 参数 > 配置文件 > 默认值"""
    cfg = Config()

    # 映射配置文件字段到 Config 属性
    field_map = {
        "ass_dir": ("ass_dir", Path),
        "epub_file": ("epub_file", Path),
        "md_file": ("md_file", Path),
        "output_apkg": ("output_apkg", str),
        "run_state_dir": ("run_state_dir", Path),
        "api_url": ("api_url", str),
        "model": ("model_name", str),
        "temperature": ("temperature", float),
        "max_tokens": ("max_tokens", int),
        "llm_timeout": ("llm_timeout", int),
        "chunk_size": ("chunk_size", int),
        "chunk_overlap": ("chunk_overlap", int),
        "anki_model_id": ("anki_model_id", int),
        "anki_deck_id": ("anki_deck_id", int),
        "tts_enabled": ("tts_enabled", bool),
    }

    # 应用配置文件
    for config_key, (attr, _) in field_map.items():
        if config_key in config_file:
            value = config_file[config_key]
            if attr in ("ass_dir", "epub_file", "md_file", "run_state_dir"):
                value = Path(value)
            setattr(cfg, attr, value)

    # CLI 参数覆盖 (非 None 值)
    cli_map = {
        "ass_dir": ("ass_dir", Path),
        "epub_file": ("epub_file", Path),
        "md_file": ("md_file", Path),
        "output": ("output_apkg", str),
        "run_state": ("run_state_dir", Path),
        "api_url": ("api_url", str),
        "model": ("model_name", str),
        "temperature": ("temperature", float),
        "max_tokens": ("max_tokens", int),
        "llm_timeout": ("llm_timeout", int),
        "chunk_size": ("chunk_size", int),
        "chunk_overlap": ("chunk_overlap", int),
        "anki_model_id": ("anki_model_id", int),
        "anki_deck_id": ("anki_deck_id", int),
        "tts": ("tts_enabled", bool),
    }

    for cli_key, (attr, _) in cli_map.items():
        value = cli_args.get(cli_key)
        if value is not None:
            if attr in ("ass_dir", "epub_file", "md_file", "run_state_dir"):
                value = Path(value)
            setattr(cfg, attr, value)

    return cfg


# ---------------------------------------------------------------------------
# Run-state helpers
# ---------------------------------------------------------------------------

def _ensure_run_state(cfg: Config):
    cfg.run_state_dir.mkdir(parents=True, exist_ok=True)


def _save_chunks(chunks: list[str], cfg: Config):
    _ensure_run_state(cfg)
    (cfg.run_state_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log.info(f"缓存分片: {len(chunks)} 个 chunk -> chunks.json")


def _load_chunks(cfg: Config) -> list[str] | None:
    p = cfg.run_state_dir / "chunks.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_progress(cfg: Config) -> set[int]:
    p = cfg.run_state_dir / "progress.json"
    if p.exists():
        return set(json.loads(p.read_text(encoding="utf-8")))
    return set()


def _save_progress(done_ids: set[int], cfg: Config):
    _ensure_run_state(cfg)
    (cfg.run_state_dir / "progress.json").write_text(
        json.dumps(sorted(done_ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _append_result(item: dict, cfg: Config):
    _ensure_run_state(cfg)
    with open(cfg.run_state_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_results(cfg: Config) -> list[dict]:
    p = cfg.run_state_dir / "results.jsonl"
    if not p.exists():
        return []
    items: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _build_audio_map(items: list[dict], audio_dir: Path) -> dict[str, Path]:
    """从已有音频目录构建 {文本 -> MP3路径} 映射"""
    from autoanki import tts
    audio_map: dict[str, Path] = {}
    texts = []
    for item in items:
        texts.append(item["word"])
        texts.append(item["sentence"])
    for text in texts:
        fname = tts._safe_filename(text) + ".mp3"
        p = audio_dir / fname
        if p.exists():
            audio_map[text] = p
    return audio_map


def _ensure_raw_dir(cfg: Config):
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)


def _append_raw(content: str, chunk_idx: int, cfg: Config):
    """追加一条 LLM 原始输出到当前 session 的 raw 文件"""
    _ensure_raw_dir(cfg)
    with open(cfg.raw_file, "a", encoding="utf-8") as f:
        f.write(f"=== chunk {chunk_idx} ===\n")
        f.write(content + "\n\n")


def _parse_raw(cfg: Config) -> list[dict]:
    """从 raw 文件解析所有 || 格式行, 返回 item 列表"""
    if not cfg.raw_file.exists():
        return []
    text = cfg.raw_file.read_text(encoding="utf-8")
    items: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if "||" not in line:
            continue
        line = re.sub(r'^[\d\.\-\s]+', '', line)
        parts = [p.strip() for p in line.split("||")]
        if len(parts) >= 4 and parts[0] and parts[1] and parts[2] and parts[3]:
            items.append({
                "word": parts[0],
                "sentence": parts[1],
                "meaning": parts[2],
                "sentence_cn": parts[3],
            })
    return items


# ---------------------------------------------------------------------------
# Retry mechanism
# ---------------------------------------------------------------------------

def _request_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict,
    max_retries: int = 3,
) -> dict | None:
    """带指数退避的重试请求"""
    for attempt in range(max_retries):
        try:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                _log.warning(f"  请求超时, {wait}s 后重试 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                _log.error(f"  重试 {max_retries} 次后仍超时: {e}")
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    _log.warning(f"  服务器错误 {e.response.status_code}, {wait}s 后重试 ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    _log.error(f"  重试 {max_retries} 次后仍失败: {e.response.status_code}")
                    return None
            else:
                _log.error(f"  请求失败 (不可重试): {e.response.status_code}")
                return None
        except Exception as e:
            _log.error(f"  请求异常: {e}")
            return None
    return None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _health_check(cfg: Config) -> list[str]:
    """启动前检查, 空列表 = OK"""
    errors = []

    if not cfg.skip_preprocess:
        if not cfg.epub_file.exists():
            errors.append(f"[EPUB] 文件不存在: {cfg.epub_file}")

    try:
        client = httpx.Client(timeout=5)
        resp = client.get(cfg.api_url.replace("/chat/completions", "/models"))
        resp.raise_for_status()
        client.close()
    except Exception as e:
        errors.append(f"[LLM] API 不可达 ({e})")

    if cfg.tts_enabled:
        from autoanki import tts
        err = tts.check_tts_available()
        if err:
            errors.append(err)

    return errors


# ---------------------------------------------------------------------------
# Failed chunks tracking
# ---------------------------------------------------------------------------

def _save_failed_chunk(chunk_idx: int, error: str, cfg: Config):
    """记录失败的 chunk"""
    _ensure_run_state(cfg)
    p = cfg.run_state_dir / "failed_chunks.json"
    failed = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    failed[str(chunk_idx)] = {
        "error": str(error),
        "time": datetime.datetime.now().isoformat(),
    }
    p.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_failed_chunks(cfg: Config) -> dict:
    """加载失败的 chunks"""
    p = cfg.run_state_dir / "failed_chunks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _clear_failed_chunks(cfg: Config):
    """清除失败记录"""
    p = cfg.run_state_dir / "failed_chunks.json"
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Stage 1: Preprocess
# ---------------------------------------------------------------------------

def stage1_preprocess(cfg: Config) -> str:
    """用 markitdown 导出 epub -> md, 返回清洗后纯文本"""
    out_path = cfg.ass_dir / "_cleaned.txt"
    md_path = cfg.md_file

    if not md_path.exists():
        _log.info("导出 epub -> md ...")
        with open(md_path, "w") as f:
            subprocess.run(
                ["markitdown", str(cfg.epub_file)],
                stdout=f, check=True,
            )
        _log.info(f"导出完成: {md_path}")

    raw = md_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith(("**Title:**", "**Authors:**", "**Language:**",
                         "**Publisher:**", "**Date:**", "**Description:**",
                         "**Identifier:**")):
            continue
        if s.startswith("![](images/") or s == "":
            continue
        if s in ("Diary Of A Wimpy Kid", "Hard Luck", "Jeff Kinney",
                 "To Charlie"):
            continue
        cleaned.append(s)

    text = "\n".join(cleaned)
    out_path.write_text(text, encoding="utf-8")
    _log.info(f"清洗后文本: {len(text)} 字符")
    return text


# ---------------------------------------------------------------------------
# Stage 2: Chunking
# ---------------------------------------------------------------------------

def _get_sentence_boundaries(text: str) -> list[int]:
    idxs = [0]
    for m in re.finditer(r'(?<=[.!?])\s+', text):
        idxs.append(m.end())
    idxs.append(len(text))
    return idxs


def stage2_chunk(text: str, cfg: Config) -> list[str]:
    """滑动窗口分片, 在句子边界切"""
    boundaries = _get_sentence_boundaries(text)
    chunks: list[str] = []
    start = 0
    b_idx = 0

    while start < len(text):
        target = start + cfg.chunk_size
        end = target

        while b_idx < len(boundaries) and boundaries[b_idx] <= target + 50:
            b_idx += 1
        if b_idx > 0 and boundaries[b_idx - 1] > start:
            end = boundaries[b_idx - 1]

        if abs(end - target) > 150:
            end = target

        if end - start < 600:
            if chunks:
                chunks[-1] += "\n" + text[start:min(end + 200, len(text))]
                start = min(end + 200, len(text))
                continue
            else:
                end = min(target + 200, len(text))

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(end - cfg.chunk_overlap, start + 1)

    chunks = [c for c in chunks if len(c) >= 200]
    _log.info(f"分片: {len(chunks)} 个 chunk")
    return chunks


# ---------------------------------------------------------------------------
# Stage 3: Inference
# ---------------------------------------------------------------------------

def stage3_infer(
    chunks: list[str],
    cfg: Config,
) -> list[dict]:
    """请求本地 llama serve, 原始输出追加到 _raw/<session>.txt, 从文件解析"""
    done_ids = _load_progress(cfg) if cfg.resume else set()

    if cfg.chunk_range:
        c_start, c_end = cfg.chunk_range
        chunks = chunks[c_start:c_end]
        _log.info(f"chunk range [{c_start}:{c_end}] -> {len(chunks)} 个 chunk")

    if cfg.limit is not None and cfg.limit < len(chunks):
        chunks = chunks[:cfg.limit]
        _log.info(f"--limit {cfg.limit} -> 只跑 {cfg.limit} 个 chunk")

    if cfg.resume:
        all_items = _parse_raw(cfg)
        _log.info(f"从断点恢复, 已有 {len(all_items)} 个单词, {len(done_ids)} 个已完成 chunk")
    else:
        all_items = []

    client = httpx.Client(timeout=cfg.llm_timeout)
    system_msg = (
        "你是一个英文词汇提取助手。你的主人是给一个高一英语水平,请从下文提取单词。"
        "每行严格输出格式: 单词 || 原文完整句子 || 中文释义 || 例句中文翻译\n"
        "不要输出任何其他文字或符号。不要用编号。"
        "示例:Dicey || but right now things are just a little dicey. || 不确定的；危险的 || 但现在事情有点不确定。"
    )

    for i, chunk in enumerate(chunks):
        if cfg.resume and i in done_ids:
            _log.info(f"  chunk {i+1}/{len(chunks)} 已跳过 (已完成)")
            continue

        _log.info(f"推理 chunk {i+1}/{len(chunks)} ({len(chunk)} 字符)")

        payload = {
            "model": cfg.model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": chunk},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }

        body = _request_with_retry(client, cfg.api_url, payload)
        if body is None:
            _save_failed_chunk(i, "LLM request failed after retries", cfg)
            continue

        content = body["choices"][0]["message"]["content"].strip()

        # 保存 raw 输出到文件
        _append_raw(content, i, cfg)

        # 从 raw 文件重新解析全部（增量追加，全量解析）
        all_items = _parse_raw(cfg)

        word_count = sum(1 for l in content.splitlines() if "||" in l)
        _log.info(f"  本轮提取: {word_count} 词 (累计 {len(all_items)})")

        if cfg.resume:
            done_ids.add(i)
            _save_progress(done_ids, cfg)

    client.close()

    # 输出推理摘要
    failed = _load_failed_chunks(cfg)
    if failed:
        _log.warning(f"推理摘要: {len(all_items)} 个单词, {len(failed)} 个 chunk 失败")
        _log.warning(f"失败 chunk: {', '.join(sorted(failed.keys(), key=int))}")
    else:
        _log.info(f"推理摘要: {len(all_items)} 个单词, 全部成功")

    # 写出 items 到 results.jsonl, 供 tts/package 子命令使用
    _ensure_run_state(cfg)
    (cfg.run_state_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in all_items) + "\n",
        encoding="utf-8",
    )
    _log.info(f"总提取单词: {len(all_items)}, raw 文件: {cfg.raw_file}")
    return all_items


# ---------------------------------------------------------------------------
# Stage 4: TTS 音频生成 (可选, 独立可断点)
# ---------------------------------------------------------------------------

def _load_tts_progress(cfg: Config) -> set[int]:
    p = cfg.run_state_dir / "tts_progress.json"
    if p.exists():
        return set(json.loads(p.read_text(encoding="utf-8")))
    return set()

def _save_tts_progress(done_ids: set[int], cfg: Config):
    _ensure_run_state(cfg)
    (cfg.run_state_dir / "tts_progress.json").write_text(
        json.dumps(sorted(done_ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )

def stage4_tts(items: list[dict], cfg: Config) -> dict[str, Path]:
    """逐词生成 TTS 音频, 支持断点续跑。返回 {文本 -> MP3路径} 映射。"""
    cfg.tts_audio_dir.mkdir(parents=True, exist_ok=True)
    from autoanki import tts

    done_ids = _load_tts_progress(cfg) if cfg.resume else set()
    audio_map: dict[str, Path] = {}

    # 收集所有需合成的文本
    tts_texts: list[str] = []
    for item in items:
        tts_texts.append(item["word"])
        tts_texts.append(item["sentence"])

    _log.info(f"[Stage 4] TTS 音频生成: {len(tts_texts)} 条")

    for i, text in enumerate(tts_texts):
        if cfg.resume and i in done_ids:
            _log.debug(f"  [{i+1}/{len(tts_texts)}] 跳过 (已有)")
            continue

        _log.debug(f"  [{i+1}/{len(tts_texts)}] 合成: {text[:40]}...")
        mp3 = tts.generate_audio(text, out_dir=cfg.tts_audio_dir)
        if mp3:
            audio_map[text] = mp3

        if cfg.resume:
            done_ids.add(i)
            _save_tts_progress(done_ids, cfg)

    _log.info(f"[Stage 4] TTS 完成: {len(audio_map)}/{len(tts_texts)} 条音频")
    return audio_map


# ---------------------------------------------------------------------------
# Stage 5: Package (无 TTS)
# ---------------------------------------------------------------------------

def stage5_package(items: list[dict], audio_map: dict[str, Path] | None, cfg: Config) -> Path:
    """构建 Anki 牌组并写出 .apkg。audio_map 可选, 来自 stage 4。"""

    media_files: list[str] = sorted({str(p) for p in audio_map.values()}) if audio_map else []

    model = Model(
        cfg.anki_model_id,
        "Wimpy Kid Vocab (Edge)",
        fields=[
            {"name": "Word"},
            {"name": "Sentence"},
            {"name": "Meaning"},
            {"name": "SentenceCN"},
            {"name": "WordAudio"},
            {"name": "SentenceAudio"},
        ],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Word}}<br>{{WordAudio}}",
            "afmt": (
                '{{FrontSide}}<hr id="answer">'
                '<div style="font-size: 16px; font-style: italic; border-left: 3px solid #4CAF50; padding-left: 10px; margin-top: 12px;">{{Sentence}}<br>{{SentenceAudio}}</div>'
                '<div style="font-size: 14px; color: #999; margin-top: 6px;">📖 {{SentenceCN}}</div>'
                '<br>'
                '<div style="font-size: 20px;">{{Meaning}}</div>'
            ),
        }],
        css=(
            ".card { font-family: 'Segoe UI', Arial, sans-serif; "
            "text-align: center; padding: 20px; }"
        ),
    )

    deck = Deck(cfg.anki_deck_id, "Wimpy Kid Vocab (MiniCPM)")

    for item in items:
        word_audio = ""
        sentence_audio = ""
        if audio_map:
            w = audio_map.get(item["word"])
            s = audio_map.get(item["sentence"])
            if w:
                word_audio = f"[sound:{w.name}]"
            if s:
                sentence_audio = f"[sound:{s.name}]"
        note = Note(
            model=model,
            fields=[
                item["word"],
                item["sentence"],
                item["meaning"],
                item.get("sentence_cn", ""),
                word_audio,
                sentence_audio,
            ],
        )
        deck.add_note(note)

    out_path = Path(cfg.output_apkg) if cfg.output_apkg else Path(cfg.epub_file.stem + ".apkg")
    Package(deck, media_files=media_files).write_to_file(str(out_path))
    _log.info(f"写出: {out_path} ({len(items)} 张卡片, {len(media_files)} 个媒体文件)")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: run full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config) -> Path | None:
    cfg.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log.info("=== autoanki Pipeline ===")

    if cfg.skip_preprocess:
        text = (cfg.ass_dir / "_cleaned.txt").read_text(encoding="utf-8")
        _log.info(f"[Stage 1] 跳过预处理, 使用已存在的 cleaned.txt ({len(text)} 字符)")
    else:
        _log.info("[Stage 1] 数据预处理")
        text = stage1_preprocess(cfg)

    _log.info("[Stage 2] 窗口流式分片")
    chunks = stage2_chunk(text, cfg)
    _save_chunks(chunks, cfg)

    if cfg.dry_run:
        _log.info(f"[Dry Run] 共 {len(chunks)} 个 chunk, 跳过推理+打包")
        return None

    _log.info("[Stage 3] 鲁棒性推理")
    items = stage3_infer(chunks, cfg)

    if not items:
        _log.warning("未提取到任何单词, 跳过打包")
        return None

    audio_map: dict[str, Path] = {}
    if cfg.tts_enabled:
        _log.info("[Stage 4] TTS 音频生成")
        audio_map = stage4_tts(items, cfg)

    _log.info("[Stage 5] 结构化打包")
    out = stage5_package(items, audio_map, cfg)

    _log.info(f"=== 完成: {out} ===")
    if not cfg.tts_enabled:
        _log.info("提示: 之后可运行 `autoanki tts && autoanki package --audio-dir _audio` 补加音频")
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_chunk_range(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2:
        raise typer.BadParameter("格式: start-end, 如 10-20")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        raise typer.BadParameter("start 和 end 必须是数字")
    if start < 1 or end < start:
        raise typer.BadParameter("必须 start >= 1 且 end >= start")
    return (start - 1, end)



app = typer.Typer(
    name="autoanki",
    help="Edge AI 词汇预热牌组 Pipeline",
    no_args_is_help=True,
)


@app.command()
def run(
    # config file
    config: Path | None = typer.Option(None, "--config", help="配置文件路径 (autoanki.toml)"),
    # common
    ass_dir: Path = typer.Option(Path("ass"), "--ass-dir", help="资源目录"),
    epub_file: Path | None = typer.Option(None, "--epub", help="输入 epub 文件"),
    md_file: Path | None = typer.Option(None, "--md", help="中间 markdown 文件"),
    output: str | None = typer.Option(None, "--output", "-o", help="输出 .apkg 路径, 默认 = epub文件名.apkg"),
    run_state: Path = typer.Option(Path("_run_state"), "--run-state", help="checkpoint 目录"),
    api_url: str = typer.Option("http://127.0.0.1:8080/v1/chat/completions", "--api-url", help="LLM API 端点"),
    model: str = typer.Option("unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL", "--model", "-m", help="模型名"),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="LLM temperature"),
    max_tokens: int = typer.Option(8064, "--max-tokens", help="LLM max tokens"),
    llm_timeout: int = typer.Option(300, "--llm-timeout", help="LLM 请求超时(秒)"),
    chunk_size: int = typer.Option(2000, "--chunk-size", help="分片目标字符数"),
    chunk_overlap: int = typer.Option(80, "--chunk-overlap", help="分片 overlap 字符数"),
    anki_model_id: int = typer.Option(16381234567, "--anki-model-id", help="Anki Model ID"),
    anki_deck_id: int = typer.Option(20593847561, "--anki-deck-id", help="Anki Deck ID"),
    # run-specific
    limit: int | None = typer.Option(None, "--limit", "-l", help="只跑前 N 个 chunk"),
    chunks: str | None = typer.Option(None, "--chunks", "-c", help="chunk 范围, 如 10-20"),
    skip_preprocess: bool = typer.Option(False, "--skip-preprocess", help="跳过清洗, 用已存在的 cleaned.txt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只看分片结果, 不推理不打包"),
    no_resume: bool = typer.Option(False, "--no-resume", help="禁用断点续跑, 从头跑"),
    reset: bool = typer.Option(False, "--reset", help="清除所有 checkpoint, 从头跑"),
    # tts
    tts: bool = typer.Option(False, "--tts", help="启用 TTS 英文音频 (stage 4, 单词 + 例句)")
):
    """全量跑: preprocess → chunk → infer → [tts] → package"""
    # 加载配置文件
    config_data = _load_config_file(config)
    cfg = _build_config(config_data, {
        "ass_dir": ass_dir,
        "epub_file": epub_file,
        "md_file": md_file,
        "output": output,
        "run_state": run_state,
        "api_url": api_url,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "llm_timeout": llm_timeout,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "anki_model_id": anki_model_id,
        "anki_deck_id": anki_deck_id,
        "tts": tts,
    })
    cfg.limit = limit
    cfg.chunk_range = _parse_chunk_range(chunks) if chunks else None
    cfg.skip_preprocess = skip_preprocess
    cfg.dry_run = dry_run
    cfg.resume = not no_resume

    # 重置 checkpoint
    if reset:
        shutil.rmtree(cfg.run_state_dir, ignore_errors=True)
        shutil.rmtree(cfg.raw_dir, ignore_errors=True)
        _log.info("已清除所有 checkpoint, 从头开始")

    # 健康检查
    errors = _health_check(cfg)
    if errors:
        print("启动前检查失败:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        raise typer.Exit(1)

    run_pipeline(cfg)


@app.command()
def chunk(
    config: Path | None = typer.Option(None, "--config", help="配置文件路径"),
    ass_dir: Path = typer.Option(Path("ass"), "--ass-dir", help="资源目录"),
    epub_file: Path | None = typer.Option(None, "--epub", help="输入 epub 文件"),
    md_file: Path | None = typer.Option(None, "--md", help="中间 markdown 文件"),
    run_state: Path = typer.Option(Path("_run_state"), "--run-state", help="checkpoint 目录"),
    chunk_size: int = typer.Option(2000, "--chunk-size", help="分片目标字符数"),
    chunk_overlap: int = typer.Option(80, "--chunk-overlap", help="分片 overlap 字符数"),
    skip_preprocess: bool = typer.Option(False, "--skip-preprocess", help="跳过清洗"),
):
    """只做 stage1+2, 输出分片到 _run_state/chunks.json"""
    config_data = _load_config_file(config)
    cfg = _build_config(config_data, {
        "ass_dir": ass_dir,
        "epub_file": epub_file,
        "md_file": md_file,
        "run_state": run_state,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })
    cfg.skip_preprocess = skip_preprocess

    if skip_preprocess:
        text = (cfg.ass_dir / "_cleaned.txt").read_text(encoding="utf-8")
    else:
        text = stage1_preprocess(cfg)
    chunks = stage2_chunk(text, cfg)
    _save_chunks(chunks, cfg)
    print(f"分片完成: {len(chunks)} 个 chunk -> {cfg.run_state_dir / 'chunks.json'}")


@app.command()
def infer(
    config: Path | None = typer.Option(None, "--config", help="配置文件路径"),
    run_state: Path = typer.Option(Path("_run_state"), "--run-state", help="checkpoint 目录"),
    api_url: str = typer.Option("http://127.0.0.1:8080/v1/chat/completions", "--api-url", help="LLM API 端点"),
    model: str = typer.Option("unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL", "--model", "-m", help="模型名"),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="LLM temperature"),
    max_tokens: int = typer.Option(8064, "--max-tokens", help="LLM max tokens"),
    llm_timeout: int = typer.Option(300, "--llm-timeout", help="LLM 请求超时(秒)"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="只跑前 N 个 chunk"),
    chunks: str | None = typer.Option(None, "--chunks", "-c", help="chunk 范围, 如 10-20"),
    no_resume: bool = typer.Option(False, "--no-resume", help="禁用断点续跑"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="只重跑失败的 chunks"),
):
    """只做 stage3, 从 _run_state/chunks.json 读取分片"""
    config_data = _load_config_file(config)
    cfg = _build_config(config_data, {
        "run_state": run_state,
        "api_url": api_url,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "llm_timeout": llm_timeout,
    })
    cfg.limit = limit
    cfg.chunk_range = _parse_chunk_range(chunks) if chunks else None
    cfg.resume = not no_resume
    cfg.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 健康检查
    errors = _health_check(cfg)
    if errors:
        print("启动前检查失败:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        raise typer.Exit(1)

    chunks_data = _load_chunks(cfg)
    if chunks_data is None:
        print("错误: 未找到分片缓存, 请先运行 `uv run autoanki chunk`", file=sys.stderr)
        raise typer.Exit(1)

    # 只重跑失败的 chunks
    if retry_failed:
        failed = _load_failed_chunks(cfg)
        if not failed:
            print("没有失败的 chunks")
            return
        failed_ids = [int(k) for k in failed.keys()]
        chunks_data = [chunks_data[i] for i in failed_ids if i < len(chunks_data)]
        _clear_failed_chunks(cfg)
        cfg.resume = False
        _log.info(f"重跑 {len(chunks_data)} 个失败的 chunks")

    items = stage3_infer(chunks_data, cfg)
    print(f"推理完成: {len(items)} 个单词")


@app.command()
def package(
    config: Path | None = typer.Option(None, "--config", help="配置文件路径"),
    run_state: Path = typer.Option(Path("_run_state"), "--run-state", help="checkpoint 目录"),
    output: str | None = typer.Option(None, "--output", "-o", help="输出 .apkg 路径, 默认 = epub文件名.apkg"),
    anki_model_id: int = typer.Option(16381234567, "--anki-model-id", help="Anki Model ID"),
    anki_deck_id: int = typer.Option(20593847561, "--anki-deck-id", help="Anki Deck ID"),
    audio_dir: Path = typer.Option(Path("_audio"), "--audio-dir", help="TTS 音频目录, 已有音频时使用"),
):
    """只做 stage5, 从 _run_state/results.jsonl 读取提取结果打包"""
    config_data = _load_config_file(config)
    cfg = _build_config(config_data, {
        "run_state": run_state,
        "output": output,
        "anki_model_id": anki_model_id,
        "anki_deck_id": anki_deck_id,
    })

    items = _load_results(cfg)
    if not items:
        print("错误: 未找到提取结果 (_run_state/results.jsonl 为空)", file=sys.stderr)
        raise typer.Exit(1)

    # 加载已有音频映射 (如有)
    audio_map: dict[str, Path] = {}
    if audio_dir and audio_dir.exists():
        audio_map = _build_audio_map(items, audio_dir)
        _log.info(f"从 {audio_dir} 加载 {len(audio_map)} 个音频")

    out = stage5_package(items, audio_map, cfg)
    print(f"打包完成: {out}")


@app.command()
def tts(
    config: Path | None = typer.Option(None, "--config", help="配置文件路径"),
    run_state: Path = typer.Option(Path("_run_state"), "--run-state", help="checkpoint 目录"),
    audio_dir: Path = typer.Option(Path("_audio"), "--audio-dir", help="音频输出目录"),
    no_resume: bool = typer.Option(False, "--no-resume", help="禁用断点续跑"),
):
    """只做 stage4 (TTS 音频), 从 _run_state/results.jsonl 读取提取结果"""
    config_data = _load_config_file(config)
    cfg = _build_config(config_data, {
        "run_state": run_state,
    })
    cfg.tts_audio_dir = audio_dir
    cfg.resume = not no_resume
    cfg.tts_enabled = True

    # 健康检查
    errors = _health_check(cfg)
    if errors:
        print("启动前检查失败:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        raise typer.Exit(1)

    items = _load_results(cfg)
    if not items:
        print("错误: 未找到提取结果 (_run_state/results.jsonl 为空)", file=sys.stderr)
        raise typer.Exit(1)
    # ponytail: session_id not needed for TTS, no raw file dependency
    audio_map = stage4_tts(items, cfg)
    print(f"TTS 完成: {len(audio_map)} 条音频 -> {audio_dir}")


@app.command()
def inspect(
    run_state: Path = typer.Option(Path("_run_state"), "--run-state", help="checkpoint 目录"),
    count: int = typer.Option(20, "--count", "-n", help="显示前 N 个单词"),
    dupes: bool = typer.Option(False, "--dupes", help="只显示重复单词"),
    status: bool = typer.Option(False, "--status", help="显示 pipeline 状态概览"),
):
    """查看已提取词汇列表"""
    cfg = Config(run_state_dir=run_state)

    # 状态概览模式
    if status:
        progress = _load_progress(cfg)
        failed = _load_failed_chunks(cfg)
        chunks = _load_chunks(cfg)
        total = len(chunks) if chunks else "?"
        print(f"Pipeline 状态:")
        print(f"  总 chunk:    {total}")
        print(f"  已完成:      {len(progress)}")
        print(f"  失败:        {len(failed)}")
        if failed:
            print(f"\n失败的 chunks:")
            for idx, info in sorted(failed.items(), key=lambda x: int(x[0])):
                print(f"  chunk {idx}: {info['error']}")
        items = _load_results(cfg)
        print(f"\n已提取单词: {len(items)}")
        return

    items = _load_results(cfg)
    if not items:
        print("未找到提取结果 (_run_state/results.jsonl 不存在或为空)")
        raise typer.Exit(0)

    print(f"总单词数: {len(items)}")

    if dupes:
        word_counts = Counter(it["word"].lower() for it in items)
        dupes_list = {w: c for w, c in word_counts.items() if c > 1}
        if dupes_list:
            print(f"\n重复单词 ({len(dupes_list)} 个):")
            for word, c in sorted(dupes_list.items(), key=lambda x: -x[1]):
                print(f"  {word}: {c} 次")
        else:
            print("无重复单词")
        return

    print(f"\n前 {min(count, len(items))} 个单词:")
    for i, item in enumerate(items[:count]):
        print(f"  {i+1}. {item['word']} — {item['meaning']}")
        print(f"     例句: {item['sentence'][:60]}...")


@app.command()
def tts_check():
    """检查 TTS 依赖是否就绪"""
    from autoanki import tts
    err = tts.check_tts_available()
    if err:
        print(f"[FAIL] {err}")
        raise typer.Exit(1)
    print(f"[OK]   binary: {tts.KITTEN_TTS_BIN}")
    print(f"[OK]   model: {tts.MODEL_DIR}")
def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    app()

if __name__ == "__main__":
    main()
