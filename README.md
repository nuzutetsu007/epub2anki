# epub2anki — Edge AI 词汇预热牌组 Pipeline

从英文 EPUB 提取 GRE/六级/考研难度词汇，生成 Anki 牌组 (.apkg)。

本地推理，无 API 费用。适配 llama.cpp server (RTX 3060)。

---

## 快速开始

```bash
# 全量跑 (无 TTS)
epub2anki run

# 全量跑 (含 TTS 音频)
epub2anki run --tts

# 只看分片结果
epub2anki run --dry-run

# 只跑前 5 个 chunk
epub2anki run --limit 5

# 查看已提取词汇
epub2anki inspect
```

**不传 `-o` 时输出文件名 = EPUB 文件名 + `.apkg`。**

---

## 架构：5 阶段闭环

```
Stage 1        Stage 2        Stage 3        Stage 4       Stage 5
EPUB → MD   窗口流式分片    本地 LLM 推理   TTS 音频生成   结构化打包
preprocess    chunk          infer           tts            package
─────────────────────────────────────────────────────────────────────
ass/*.epub   _run_state/    _run_state/     _audio/        *.apkg
             chunks.json    results.jsonl   *.mp3
```

---

## 部署与模型配置

### 硬件要求

| 组件 | 最低 | 推荐 |
|---|---|---|
| GPU VRAM | 2 GB | 6 GB (RTX 3060) |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 6 GB | 12 GB |

### LLM 模型 (核心)

使用 `llama.cpp` 加载 GGUF 量化模型，提供兼容 OpenAI 的 API。

```bash
llama serve --no-mmproj \
  -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL

# → 监听 127.0.0.1:8080
# → -hf 首次自动下载 GGUF
```

### TTS 模型 (可选)

使用 `KittenML/kitten-tts` 全家桶。模型由 HuggingFace Hub 自动缓存到 `~/.cache/huggingface/hub/`。

### 项目安装

```bash
git clone https://github.com/nuzutetsu007/epub2anki
cd epub2anki
uv add genanki httpx typer
```

### 启动检查

```bash
epub2anki tts-check          # 检查 TTS 依赖
epub2anki run --dry-run      # 健康检查 (LLM API + EPUB)
```

---

## CLI 命令

| 命令 | 功能 |
|---|---|
| `epub2anki run` | 全量 pipeline, 自动断点续跑 |
| `epub2anki chunk` | 只做 stage 1+2 (预处理 + 分片) |
| `epub2anki infer` | 只做 stage 3 (LLM 推理) |
| `epub2anki tts` | 只做 stage 4 (TTS 音频) |
| `epub2anki package` | 只做 stage 5 (打包 .apkg) |
| `epub2anki inspect` | 查看已提取词汇 |
| `epub2anki inspect --status` | pipeline 状态概览 |
| `epub2anki inspect --dupes` | 查看重复词汇 |
| `epub2anki tts-check` | 检查 TTS 依赖是否就绪 |

## 全部选项

### 通用选项 (适用于 run / chunk / infer / tts / package)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config PATH` | `autoanki.toml` | 配置文件路径 |
| `--run-state PATH` | `_run_state` | checkpoint 目录 |

### 输入输出 (run / chunk / infer)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ass-dir PATH` | `ass` | 素材目录 |
| `--epub PATH` | — | 输入 epub 文件 |
| `--md PATH` | — | 中间 markdown 文件 |
| `--output / -o PATH` | EPUB文件名.apkg | 输出 .apkg 路径 |

### LLM 配置 (run / infer)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--api-url TEXT` | `http://127.0.0.1:8080/v1/chat/completions` | LLM API 端点 |
| `--model / -m TEXT` | `unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL` | 模型名 |
| `--temperature / -t FLOAT` | `0.3` | LLM temperature |
| `--max-tokens INT` | `8064` | LLM max tokens |
| `--llm-timeout INT` | `300` | LLM 请求超时(秒) |

### 分片配置 (run / chunk)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--chunk-size INT` | `2000` | 分片目标字符数 |
| `--chunk-overlap INT` | `80` | 分片 overlap 字符数 |

### 执行控制 (run / infer)

| 参数 | 说明 |
|---|---|
| `--limit N` / `-l` | 只跑前 N 个 chunk |
| `--chunks M-N` / `-c` | chunk 范围, 如 `10-20` |
| `--no-resume` | 禁用断点续跑, 从头跑 |
| `--reset` | 清除所有 checkpoint, 从头跑 |
| `--retry-failed` | 只重跑失败的 chunks (仅 infer) |

### 分支控制 (run / chunk)

| 参数 | 说明 |
|---|---|
| `--skip-preprocess` | 跳过清洗, 用已存在的 cleaned.txt |
| `--dry-run` | 只看分片结果, 不推理不打包 |

### TTS (run / tts)

| 参数 | 说明 |
|---|---|
| `--tts` | 启用 TTS 英文音频 (仅 run) |
| `--audio-dir PATH` | 音频目录 (默认 `_audio`) |

### inspect 选项

| 参数 | 说明 |
|---|---|
| `--count N` / `-n` | 显示前 N 个单词 (默认 20) |
| `--status` | 显示 pipeline 状态概览 |
| `--dupes` | 只显示重复单词 |

## TTS 音频

### 一次性启用

```bash
epub2anki run --tts
# → 全量 pipeline, 生成英文音频 + 打包
```

### 后补 TTS

如果先跑 `epub2anki run` 无 TTS, 之后想补音频:

```bash
epub2anki tts                    # 从 results.jsonl 生成音频到 _audio/
epub2anki package                # 重新打包 (自动加载 _audio/)
# 或指定输出路径:
epub2anki package -o my_vocab.apkg
```

### 检查 TTS 依赖

```bash
epub2anki tts-check
# → 验证 kitten-tts 可用
```

## 配置文件 autoanki.toml

优先级: **CLI 参数 > 配置文件 > 默认值**

```toml
# cp autoanki.toml.example autoanki.toml

[autoanki]
# 输入/输出
# epub_file = "ass/my_book.epub"
# md_file = "ass/my_book.md"
# output_apkg = "output.apkg"

# LLM 配置
# api_url = "http://127.0.0.1:8080/v1/chat/completions"
# model = "unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL"
# temperature = 0.3
# max_tokens = 8064
# llm_timeout = 300

# 分片配置
# chunk_size = 2000
# chunk_overlap = 80

# Anki 配置
# anki_model_id = 16381234567
# anki_deck_id = 20593847561

# TTS 配置
# tts_enabled = false
```

## 断点续跑

Stage 3 每推理一个 chunk 自动记录进度。中断后重跑:

```bash
epub2anki run
# → 自动跳过已完成 chunk, 从断点继续
```

状态存于 `_run_state/`:

| 文件 | 作用 |
|---|---|
| `chunks.json` | 分片缓存 |
| `progress.json` | 已完成 chunk ID |
| `results.jsonl` | 逐行追加提取结果 |
| `failed_chunks.json` | 失败的 chunk (含错误信息) |

## 健康检查

`run` / `infer` / `tts` 启动前自动检查:

- EPUB 文件是否存在
- LLM API 是否可达 (连接 localhost:8080)
- TTS 依赖是否就绪 (kitten-tts 二进制 + 模型文件)

任一检查失败直接退出，不浪费算力。

## 文件结构

```
epub2anki/
├── autoanki/
│   ├── cli.py              # typer CLI + 5 阶段实现
│   └── tts.py              # TTS 音频生成模块
├── ass/                    # 原始素材 (epub / md)
├── _run_state/             # 断点状态 (自动创建)
│   ├── chunks.json         # 分片缓存
│   ├── progress.json       # 已完成 chunk ID
│   ├── results.jsonl       # 提取结果 (逐行追加)
│   └── failed_chunks.json  # 失败记录
├── _audio/                 # TTS 音频 (自动创建)
├── autoanki.toml.example   # 配置文件示例
├── pyproject.toml
└── README.md
```

## 依赖

- Python ≥ 3.13
- [llama.cpp server](https://github.com/ggml-org/llama.cpp) (本地, 端口 8080)
- `uv` — 包管理
- `genanki` — Anki 牌组打包
- `httpx` — HTTP 客户端
- `typer` — CLI 框架

## 安装

```bash
uv add genanki httpx typer
```

项目使用 `uv` 管理依赖，无需手动创建虚拟环境。
