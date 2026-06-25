# autoanki — Edge AI 词汇预热牌组 Pipeline


## 架构总览：五阶段闭环状态机

```
+------------------------+      +-------------------------+      +--------------------------+      +---------------------------+      +------------------------+
|  Stage 1: 数据预处理   | ---> |   Stage 2: 窗口流式分片  | ---> |   Stage 3: 鲁棒性推理    | ---> |   Stage 4: TTS 音频生成  | ---> |  Stage 5: 结构化打包    |
| (markitdown 导出 md)   |      |  (按 1200 字符重组文本) |      | (对齐 local 8080 接口)  |      | (kitten-tts, 可选)      |      | (genanki 自动封装apkg)  |
+------------------------+      +-------------------------+      +--------------------------+      +---------------------------+      +------------------------+
```


## Pipeline 完整工作流

```
终端 1:
  llama serve --no-mmproj -hf openbmb/MiniCPM5-1B-GGUF -ngl 99
  长期驻守, 监听 127.0.0.1:8080, RTX 3060 全天候推理

终端 2:
  cd /home/hashira/code/ankiword/autoanki

  # 全量跑 (无 TTS):
  uv run python pipeline.py   # preprocess → chunk → infer → package
  # 产出: wimpy_kid_minicpm.apkg

  # 全量跑 (含 TTS):
  uv run autoanki run --tts    # preprocess → chunk → infer → tts → package

  # 分步跑 (适合断点续跑、分开调试):
  uv run autoanki chunk        # preprocess + chunk
  uv run autoanki infer        # 推理 (LLM 高算力)
  uv run autoanki tts          # TTS 音频 (可选, 高算力, 可单独跑)
  uv run autoanki package      # 打包 (无 TTS)

Anki:
  双击导入 .apkg -> 同步手机 -> 20分钟刷完 Greq 黑话 -> 打开 epub 无痛阅读
```

---

## 缺失依赖安装

```bash
uv add genanki httpx
```

当前环境: Python 3.14, uv 已安装, 项目无 venv。`uv` 自动创建 venv + 安装依赖。

# 调试代码:
# rm -rf _run_state _audio _raw
# uv run autoanki run --tts --chunk-size 1
