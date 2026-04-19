# magic-generator

模块化 LLM 生成库，专为 RAG 应用设计。

## 功能特性

| 策略 | 类 | 说明 |
|------|-----|------|
| OpenAI | `OpenAIGenerator` | GPT-4o、GPT-4o-mini 等 |
| Anthropic | `AnthropicGenerator` | Claude Sonnet/Haiku/Opus 等 |

## 安装

```bash
pip install openai anthropic
```

## 快速开始

```python
from magic_generator.core import GenerationPrompt
from magic_generator.strategies import OpenAIGenerator

generator = OpenAIGenerator(model="gpt-4o-mini")
prompt = GenerationPrompt(
    user_prompt="公司营收是多少？",
    context=["A公司2024年营收为100亿元。"]
)
result = generator.generate(prompt)
print(result.response)
```

## 模块结构

```
magic_generator/
    __init__.py          # 统一导出
    run.py               # CLI 入口
    core/                # GenerationPrompt, GenerationResult, BaseGenerator
    strategies/
        openai/          # OpenAI 生成器
        anthropic/       # Anthropic Claude 生成器
    utils/
```

## CLI 用法

```bash
python -m magic_generator.run \
    --query "公司营收是多少？" \
    --context "A公司2024年营收为100亿元。" \
    --strategy openai \
    --model gpt-4o-mini
```

## 设计原则

1. **GenerationPrompt** 支持 `system_prompt`、`user_prompt`、`context`（检索结果列表）
2. **自动上下文构建**：当 `context` 非空时，自动将上下文与用户问题拼接
3. **统一返回**：返回 `GenerationResult`，包含 `response`、`prompt`、`metadata`（含 token 用量）
