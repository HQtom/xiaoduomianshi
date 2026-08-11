"""
AI客服回复幻觉检测脚本

功能：
    读取 input.json 中的AI客服问答记录，调用 DeepSeek 大模型接口，
    依据五类幻觉定义对每条记录进行幻觉判定与分类，将结构化结果写入 output.json。

依赖环境变量（在 .env 文件中配置）：
    DEEPSEEK_API_KEY   DeepSeek 平台的 API Key（必填）
    DEEPSEEK_MODEL     使用的模型名称，默认 "deepseek-chat"
    MAX_TOKENS         单次请求最大生成 token 数，默认 1024

用法：
    pip install -r requirements.txt
    python eval_script.py
"""

import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
MAX_TOKENS = int(os.getenv("MAX_TOKENS") or 1024)

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.json")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.json")

# ---------------------------------------------------------------------------
# RTF (Role / Task / Format) 提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """# Role（角色）
你是一名资深的AI客服质检审核专家，专注于识别智能客服系统回复中的“幻觉”问题。你的判断必须严格基于给定的知识库内容，不得引入任何外部常识或主观猜测。

# Task（任务）
对于给定的一组【用户问题】【AI客服回复】【知识库内容】，请你完成以下两步判断：

第一步：判断该AI客服回复是否存在“幻觉”。
幻觉的定义：AI回答的内容与事实（即知识库内容）相违背，或超出了知识库/系统能力的范围，但AI在回复中并未表现出任何怀疑或不确定，而是以自信、确定的语气给出。简单来说就是“正经的胡说”。

第二步：如果存在幻觉，必须从以下五个类别中选择一个最匹配的类型（只能选一个最主要的类型），并给出具体理由。如果不存在幻觉，hallucination_type 填 null。

五类幻觉的定义与判断标准如下：

1. 虚构事实
   - 定义：AI在知识库完全没有相关参照信息的情况下，凭空捏造出具体、自信的答案（无中生有）。这类回答明显是基于模型预训练知识或臆测，而非知识库内容。
   - 典型例子：用户询问投诉升级处理进度，知识库中并无该系统功能，但AI却自信地回复“已升级工单，2小时内联系您”；用户询问学生优惠，知识库中根本没有该政策，AI却编造出具体的优惠比例和操作入口。

2. 参数和事实错误
   - 定义：知识库中存在明确的相关信息（如产品参数、规格、政策条款等），但AI的回复内容与知识库记载的事实相违背，或者AI没有正确理解知识库信息，篡改了具体的数值、型号、接口类型、政策细节等。
   - 典型例子：知识库标注充电头接口为USB-A，AI却回复为Type-C接口；知识库提示某成分孕妇需谨慎使用，AI却理解错误，直接说明可以放心使用（该例子也可能同时涉及安全风险）。

3. 政策规则错误
   - 定义：AI的回复内容不完整，只回答了知识库信息的一部分，遗漏了知识库中同样重要甚至更关键的限制条件、例外情况或补充说明，导致回复本身的规则表述出现偏差或不准确。
   - 典型例子：知识库注明“质量问题30天内可退换，非质量问题运费买家承担”，AI回复却只说“全品类支持30天无理由退货，运费我们承担”，遗漏了区分条件，规则表述整体错误。

4. 安全风险
   - 定义：知识库中包含与用户健康、财产或人身安全相关的警示/风险信息（如成分禁忌、使用禁忌、安全须知等），AI的回复没有清晰、准确地识别并传达这些风险提示，可能对用户造成实际伤害或损失。
   - 典型例子：知识库中标注某产品含有需谨慎使用的成分并建议孕妇咨询医生，AI却直接答复孕妇可以放心使用，未能识别并传达潜在的安全风险。

5. 未采纳文档
   - 定义：知识库中明确包含与用户问题直接相关、且理应被参考并体现在回复中的信息，但AI的回复完全没有采纳或提及这部分内容，导致给出的建议或结论与知识库的完整信息不符。
   - 典型例子：知识库中记录“约30%用户反馈偏大半码”，AI回复却直接说“尺码标准，不偏大不偏小”，完全未采纳知识库中已有的、与问题直接相关的用户反馈信息。

判断顺序建议：
- 若知识库中完全没有相关信息，AI却给出确定性答案 -> 优先判定为“虚构事实”。
- 若知识库中有相关信息，但AI给出的具体数值/条款与知识库矛盾 -> 判定为“参数和事实错误”。
- 若知识库中有相关信息，AI只采纳了部分、遗漏了限制条件导致整体规则表述错误 -> 判定为“政策规则错误”。
- 若涉及健康、安全、资金安全等风险提示未被清晰传达 -> 判定为“安全风险”。
- 若知识库中有与问题直接相关的信息，但AI的回复完全没有参考、体现该信息 -> 判定为“未采纳文档”。
- 若一条回复可能同时符合多个类别，选择最贴切、最本质的一个类别。

# Format（输出格式）
你必须只输出一个合法的JSON对象，不要输出任何其他说明文字、前后缀、Markdown代码块标记。JSON对象的字段如下：
{
  "id": "输入数据中的id，原样返回",
  "is_hallucination": true 或 false,
  "hallucination_type": "五类之一的类型名称字符串，如果is_hallucination为false则为null",
  "reason": "简明扼要说明判断依据，需要具体指出AI回复中的哪部分内容与知识库的哪部分内容存在冲突或遗漏，30-80字左右"
}
"""

USER_PROMPT_TEMPLATE = """请对以下这条AI客服问答记录进行幻觉判定：

【id】：{id}
【用户问题】：{user_question}
【AI客服回复】：{system_reply}
【知识库内容】：{knowledge_base}

请严格按照系统提示中的Format要求，只输出一个JSON对象。"""


def build_messages(item: dict) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                id=item.get("id"),
                user_question=item.get("user_question"),
                system_reply=item.get("system_reply"),
                knowledge_base=item.get("knowledge_base"),
            ),
        },
    ]


def extract_json(text: str) -> dict:
    """从模型返回文本中提取JSON对象，兼容可能出现的```json代码块包裹。"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)


def call_deepseek(item: dict, retries: int = 3, timeout: int = 60) -> dict:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": build_messages(item),
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = extract_json(content)
            result.setdefault("id", item.get("id"))
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[警告] id={item.get('id')} 第{attempt}次请求失败：{exc}", file=sys.stderr)
            time.sleep(2 * attempt)

    return {
        "id": item.get("id"),
        "is_hallucination": None,
        "hallucination_type": None,
        "reason": f"调用DeepSeek接口失败：{last_error}",
    }


def main() -> None:
    if not DEEPSEEK_API_KEY:
        print("[错误] 未检测到 DEEPSEEK_API_KEY，请在 .env 文件中配置后重试。", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{total}] 正在分析 id={item.get('id')} ...")
        result = call_deepseek(item)
        results.append(result)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成，结果已保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
