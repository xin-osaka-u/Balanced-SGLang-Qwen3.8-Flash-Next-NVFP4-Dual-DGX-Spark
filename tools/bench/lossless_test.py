"""Lossless check for MTP: Chinese / Japanese household documents, T=0, thinking off.

Usage: python lossless_test.py <base_url> <out_dir> <tag>
Writes <out_dir>/<tag>_<case>.txt and prints an accuracy score per case vs ground truth.
"""
import json
import re
import sys
import time
import urllib.request

URL = sys.argv[1] + "/v1/chat/completions"
OUT, TAG = sys.argv[2], sys.argv[3]

CASES = {
    "cn_lease": {
        "doc": """房屋租赁合同（节选）
出租方（甲方）：王建国，身份证尾号 4417
承租方（乙方）：李晓梅，身份证尾号 0826
房屋地址：上海市浦东新区锦绣路 1288 弄 7 号 1502 室，建筑面积 89.6 平方米。
租赁期限：自 2026 年 10 月 1 日起至 2028 年 9 月 30 日止，共 24 个月。
月租金：人民币 8,600 元，按季度支付，每季度首月 5 日前支付当季租金 25,800 元。
押金：人民币 17,200 元（两个月租金），合同期满且房屋无损坏时 7 个工作日内无息退还。
物业费由甲方承担；水、电、燃气、宽带费用由乙方承担。
第二年起月租金上调 3%，即 8,858 元。
乙方提前退租须提前 60 日书面通知，并支付一个月租金作为违约金。""",
        "ask": "从合同中提取 JSON：{\"lessor\",\"lessee\",\"address\",\"area_sqm\",\"start\",\"end\",\"months\","
               "\"monthly_rent\",\"quarterly_payment\",\"deposit\",\"deposit_refund_days\",\"rent_year2\","
               "\"notice_days\",\"penalty\"}。日期用 YYYY-MM-DD，金额用数值。只输出 JSON。",
        "truth": {"lessor": "王建国", "lessee": "李晓梅", "area_sqm": 89.6, "start": "2026-10-01",
                  "end": "2028-09-30", "months": 24, "monthly_rent": 8600, "quarterly_payment": 25800,
                  "deposit": 17200, "deposit_refund_days": 7, "rent_year2": 8858, "notice_days": 60,
                  "penalty": 8600},
    },
    "cn_bills": {
        "doc": """家庭 2026 年 9 月支出流水
09-02 盒马 买菜 268.40
09-03 国家电网 电费 405.10
09-05 物业 物业费 480.00
09-08 中国移动 话费宽带 199.00
09-12 京东 儿童书包 159.00
09-15 盒马 买菜 312.60
09-18 小区停车 月租 300.00
09-21 燃气公司 燃气费 61.30
09-24 医院 儿科门诊 186.50
09-27 盒马 买菜 289.90
09-28 自来水 水费 72.40""",
        "ask": "请把这些支出按类别汇总成 JSON：{\"groceries\",\"utilities\",\"housing\",\"telecom\",\"kids\","
               "\"medical\",\"total\"}。utilities 包含电、燃气、水；housing 包含物业和停车；kids 包含儿童用品；"
               "medical 包含医疗。金额保留两位小数，只输出 JSON。",
        "truth": {"groceries": 870.90, "utilities": 538.80, "housing": 780.00, "telecom": 199.00,
                  "kids": 159.00, "medical": 186.50, "total": 2734.20},
    },
    "jp_invoice": {
        "doc": """請求書
株式会社さくら設備 御中
請求日：2026年9月25日　お支払期限：2026年10月31日
請求番号：INV-2026-0917
品目　数量　単価　金額
エアコン分解洗浄　2台　12,000円　24,000円
室外機洗浄オプション　2台　3,500円　7,000円
防カビコート　2台　2,200円　4,400円
出張費　1式　3,000円　3,000円
小計　38,400円
消費税（10%）　3,840円
合計　42,240円
振込先：みずほ銀行 渋谷支店 普通 1234567""",
        "ask": "この請求書から JSON を抽出してください：{\"invoice_no\",\"issue_date\",\"due_date\","
               "\"items\":[{\"name\",\"qty\",\"unit_price\",\"amount\"}],\"subtotal\",\"tax\",\"total\"}。"
               "日付は YYYY-MM-DD、金額は数値。JSON のみ出力。",
        "truth": {"invoice_no": "INV-2026-0917", "issue_date": "2026-09-25", "due_date": "2026-10-31",
                  "subtotal": 38400, "tax": 3840, "total": 42240, "_items": {(2, 12000), (2, 3500), (2, 2200), (1, 3000)}},
    },
    "jp_school": {
        "doc": """保護者の皆様へ
秋の遠足のお知らせ（3年2組）
日時：2026年10月16日（金）　集合 8時20分 校庭　解散 14時30分 予定
行き先：高尾山（雨天の場合は10月20日（火）に延期）
持ち物：お弁当、水筒（500ml以上）、レインコート、敷物、ハンカチ、ティッシュ
費用：交通費 1,280円を10月9日（金）までに集金袋でご提出ください。
服装：動きやすい服装、運動靴。帽子を着用してください。
お菓子は300円以内とします。
連絡先：担当 田中先生（学校代表 042-555-0123）""",
        "ask": "请用中文把这份通知整理成 JSON：{\"date\",\"meet_time\",\"dismiss_time\",\"place\",\"rain_date\","
               "\"items\":[中文物品名],\"fee\",\"fee_deadline\",\"snack_limit\",\"teacher\",\"phone\"}。"
               "日期用 YYYY-MM-DD，时间用 HH:MM，金额用数值。只输出 JSON。",
        "truth": {"date": "2026-10-16", "meet_time": "08:20", "dismiss_time": "14:30", "rain_date": "2026-10-20",
                  "fee": 1280, "fee_deadline": "2026-10-09", "snack_limit": 300, "phone": "042-555-0123",
                  "_n_items": 6},
    },
    "jp_summary": {
        "doc": """重要事項説明（抜粋）
本物件は築18年の鉄筋コンクリート造マンションの7階部分です。2024年に大規模修繕工事を実施済みで、
次回は2036年頃を予定しています。管理費は月額14,500円、修繕積立金は月額18,200円で、2027年4月より
修繕積立金を月額21,000円に改定する旨が管理組合総会で決議されています。ペットは小型犬・猫1匹まで飼育可能ですが、
事前に管理組合への届出が必要です。駐車場は空きがなく、現在4名がキャンセル待ちです。本物件は土砂災害警戒区域外ですが、
ハザードマップ上、洪水時に0.5〜3.0mの浸水が想定される区域に含まれます。""",
        "ask": "请用中文分条总结这段重要事项说明，列出购房者最需要注意的 5 点，每点一句话。",
        "truth": None,
    },
}


def call(prompt):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], time.perf_counter() - t


def num(x):
    try:
        return float(str(x).replace(",", "").replace("元", "").replace("円", ""))
    except Exception:
        return None


def norm_time(v):
    m = re.match(r"^(\d{1,2}):(\d{2})", str(v))
    return f"{int(m[1]):02d}:{m[2]}" if m else str(v)


def score(case, text):
    truth = CASES[case]["truth"]
    if truth is None:
        return "free-form"
    t = re.sub(r"^```\w*|```\s*$", "", text.strip(), flags=re.M).strip()
    try:
        d = json.loads(t)
    except Exception:
        return "PARSE_FAIL"
    ok, total, bad = 0, 0, []
    for k, v in truth.items():
        if k == "_items":
            got = {(int(num(i.get("qty"))), int(num(i.get("unit_price")))) for i in d.get("items", [])}
            total += 1; ok += got == v
            if got != v: bad.append("items")
            continue
        if k == "_n_items":
            total += 1; ok += len(d.get("items", [])) == v
            if len(d.get("items", [])) != v: bad.append("items")
            continue
        total += 1
        g = d.get(k)
        if isinstance(v, (int, float)):
            hit = num(g) is not None and abs(num(g) - v) < 0.01
        elif k.endswith("time"):
            hit = norm_time(g) == v
        else:
            hit = str(g).strip() == v
        ok += hit
        if not hit: bad.append(f"{k}={g}")
    return f"{ok}/{total}" + (f" wrong: {', '.join(bad)}" if bad else "")


res = {}
for name, c in CASES.items():
    text, n, sec = call(c["doc"] + "\n\n" + c["ask"])
    open(f"{OUT}/{TAG}_{name}.txt", "w").write(text)
    res[name] = score(name, text)
    print(f"[{TAG}] {name:11s} tokens={n:4d} {n/sec:5.1f} tok/s(e2e)  accuracy={res[name]}")
json.dump(res, open(f"{OUT}/{TAG}_scores.json", "w"), ensure_ascii=False)
