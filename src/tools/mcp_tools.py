import json
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

from src.tools.base import ToolDefinition

# ======================== Weather ========================
_CITY_WEATHER = {
    "北京": {"temp": 22, "humidity": 45, "condition": "晴", "wind": "北风3级", "aqi": 65},
    "上海": {"temp": 25, "humidity": 70, "condition": "多云", "wind": "东南风4级", "aqi": 48},
    "广州": {"temp": 30, "humidity": 85, "condition": "雷阵雨", "wind": "南风2级", "aqi": 35},
    "深圳": {"temp": 29, "humidity": 80, "condition": "阵雨", "wind": "西南风3级", "aqi": 40},
    "杭州": {"temp": 24, "humidity": 65, "condition": "阴", "wind": "东风2级", "aqi": 55},
    "成都": {"temp": 23, "humidity": 60, "condition": "多云转晴", "wind": "无持续风向", "aqi": 72},
    "武汉": {"temp": 27, "humidity": 55, "condition": "晴转多云", "wind": "南风3级", "aqi": 68},
    "重庆": {"temp": 26, "humidity": 75, "condition": "阴转小雨", "wind": "无持续风向", "aqi": 80},
    "西安": {"temp": 21, "humidity": 40, "condition": "晴", "wind": "东北风2级", "aqi": 90},
    "哈尔滨": {"temp": 12, "humidity": 35, "condition": "晴", "wind": "西风4级", "aqi": 30},
    "昆明": {"temp": 20, "humidity": 50, "condition": "晴", "wind": "西南风2级", "aqi": 25},
    "三亚": {"temp": 32, "humidity": 78, "condition": "晴", "wind": "海风3级", "aqi": 20},
    "拉萨": {"temp": 15, "humidity": 25, "condition": "晴", "wind": "无持续风向", "aqi": 18},
    "乌鲁木齐": {"temp": 18, "humidity": 30, "condition": "晴", "wind": "北风3级", "aqi": 45},
}


def _query_weather(city: str) -> dict:
    weather = _CITY_WEATHER.get(city)
    if weather is None:
        for key in _CITY_WEATHER:
            if city in key or key in city:
                weather, city = _CITY_WEATHER[key], key
                break
    if weather is None:
        return {"city": city, "found": False, "message": f"No weather data for '{city}'. Supported: {list(_CITY_WEATHER.keys())}"}
    suggestion = "Good for outdoor activities!" if weather["condition"] in ("晴", "晴转多云", "多云转晴") else \
        "Bring an umbrella!" if "雨" in weather["condition"] else "Moderate weather."
    return {"city": city, "found": True, "temperature": weather["temp"], "humidity": weather["humidity"],
            "condition": weather["condition"], "wind": weather["wind"], "aqi": weather["aqi"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "suggestion": suggestion}


# ======================== Geo ========================
_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "北京": (39.9, 116.4), "上海": (31.2, 121.5), "广州": (23.1, 113.3), "深圳": (22.5, 114.1),
    "杭州": (30.3, 120.2), "成都": (30.6, 104.1), "武汉": (30.6, 114.3), "重庆": (29.4, 106.9),
    "西安": (34.3, 108.9), "哈尔滨": (45.8, 126.5), "昆明": (25.0, 102.7), "三亚": (18.3, 109.5),
    "拉萨": (29.7, 91.1), "乌鲁木齐": (43.8, 87.6), "南京": (32.1, 118.8), "天津": (39.3, 117.4),
    "长沙": (28.2, 112.9), "郑州": (34.7, 113.6), "东京": (35.7, 139.7), "纽约": (40.7, -74.0),
    "伦敦": (51.5, -0.1), "巴黎": (48.9, 2.4), "悉尼": (-33.9, 151.2),
}


def _get_coordinates(city: str) -> dict:
    coords = _CITY_COORDS.get(city)
    if coords is None:
        for key in _CITY_COORDS:
            if city in key or key in city:
                coords, city = _CITY_COORDS[key], key
                break
    if coords is None:
        return {"city": city, "found": False}
    return {"city": city, "latitude": coords[0], "longitude": coords[1], "found": True}


def _city_distance(city1: str, city2: str) -> dict:
    c1, c2 = _CITY_COORDS.get(city1), _CITY_COORDS.get(city2)
    if not c1: return {"error": f"City not found: {city1}"}
    if not c2: return {"error": f"City not found: {city2}"}
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return {"city1": city1, "city2": city2, "distance_km": round(6371 * 2 * math.asin(math.sqrt(a)), 2)}


# ======================== MBTI ========================
_MBTI_QUESTIONS = [
    {"id": "EI-1", "dimension": "EI", "question": "社交聚会后，你通常感觉精力充沛还是疲惫？", "options": ["精力充沛 (E)", "疲惫 (I)"]},
    {"id": "EI-2", "dimension": "EI", "question": "你更喜欢在团队中工作还是独自工作？", "options": ["团队合作 (E)", "独自工作 (I)"]},
    {"id": "EI-3", "dimension": "EI", "question": "你倾向于先行动再思考，还是先思考再行动？", "options": ["先行动 (E)", "先思考 (I)"]},
    {"id": "SN-1", "dimension": "SN", "question": "你更关注具体事实还是整体模式和可能性？", "options": ["具体事实 (S)", "整体模式 (N)"]},
    {"id": "SN-2", "dimension": "SN", "question": "你更信任过往经验还是直觉预感？", "options": ["过往经验 (S)", "直觉预感 (N)"]},
    {"id": "SN-3", "dimension": "SN", "question": "解决问题时更依赖已验证的方法还是创新思路？", "options": ["已验证方法 (S)", "创新思路 (N)"]},
    {"id": "TF-1", "dimension": "TF", "question": "做决策时更依赖逻辑分析还是个人价值观？", "options": ["逻辑分析 (T)", "个人价值观 (F)"]},
    {"id": "TF-2", "dimension": "TF", "question": "朋友倾诉时，你倾向于提供解决方案还是情感支持？", "options": ["解决方案 (T)", "情感支持 (F)"]},
    {"id": "TF-3", "dimension": "TF", "question": "批评别人时更关注客观对错还是对方感受？", "options": ["客观对错 (T)", "对方感受 (F)"]},
    {"id": "JP-1", "dimension": "JP", "question": "你更喜欢按计划行事还是随性灵活？", "options": ["按计划 (J)", "随性灵活 (P)"]},
    {"id": "JP-2", "dimension": "JP", "question": "截止日期让你感到有动力还是压力？", "options": ["有动力 (J)", "有压力 (P)"]},
    {"id": "JP-3", "dimension": "JP", "question": "倾向于把工作提前完成还是拖到最后？", "options": ["提前完成 (J)", "最后关头 (P)"]},
]

_MBTI_DESC = {
    "ISTJ": "物流师 — 务实、严谨、可靠，重视传统和秩序。",
    "ISFJ": "守卫者 — 温暖、细心、奉献，重视和谐与稳定。",
    "INFJ": "提倡者 — 理想主义、有洞察力，追求深层意义。",
    "INTJ": "建筑师 — 战略思维、独立、高标准，善于长远规划。",
    "ISTP": "鉴赏家 — 冷静、务实、灵活，喜欢动手实践。",
    "ISFP": "探险家 — 温和、艺术感、活在当下。",
    "INFP": "调停者 — 理想主义、善解人意，忠于价值观。",
    "INTP": "逻辑学家 — 分析型、好奇、创新，享受理论构建。",
    "ESTP": "企业家 — 精力充沛、灵活务实，喜欢冒险。",
    "ESFP": "表演者 — 热情、友好、乐于分享快乐。",
    "ENFP": "竞选者 — 热情、创意、善于社交。",
    "ENTP": "辩论家 — 机智、好奇、享受智力挑战。",
    "ESTJ": "总经理 — 高效、果断、组织力强。",
    "ESFJ": "执政官 — 热心、负责、善于关怀他人。",
    "ENFJ": "主人公 — 富有魅力、善解人意，激励他人。",
    "ENTJ": "指挥官 — 果断、自信、富有远见。",
}


def _mbti_questions() -> dict:
    return {"total": len(_MBTI_QUESTIONS), "questions": _MBTI_QUESTIONS}


def _mbti_calculate(answers: List[int]) -> dict:
    if len(answers) != 12:
        return {"error": f"Need exactly 12 answers, got {len(answers)}"}
    try:
        answers = [int(a) for a in answers]
    except (ValueError, TypeError):
        return {"error": "All answers must be integers (0 or 1)"}

    ei_score = sum(answers[0:3])
    sn_score = sum(answers[3:6])
    tf_score = sum(answers[6:9])
    jp_score = sum(answers[9:12])

    mbti = ""
    mbti += "E" if ei_score < 2 else "I"
    mbti += "S" if sn_score < 2 else "N"
    mbti += "T" if tf_score < 2 else "F"
    mbti += "J" if jp_score < 2 else "P"

    return {"type": mbti, "description": _MBTI_DESC.get(mbti, "Unknown"),
            "dimension_scores": {
                "EI": {"E": 3 - ei_score, "I": ei_score},
                "SN": {"S": 3 - sn_score, "N": sn_score},
                "TF": {"T": 3 - tf_score, "F": tf_score},
                "JP": {"J": 3 - jp_score, "P": jp_score},
            }}


def _mbti_query(mbti_type: str) -> dict:
    t = mbti_type.upper().strip()
    if t not in _MBTI_DESC:
        return {"error": f"Unknown MBTI type: {t}", "valid_types": list(_MBTI_DESC.keys())}
    return {"type": t, "description": _MBTI_DESC[t]}


# ======================== Food ========================
_REGIONAL_FOODS: Dict[str, List[Dict]] = {
    "北京": [{"name": "北京烤鸭", "cuisine": "京菜", "price_range": "150-300元/人", "description": "外酥里嫩", "popularity": 5},
             {"name": "炸酱面", "cuisine": "京菜", "price_range": "20-40元/人", "description": "酱香浓郁", "popularity": 5},
             {"name": "涮羊肉", "cuisine": "清真", "price_range": "100-200元/人", "description": "铜锅涮肉", "popularity": 4}],
    "上海": [{"name": "小笼包", "cuisine": "沪菜", "price_range": "30-60元/人", "description": "皮薄汤多", "popularity": 5},
             {"name": "生煎包", "cuisine": "沪菜", "price_range": "20-50元/人", "description": "金黄酥脆", "popularity": 4},
             {"name": "本帮红烧肉", "cuisine": "沪菜", "price_range": "80-150元/人", "description": "浓油赤酱", "popularity": 5}],
    "广州": [{"name": "广式早茶", "cuisine": "粤菜", "price_range": "50-150元/人", "description": "一盅两件", "popularity": 5},
             {"name": "白切鸡", "cuisine": "粤菜", "price_range": "60-100元/人", "description": "原汁原味", "popularity": 5}],
    "成都": [{"name": "麻辣火锅", "cuisine": "川菜", "price_range": "80-200元/人", "description": "麻辣鲜香", "popularity": 5},
             {"name": "麻婆豆腐", "cuisine": "川菜", "price_range": "25-50元/人", "description": "嫩滑入味", "popularity": 5},
             {"name": "担担面", "cuisine": "川菜", "price_range": "15-30元/人", "description": "肉臊酥香", "popularity": 4}],
    "重庆": [{"name": "重庆火锅", "cuisine": "川渝", "price_range": "70-180元/人", "description": "九宫格老火锅", "popularity": 5},
             {"name": "重庆小面", "cuisine": "川渝", "price_range": "10-25元/人", "description": "麻辣鲜香", "popularity": 5}],
    "杭州": [{"name": "西湖醋鱼", "cuisine": "浙菜", "price_range": "80-150元/人", "description": "酸甜嫩滑", "popularity": 5},
             {"name": "东坡肉", "cuisine": "浙菜", "price_range": "60-100元/人", "description": "肥而不腻", "popularity": 5}],
    "西安": [{"name": "肉夹馍", "cuisine": "陕菜", "price_range": "10-25元/人", "description": "外酥里嫩", "popularity": 5},
             {"name": "羊肉泡馍", "cuisine": "陕菜", "price_range": "30-60元/人", "description": "羊肉鲜香", "popularity": 5}],
}
_TASTE_MAP = {"辣": ["成都", "重庆"], "甜": ["上海", "杭州", "广州"], "清淡": ["广州", "杭州", "昆明"]}


def _food_recommend(city: str, taste: str = "", budget: str = "") -> dict:
    foods = _REGIONAL_FOODS.get(city)
    if foods is None:
        for key in _REGIONAL_FOODS:
            if city in key or key in city:
                foods, city = _REGIONAL_FOODS[key], key
                break
    if foods is None:
        return {"city": city, "found": False, "message": f"No food data for '{city}'"}
    if taste:
        taste_kw = {"辣": ["辣", "麻"], "甜": ["甜", "糖"], "清淡": ["清", "鲜", "淡"], "麻": ["麻", "辣"]}.get(taste, [taste])
        foods = [f for f in foods if any(k in f["description"]+f["cuisine"]+f["name"] for k in taste_kw)] or foods
    if budget:
        bm = {"低": (0, 50), "中": (50, 150), "高": (150, 9999)}.get(budget)
        if bm:
            filtered = []
            for f in foods:
                try:
                    upper = int(f["price_range"].split("-")[1].replace("元/人", ""))
                    if bm[0] <= upper <= bm[1] or (budget == "高" and upper >= 150):
                        filtered.append(f)
                except Exception:
                    filtered.append(f)
            foods = filtered or foods
    return {"city": city, "taste": taste or "any", "budget": budget or "any",
            "count": len(foods), "recommendations": sorted(foods, key=lambda x: x["popularity"], reverse=True)}


def _get_current_time() -> dict:
    """获取当前时间"""
    from datetime import datetime
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def create_mcp_tools() -> List[ToolDefinition]:
    return [
        ToolDefinition(name="get_weather", description="Query weather by city name.",
                       parameters={"type": "object", "properties": {"city": {"type": "string"}},
                                   "required": ["city"]}, execute=_query_weather, category="mcp"),
        ToolDefinition(name="get_coordinates", description="Get lat/lng for a city.",
                       parameters={"type": "object", "properties": {"city": {"type": "string"}},
                                   "required": ["city"]}, execute=_get_coordinates, category="mcp"),
        ToolDefinition(name="city_distance", description="Calculate distance between two cities (Haversine).",
                       parameters={"type": "object", "properties": {"city1": {"type": "string"}, "city2": {"type": "string"}},
                                   "required": ["city1", "city2"]}, execute=_city_distance, category="mcp"),
        ToolDefinition(name="mbti_questions", description="Get all 12 MBTI test questions.",
                       parameters={"type": "object", "properties": {}, "required": []},
                       execute=_mbti_questions, category="mcp"),
        ToolDefinition(name="mbti_calculate", description="Calculate MBTI type from 12 answers (0 or 1 each).",
                       parameters={"type": "object", "properties": {"answers": {"type": "array", "items": {"type": "integer"}, "minItems": 12, "maxItems": 12}},
                                   "required": ["answers"]}, execute=_mbti_calculate, category="mcp"),
        ToolDefinition(name="mbti_query", description="Look up MBTI type description.",
                       parameters={"type": "object", "properties": {"mbti_type": {"type": "string"}},
                                   "required": ["mbti_type"]}, execute=_mbti_query, category="mcp"),
        ToolDefinition(name="food_recommend", description="Recommend local food by city, taste, and budget.",
                       parameters={"type": "object", "properties": {
                           "city": {"type": "string"}, "taste": {"type": "string"}, "budget": {"type": "string"}},
                                   "required": ["city"]}, execute=_food_recommend, category="mcp"),
        ToolDefinition(name="get_current_time", description="获取当前日期和时间，返回日期、时间、星期几。用于用户询问现在几点、今天几号、今天星期几等时间相关问题。",
                       parameters={"type": "object", "properties": {}, "required": []},
                       execute=_get_current_time, category="mcp"),
    ]
