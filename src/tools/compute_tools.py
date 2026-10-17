import ast
import math
import operator
from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.tools.base import ToolDefinition

# Safe math operations (AST-based, no eval)
_SAFE_OPS: Dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "int": int, "float": float, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor,
    "degrees": math.degrees, "radians": math.radians, "factorial": math.factorial,
}

_UNIT_CONVERSIONS = {
    ("km", "m"): 1000, ("m", "km"): 0.001, ("cm", "m"): 0.01, ("m", "cm"): 100,
    ("kg", "g"): 1000, ("g", "kg"): 0.001, ("h", "s"): 3600, ("min", "s"): 60,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Unsupported function: {func_name}")
        return _SAFE_FUNCS[func_name](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS and not callable(_SAFE_FUNCS[node.id]):
            return _SAFE_FUNCS[node.id]
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def _calculate(expression: str) -> dict:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return {"expression": expression, "result": _eval_node(tree.body)}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}


def _unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    conv = _UNIT_CONVERSIONS.get((from_unit.lower(), to_unit.lower()))
    if conv is None:
        return {"error": f"No conversion from {from_unit} to {to_unit}"}
    result = conv(value) if callable(conv) else value * conv
    return {"value": value, "from": from_unit, "to": to_unit, "result": result}


def _text_stats(text: str) -> dict:
    return {
        "characters": len(text),
        "characters_no_space": len(text.replace(" ", "").replace("\n", "").replace("\t", "")),
        "words": len(text.split()),
        "lines": text.count("\n") + 1 if text else 0,
        "chinese_characters": sum(1 for c in text if "一" <= c <= "鿿"),
    }


def _date_calc(operation: str, date_str: str = "", value: int = 0) -> dict:
    try:
        if operation == "today":
            return {"date": datetime.now().strftime("%Y-%m-%d"), "weekday": datetime.now().strftime("%A")}
        if operation == "weekday":
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return {"date": date_str, "weekday": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][dt.weekday()]}
        if operation == "add_days":
            dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=value)
            return {"date": dt.strftime("%Y-%m-%d")}
        if operation == "diff_days":
            d1, d2 = [datetime.strptime(d.strip(), "%Y-%m-%d") for d in date_str.split(",")]
            return {"date1": d1.strftime("%Y-%m-%d"), "date2": d2.strftime("%Y-%m-%d"), "diff_days": (d2 - d1).days}
        return {"error": f"Unknown operation: {operation}"}
    except Exception as e:
        return {"error": str(e)}


def create_compute_tools() -> List[ToolDefinition]:
    return [
        ToolDefinition(name="calculate", description="Safely evaluate a math expression (supports +-*/^, sqrt, sin, cos, log, etc.).",
                       parameters={"type": "object", "properties": {"expression": {"type": "string"}},
                                   "required": ["expression"]}, execute=_calculate, category="compute"),
        ToolDefinition(name="unit_convert", description="Convert between units (km<->m, kg<->g, celsius<->fahrenheit, etc.).",
                       parameters={"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
                                   "required": ["value", "from_unit", "to_unit"]}, execute=_unit_convert, category="compute"),
        ToolDefinition(name="text_stats", description="Compute text statistics (chars, words, lines, Chinese chars).",
                       parameters={"type": "object", "properties": {"text": {"type": "string"}},
                                   "required": ["text"]}, execute=_text_stats, category="compute"),
        ToolDefinition(name="date_calc", description="Date calculations: today, weekday, add_days, diff_days.",
                       parameters={"type": "object", "properties": {
                           "operation": {"type": "string", "enum": ["today","weekday","add_days","diff_days"]},
                           "date_str": {"type": "string"}, "value": {"type": "integer"}},
                                   "required": ["operation"]}, execute=_date_calc, category="compute"),
    ]
