import json

from src.tools.compute_tools import _calculate, _date_calc, _text_stats, _unit_convert


class TestCalculator:
    def test_simple_arithmetic(self):
        r = _calculate("2 + 3 * 4")
        assert r["result"] == 14

    def test_sqrt(self):
        r = _calculate("sqrt(144)")
        assert r["result"] == 12

    def test_trig(self):
        import math
        r = _calculate("sin(pi/2)")
        assert abs(r["result"] - 1.0) < 0.01

    def test_error_expression(self):
        r = _calculate("1/0")
        assert "error" in r

    def test_safe_no_eval(self):
        r = _calculate("__import__('os').system('ls')")
        assert "error" in r


class TestUnitConvert:
    def test_km_to_m(self):
        r = _unit_convert(1, "km", "m")
        assert r["result"] == 1000

    def test_celsius_to_fahrenheit(self):
        r = _unit_convert(0, "celsius", "fahrenheit")
        assert r["result"] == 32

    def test_unknown(self):
        r = _unit_convert(1, "lightyear", "cm")
        assert "error" in r


class TestTextStats:
    def test_stats(self):
        r = _text_stats("Hello World")
        assert r["characters"] == 11
        assert r["words"] == 2
        assert r["lines"] == 1

    def test_chinese(self):
        r = _text_stats("你好世界")
        assert r["chinese_characters"] == 4


class TestDateCalc:
    def test_today(self):
        r = _date_calc("today")
        assert "date" in r
        assert "weekday" in r

    def test_add_days(self):
        r = _date_calc("add_days", "2026-01-01", 5)
        assert r["date"] == "2026-01-06"

    def test_diff_days(self):
        r = _date_calc("diff_days", "2026-01-01, 2026-01-10")
        assert r["diff_days"] == 9
