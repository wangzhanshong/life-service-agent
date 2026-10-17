from src.tools.mcp_tools import (
    _city_distance, _food_recommend, _get_coordinates, _mbti_calculate, _mbti_questions, _mbti_query, _query_weather
)


class TestWeather:
    def test_query_beijing(self):
        r = _query_weather("北京")
        assert r["found"] is True
        assert r["temperature"] == 22

    def test_query_unknown(self):
        r = _query_weather("火星")
        assert r["found"] is False

    def test_fuzzy_match(self):
        r = _query_weather("成都")
        assert r["found"] is True


class TestGeo:
    def test_coordinates(self):
        r = _get_coordinates("北京")
        assert r["found"] is True
        assert r["latitude"] == 39.9

    def test_distance(self):
        r = _city_distance("北京", "上海")
        assert 1000 < r["distance_km"] < 1200  # ~1067 km


class TestMBTI:
    def test_questions(self):
        r = _mbti_questions()
        assert r["total"] == 12

    def test_calculate_intj(self):
        answers = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # I, N, T, J
        r = _mbti_calculate(answers)
        assert r["type"] == "INTJ"

    def test_calculate_esfp(self):
        answers = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # E, S, F, P
        r = _mbti_calculate(answers)
        assert r["type"] == "ESFP"

    def test_query_type(self):
        r = _mbti_query("INTJ")
        assert "建筑师" in r["description"]

    def test_invalid_answers(self):
        r = _mbti_calculate([0, 0])
        assert "error" in r


class TestFood:
    def test_recommend(self):
        r = _food_recommend("成都", "辣")
        assert r["count"] > 0

    def test_unknown_city(self):
        r = _food_recommend("火星")
        assert r.get("found") is False
