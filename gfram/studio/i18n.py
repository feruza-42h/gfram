"""Two interface languages: Uzbek (default) and Russian. Usage: L('o‘zbekcha', 'по-русски')."""

_lang = 'uz'


def set_language(lang: str):
    global _lang
    _lang = lang if lang in ('uz', 'ru') else 'uz'


def language() -> str:
    return _lang


def L(uz: str, ru: str) -> str:
    return ru if _lang == 'ru' else uz


def pct(x: float, digits: int = 2) -> str:
    """0.9967 -> '99,67%' (comma decimal, as in the dissertation)."""
    return f'{x * 100:.{digits}f}%'.replace('.', ',')


def num(x: float, digits: int = 3) -> str:
    return f'{x:.{digits}f}'.replace('.', ',')
