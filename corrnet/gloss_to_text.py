"""
gloss_to_text.py — Phoenix gloss sequences → readable German sentences
-----------------------------------------------------------------------
Inspired by INDUCE-Lab's sign-to-text output layer.

Phoenix glosses are annotation tokens (REGEN, loc-SUED, __PU__, etc.).
This converts them to natural German text.

Usage:
    from gloss_to_text import glosses_to_german
    text = glosses_to_german(['HEUTE', 'REGEN', 'KOMMEN'])
    # → 'Heute kommt Regen.'
"""
from __future__ import annotations

import re

# ── Gloss → German surface form ───────────────────────────────────────────────
_MAP: dict[str, str] = {
    # Weather nouns
    "REGEN":          "Regen",
    "WIND":           "Wind",
    "SONNE":          "Sonne",
    "WOLKE":          "Wolken",
    "SCHNEE":         "Schnee",
    "EIS":            "Eis",
    "FROST":          "Frost",
    "NEBEL":          "Nebel",
    "GEWITTER":       "Gewitter",
    "STURM":          "Sturm",
    "HAGEL":          "Hagel",
    "SCHAUER":        "Schauer",
    # Temperature / adjectives
    "WARM":           "warm",
    "KALT":           "kalt",
    "KUEHL":          "kühl",
    "MILD":           "mild",
    "STARK":          "stark",
    "SCHWACH":        "schwach",
    "LEICHT":         "leicht",
    "TROCKEN":        "trocken",
    "NASS":           "nass",
    "SONNIG":         "sonnig",
    "BEWOELKT":       "bewölkt",
    "WECHSELHAFT":    "wechselhaft",
    "ANGENEHM":       "angenehm",
    "FRISCH":         "frisch",
    # Time
    "HEUTE":          "heute",
    "MORGEN":         "morgen",
    "GESTERN":        "gestern",
    "JETZT":          "jetzt",
    "SPAETER":        "später",
    "ABEND":          "abends",
    "NACHT":          "in der Nacht",
    "MITTAG":         "mittags",
    "FRUEH":          "früh",
    "AM-TAG":         "tagsüber",
    "ANFANG":         "zu Beginn",
    # Directions / regions
    "NORD":           "im Norden",
    "SUED":           "im Süden",
    "OST":            "im Osten",
    "WEST":           "im Westen",
    "DEUTSCHLAND":    "in Deutschland",
    "ALPEN":          "in den Alpen",
    "KUESTE":         "an der Küste",
    "MEER":           "am Meer",
    "REGION":         "in der Region",
    "SUED-OST":       "im Südosten",
    "SUED-WEST":      "im Südwesten",
    "NORD-OST":       "im Nordosten",
    "NORD-WEST":      "im Nordwesten",
    # Temperature values
    "GRAD":           "Grad",
    "TEMPERATUR":     "die Temperatur",
    # Verbs
    "KOMMEN":         "kommt",
    "BLEIBEN":        "bleibt",
    "STEIGEN":        "steigt",
    "FALLEN":         "fällt",
    "SINKEN":         "sinkt",
    "AUFHOEREN":      "hört auf",
    "AUFKOMMEN":      "kommt auf",
    "ABKUEHLEN":      "kühlt ab",
    "ERWARTET":       "wird erwartet",
    "AUFLOESEN":      "löst sich auf",
    "AUFHEITERN":     "heitert auf",
    "AUFKLAREN":      "klart auf",
    # Adverbs / modifiers
    "VEREINZELT":     "vereinzelt",
    "MOEGLICH":       "möglich",
    "WAHRSCHEINLICH": "wahrscheinlich",
    "AUCH":           "auch",
    "ABER":           "aber",
    "UND":            "und",
    "ODER":           "oder",
    "DANN":           "dann",
    "ERST":           "zunächst",
    "NOCH":           "noch",
    "NUR":            "nur",
    "SCHON":          "schon",
    "SEHR":           "sehr",
    "ETWAS":          "etwas",
    "VIEL":           "viel",
    "WENIG":          "wenig",
    "ALLGEMEIN":      "allgemein",
    "MEIST":          "meistens",
    "TEILS":          "teils",
    "MANCHMAL":       "manchmal",
    # Numbers — full 0-100 range present in Phoenix vocabulary
    "NULL":           "null",
    "EIN":            "ein",
    "ZWEI":           "zwei",
    "DREI":           "drei",
    "VIER":           "vier",
    "FUENF":          "fünf",
    "SECHS":          "sechs",
    "SIEBEN":         "sieben",
    "ACHT":           "acht",
    "NEUN":           "neun",
    "ZEHN":           "zehn",
    "ELF":            "elf",
    "ZWOELF":         "zwölf",
    "DREIZEHN":       "dreizehn",
    "VIERZEHN":       "vierzehn",
    "FUENFZEHN":      "fünfzehn",
    "SIEBZEHN":       "siebzehn",
    "ACHTZEHN":       "achtzehn",
    "NEUNZEHN":       "neunzehn",
    "ZWANZIG":        "zwanzig",
    "DREISSIG":       "dreißig",
    "VIERZIG":        "vierzig",
    "FUENFZIG":       "fünfzig",
    "SECHZIG":        "sechzig",
    "SIEBZIG":        "siebzig",
    "ACHTZIG":        "achtzig",
    "NEUNZIG":        "neunzig",
    "HUNDERT":        "hundert",
    # Arithmetic / temperature operators
    "MINUS":          "minus",
    "PLUS":           "plus",
    "MAXIMAL":        "maximal",
    "MINIMAL":        "minimal",
    "BIS":            "bis",
    "EINS":           "eins",
    "SECHSZEHN":      "sechzehn",
    # Weather — high-frequency missing entries
    "WETTER":         "Wetter",
    "FREUNDLICH":     "freundlich",
    "KLAR":           "klar",
    "HEISS":          "heiß",
    "MAESSIG":        "mäßig",
    "UEBERWIEGEND":   "überwiegend",
    "HAUPTSAECHLICH": "hauptsächlich",
    "TEILWEISE":      "teilweise",
    "DURCHGEHEND":    "durchgehend",
    "BESONDERS":      "besonders",
    "SCHNEIEN":       "schneit",
    "GLATT":          "glatt",
    "RUHIG":          "ruhig",
    "SCHOEN":         "schön",
    "GUT":            "gut",
    "BESSER":         "besser",
    "ORKAN":          "Orkan",
    "UNWETTER":       "Unwetter",
    "WARNUNG":        "Warnung",
    "VORSICHT":       "Vorsicht",
    "ENORM":          "enorm",
    # Pressure / sky
    "TIEF":           "Tief",
    "HOCH":           "Hoch",
    "DRUCK":          "Druck",
    "LUFT":           "Luft",
    "HIMMEL":         "Himmel",
    # Directions (compound)
    "NORDWEST":       "im Nordwesten",
    "NORDOST":        "im Nordosten",
    "SUEDWEST":       "im Südwesten",
    "SUEDOST":        "im Südosten",
    # Geography
    "BERG":           "am Berg",
    "MITTE":          "in der Mitte",
    "FLUSS":          "am Fluss",
    "LAND":           "im Land",
    "WALD":           "im Wald",
    "SEE":            "am See",
    "ZONE":           "Zone",
    "ORT":            "Ort",
    "EUROPA":         "Europa",
    "BAYERN":         "Bayern",
    "DEUTSCH":        "Deutschland",
    "UEBER":          "über",
    "STERN":          "Stern",
    # Days of week
    "MONTAG":         "Montag",
    "DIENSTAG":       "Dienstag",
    "MITTWOCH":       "Mittwoch",
    "DONNERSTAG":     "Donnerstag",
    "FREITAG":        "Freitag",
    "SAMSTAG":        "Samstag",
    "SONNTAG":        "Sonntag",
    "WOCHENENDE":     "Wochenende",
    # Time
    "TAG":            "am Tag",
    "NAECHSTE":       "nächste",
    "LANG":           "lang",
    "ZWISCHEN":       "zwischen",
    "IM-VERLAUF":     "im Verlauf",
    "IN-KOMMEND":     "kommend",
    "WIE-IMMER":      "wie immer",
    # Verbs / actions
    "WEHEN":          "weht",
    "VERSCHWINDEN":   "verschwindet",
    "HABEN":          "hat",
    "MACHEN":         "macht",
    "SEHEN":          "sieht",
    "ZEIGEN":         "zeigt",
    "WEITER":         "weiter",
    "KOENNEN":        "kann",
    # Particles / connectors
    "MEHR":           "mehr",
    "BISSCHEN":       "ein bisschen",
    "MEISTENS":       "meistens",
    "SONST":          "sonst",
    "WENN":           "wenn",
    "WIE":            "wie",
    "SO":             "so",
    "AB":             "ab",
    "NACH":           "nach",
    "MIT":            "mit",
    "VOR":            "vor",
    "MAL":            "mal",
    "DESHALB":        "deshalb",
    "WIEDER":         "wieder",
    "DABEI":          "dabei",
    "DAZU":           "dazu",
    "NEU":            "neu",
    "LIEB":           "liebe",
    "TEIL":           "Teil",
    "UNGEFAEHR":      "ungefähr",
    "ZWISCHEN":       "zwischen",
    "DRUCK":          "Druck",
    # TV-specific
    "ZUSCHAUER":      "Zuschauer",
}

# ── Patterns to drop ──────────────────────────────────────────────────────────
_DROP = [
    re.compile(r"^__\w+__$"),              # __PU__, __LEFTHAND__
    re.compile(r"^cl-", re.I),             # classifier glosses
    re.compile(r"^loc-", re.I),            # location-only tokens
    re.compile(r"^poss-", re.I),           # possessive
    re.compile(r"^neg-", re.I),            # negation-prefix glosses
    re.compile(r"^IX\d*$", re.I),          # pointing / deixis
    re.compile(r"^<.*>$"),                 # <unknown>
    re.compile(r"^[A-Z]-[A-Z]$"),          # fingerspelling single letters
    re.compile(r"^ZEIGEN-BILDSCHIRM$", re.I),  # screen-pointing gesture
    re.compile(r"^WIE-AUSSEHEN$", re.I),   # appearance-question gloss
]


def _token(gloss: str) -> str | None:
    """Normalise one gloss token → German word, or None to drop."""
    g = gloss.strip()
    upper = g.upper()

    # loc-* and poss-* must be checked BEFORE _DROP, which would swallow them.
    # Map to the base form's German word if known; otherwise drop silently.
    for prefix in ("LOC-", "POSS-"):
        if upper.startswith(prefix):
            base = upper[len(prefix):]
            return _MAP.get(base)  # None → drop if base not in _MAP

    for pat in _DROP:
        if pat.match(g):
            return None

    if upper in _MAP:
        return _MAP[upper]

    # Numeric: keep as-is
    if g.isdigit():
        return g

    # Fallback: treat as German noun (capitalise first letter)
    if g.isalpha():
        return g[0].upper() + g[1:].lower()

    return None


def glosses_to_german(glosses: list[str],
                       add_period: bool = True,
                       use_grammar_tool: bool = False) -> str:
    """
    Convert a Phoenix gloss sequence to a readable German sentence.

    Args:
        glosses:         e.g. ['HEUTE', 'REGEN', 'KOMMEN', '__PU__']
        add_period:      append '.' to the sentence
        use_grammar_tool: run LanguageTool for grammar correction
                          (requires: pip install language_tool_python)

    Returns:
        German sentence string, e.g. 'Heute kommt Regen.'
    """
    words = [_token(g) for g in glosses]
    words = [w for w in words if w]

    if not words:
        return ""

    sentence = " ".join(words)

    if use_grammar_tool:
        sentence = _grammar_correct(sentence)

    # Capitalise first word
    sentence = sentence[0].upper() + sentence[1:]

    if add_period and not sentence.endswith((".", "!", "?")):
        sentence += "."

    return sentence


def _grammar_correct(text: str) -> str:
    """Apply LanguageTool German grammar correction (optional dependency)."""
    try:
        import language_tool_python
        tool    = language_tool_python.LanguageTool("de")
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except ImportError:
        return text
    except Exception:
        return text


if __name__ == "__main__":
    tests = [
        ["HEUTE", "REGEN", "KOMMEN"],
        ["MORGEN", "NORD", "WIND", "STARK"],
        ["TEMPERATUR", "KALT", "BLEIBEN"],
        ["__PU__", "SONNE", "IX", "WARM"],
        ["ABEND", "BEWOELKT", "VEREINZELT", "REGEN"],
        ["loc-SUED", "TROCKEN", "ABER", "NORD", "WIND"],
    ]
    for g in tests:
        print(f"  {g}")
        print(f"  → \"{glosses_to_german(g)}\"\n")
