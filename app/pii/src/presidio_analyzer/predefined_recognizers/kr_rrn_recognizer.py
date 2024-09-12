from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class RRNRecognizer(PatternRecognizer):

    PATTERNS = [
        Pattern(
            "Resident Registration Number",
            r"\d{2}([0][1-9]|[1][0-2])([0][1-9]|[1-2]\d|[3][0-1])-?([1-4]{1})([0-9]{6})",
            0.1,
        ),
    ]
    CONTEXT = [
        "RRN",
        "주민등록번호",
        "Resident Registration Number",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "주민등록번호",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
