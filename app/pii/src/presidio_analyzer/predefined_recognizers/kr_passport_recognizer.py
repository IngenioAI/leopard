from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class KRPassportRecognizer(PatternRecognizer):

    PATTERNS = [
        Pattern(
            "Passport",
            r"[a-zA-Z]{1}[0-9a-zA-Z]{1}[0-9]{7}",
            0.1,
        ),
    ]
    CONTEXT = [
        "여권번호",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "여권번호",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )