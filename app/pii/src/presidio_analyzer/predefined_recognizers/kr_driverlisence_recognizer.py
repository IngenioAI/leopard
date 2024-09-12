from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class KRDriverLicenseRecognizer(PatternRecognizer):

    PATTERNS = [
        Pattern(
            "Driver License",
            r"[1-2][1-8]-\d{2}-\d{6}-\d{2}",
            0.1,
        ),
    ]
    CONTEXT = [
        "운전면허번호",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "운전면허번호",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )