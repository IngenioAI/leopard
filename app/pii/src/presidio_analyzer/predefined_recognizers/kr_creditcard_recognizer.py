from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class KRCreditCardRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            "Credit Card",
            r"([234569][0-9]{3}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4})",
            0.1,
        ),
    ]
    CONTEXT = [
        "신용카드번호",
    ]

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = "ko",
            supported_entity: str = "신용카드번호",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
