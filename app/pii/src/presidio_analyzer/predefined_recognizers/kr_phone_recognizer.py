from typing import Optional, List

from presidio_analyzer import Pattern, PatternRecognizer


class KRPhoneRecognizer(PatternRecognizer):

    PATTERNS = [
        Pattern(
            "Phone Number",
            r"01([0|1|6|7|8|9])-?([0-9]{3,4})-?([0-9]{4})",
            0.1,
        ),
        Pattern(
            "Landline Number",
            r"(0(2|3[1-3]|4[1-4]|5[1-5]|6[1-4]))-(\d{3,4})-(\d{4})",
            0.1,
        ),
    ]
    CONTEXT = [
        "휴대폰번호",
        "폰번호",
        "전화번호",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "전화번호",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
