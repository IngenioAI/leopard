from .. import Provider as PhoneNumberProvider
from numpy.random import choice

class Provider(PhoneNumberProvider):

    formats = (
        "02-####-####", # 서울
        "031-###-####", # 경기
        "032-###-####", # 인천
        "033-###-####", # 강원
        "041-###-####", # 충남
        "042-###-####", # 대전
        "043-###-####", # 충북
        "044-###-####", # 세종
        "051-###-####", # 부산
        "052-###-####", # 울산
        "053-###-####", # 대구
        "054-###-####", # 경북
        "055-###-####", # 경남
        "061-###-####", # 전남
        "062-###-####", # 광주
        "063-###-####", # 전북
        "064-7##-####", # 제주
        "010-####-####", # 휴대폰
        "011-###-####",  # 휴대폰
        "016-###-####", # 휴대폰
        "017-###-####", # 휴대폰
        "018-###-####", # 휴대폰
        "019-###-####", # 휴대폰
        "070-####-####", # 인터넷 전화
    )
    # 각 전화번호 실제 비율로 조절 (실제로 011,016,019등은 거의 존재안함)
    weights = [10 if format == "010-####-####" else 1 for format in formats]

    def phone_number(self):
        # 가중치에 따라 형식을 선택
        format = choice(self.formats, p=[weight/sum(self.weights) for weight in self.weights])
        return self.numerify(self.generator.parse(format))