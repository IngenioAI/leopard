# Leopard: 개인정보보호를 위한 인공지능 모델 개발 통합 프레임워크

개인정보보호를 준수하는 기계학습 모델 개발, 모델의 정책 반영/준수 성능 평가 등을 지원하는 웹 기반 프레임워크
  * 얼굴 이미지 데이터 비식별화 적용 --> 모델 학습, 성능 분석
  * 개인식별정보 가상데이터 생성, 비식별화 --> 모델 학습, 성능 분석
  * 연속학습을 통한 Machine Unlearning
  * 이메일/지문에 동형암호 적용 --> 모델 학습
  * 프레임워크에 데이터 블랙박스 적용

비식별화 미처리 데이터 인식, 비식별화 적용 등 데이터 처리 관련 기능

유연한 딥러닝 모델 학습/평가 기능
  * 각 개발 기술에 맞는 학습/평가 방법을 적용할 수 있는 유연한 학습/평가 프레임워크

벤치마크 데이터셋 (이미지 공개데이터) 포함
  * 얼굴 데이터: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [FFHQ](https://github.com/NVlabs/ffhq-dataset)
  * 지문 데이터: [SocoFing](https://www.kaggle.com/datasets/ruizgara/socofing)
  * 이메일 데이터: [enron dataset](https://github.com/MWiechmann/enron_spam_data)

## 설치방법

Leopard에서 데이터셋 가공, 모델 학습/평가를 수행하는 경우 도커의 컨터이너를 통해서 실행이되기 때문에 해당 시스템에 먼저 도커를 설치해야한다.

[도커 설치 참고](https://docs.docker.com/engine/install/)

Leopard는 python 환경에서 실행되는데 필요로하는 패키지들을 python 패키지 관리자인 poetry를 사용하고 있다. 따라서 poetry를 통해 환경을 구성하는 것을 권장한다.

[poetry 설치](https://python-poetry.org/docs/)

poetry를 설치한 후에는 다음 명령어로 관련 패키지를 설치한다.

```bash
poetry install
```

## 실행 방법

다음 명령어로 서버를 실행할 수 있다.

```bash
python server.py
```

서버가 실행되면 웹브라우저로 서버 주소의 12700 포트로 접속하면 된다.

실행된 서버가 도커 시스템을 사용할 수 있어야 하기 때문에 사용 권한이 없는 경우 /var/run/docker.sock 파일에 대한 접근 허가 등의 작업이 필요할 수 있다.
