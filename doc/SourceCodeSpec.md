# Source Code Spec

## Model Code

### 도커 내부 경로

도커 환경에서 실행되는 모델 코드는 다음과 같은 경로를 사용해야 한다.

| 명칭        | 경로          | 설명                                    |
|-----------|-------------|---------------------------------------|
| 입력 데이터 경로 | /data/input | 입력 데이터의 경로는 "/data/input"으로 참조할 수 있다. |
| 실행 경로     | /apprun     | 실행되는 소스 코드는 "/apprun" 위치에 마운트되어 실행된다  |

### 학습 진행 과정 저장

모델의 학습과정을 UI에 표시하기 위해서는 실행 경로에 progress.json 파일을 다음과 같이 생성하면 된다.

**/apprun/progress.json 예시**

```json
{
    "main_progress": 10.0,
    "epoch": 1,
    "batch": 1
}
```

각 필드 값은 다음과 같다.

| 필드명           | 타입                | 설명                     |
|---------------|-------------------|------------------------|
| main_progress | Float (0.0~100.0) | 현재 진행을 백분율로 표기         |
| epoch         | Integer           | 현재까지 완료된 epoch 수       |
| batch         | Integer           | 현재 epoch에서 진행한 batch 수 |

해당 필드에서 최소 "main_progress"값은 지정해야 학습 과정을 표시할 수 있으며, epoch와 batch는 생략할 수 있다.

### 평가 결과 저장

또한 학습 모델은 평가 결과를 저장하여 UI측에서 사용할 수 있다.

평가 결과는 실행경로에 result.json에 저장하면 된다.

**/apprun/result.json 예시**

```json
{
    "loss": 0.195734,
    "metric": 0.98641,
    "metric_name": "accuracy"
}
```

각 필드값은 다음과 같다.

| 필드명         | 타입     | 설명                     |
|-------------|--------|------------------------|
| loss        | Float  | 평가의 loss 값             |
| metric      | Float  | 평가의 metric 값           |
| metric_name | String | 평가 metric의 이름 (UI 표시용) |

## Data Processing Code

### 도커 내부 경로

도커 환경에서 실행되는 데이터 처리 코드는 다음과 같은 경로를 사용해야 한다.

| 명칭 | 경로 | 설명 |
|-----|-----|-----|
| 입력 데이터 경로 | /data/input | 입력 데이터의 경로는 "/data/input"으로 참조할 수 있다. |
| 실행 경로 | /apprun | 실행되는 소스 코드는 "/apprun" 위치에 마운트되어 실행된다 |
| 출력 데이터 경로 | /data/output | 데이터 처리 실행 결과는 "/data/output" 위치에 저장해야한다 |