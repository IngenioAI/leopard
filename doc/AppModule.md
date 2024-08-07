# AppModule

앱 모듈은 모델 또는 데이터 처리 소스 코드와 유사하지만 사용자가 업로드하는 코드가 아닌 이미 내재하고 있는 모듈을 의미한다.

앱 모듈은 사용자 코드와는 달리 커스텀한 결과를 전달할 수 있고, 이것을 커스텀 웹 페이지에서 표현할 수 있다.

주요 용도는 다음과 같다.
* 데이터의 수집
* 데이터의 개인 정보 검출
* 데이터 처리 (개인 정보 익명화 등)

## AppMoudle 정의

앱 모듈은 /app/{app_id}/appinfo.json 파일에 정의 되어야 사용이 가능하다.
각 앱 모듈을 정의하는 필드의 의미는 다음과 같다.

| 이름 | 타입 | 설명 |
|---|---|---|
| id | String | 앱 모듈의 id로, 필요시 디렉토리 이름으로 사용된다 |
| name | String | 앱 모듈의 명칭으로 UI에서 사용된다 |
| type | String | 실행 타입으로 커맨드 라인인 경우 "script", 서버인 경우 "server"로 지정한다 |
| image | Object | 실행할 도커의 이미지를 정의한다 |
| image.tag | String | 사용할 도커 이미지의 태그 이름 |
| image.build | Object | 도커 이미지를 새로 생성해야할때 참조할 정보 |
| image.build.base | Object | 도커 이미지 빌드시에 사용할 베이스 이미지 태그 |
| image.build.update | Boolean | 도커 이미지 빌드시 apt update를 수행할지의 여부 |
| image.build.apt | String | 도커 이미지 빌드시 apt로 설치할 패키지 목록으로, 각기 빈칸으로 구분 |
| image.build.pip | String | 도커 이미지 빌드시 pip로 설치할 패키지 목록으로, 각기 빈칸으로 구분 |
| execution | Object | 실행을 위한 정보 |
| execution.src | String | 실행할 소스의 기본 위치 |
| execution.main | String | 실행할 메인 소스 파일의 이름, 기본적으로 서버는 server.py, 커맨드 라인은 main.py 파일을 실행 |
| execution.command_params | String[]  | 실행시 추가할 커맨드 파라메터 |
| execution.input | String | 실행시에 연결할 입력 데이터의 경로 |
| execution.output | String |실행시에 연결할 출력 데이터의 경로 |
| execution.port | Integer | 서버로 실행시에 사용할 포트 번호 |


## AppModule List

| 이름 | 코드 위치 | 웹 페이지 | 설명 |
|---|---|---|---|
| mtcnn | /app/mtcnn | /ui/page/face_detect.html | 얼굴 검출 모듈 |
| faker | /app/faker | /ui/page/faker.html | 가상 개인 정보 생성기 |
| presidio | /app/presidio | /ui/page/presidio.html | 개인 정보 검출 및 처리기 |
| diffuser | /app/diffusers | /ui/generic_test.html | 가상 이미지 생성기 |

## 도커 내부 경로

앱 모듈은 도커를 통해 실행되기 떄문에 미리 정의된 내부 경로에서 데이터를 읽거나 써야 한다. 정의된 내부 경로는 다음과 같다.

| 이름 | 경로 | 설명 |
|---|---|---|
| 입력 데이터 | /data/input | 앱 모듈의 입력으로 전달되는 데이터, 파라메터 등의 위치 |
| 출력 데이터 | /data/output | 앱 모듈의 출력으로 전달되는 데이터의 위치 |

## 커맨드 라인 모듈

* 기본 소스코드는 "main.py"로 지정된다. (appinfo.json에서 변경 가능)
* 사용자의 입력 데이터 (웹 페이지를 통한 입력 또는 업로드된 파일)은 입력 데이터 경로인 "/data/input"에 저장된다.
* 파라메터 데이터의 경우 params.json 파일에 저장된다.
* 파라메터 이외의 데이터 파일이 업로드되는 경우 해당 파일을 입력 데이터 경로에 저장하고 params.json을 통해서 해당 파일의 이름을 알려주는 방식으로 구현해야 한다. (presidio의 image_redact 예제 참고)
* 커맨드 라인 모듈은 출력 경로에 result.json 파일을 생성하여 결과를 돌려줄 수 있다. 해당 결과는 실행 완료 시점에 Web API의 response로 전달된다.
* 출력으로 데이터 또는 이미지 파일을 전달하려는 경우 해당 파일을 출력 경로에 저장하고, 해당 파일 이름을 reponse에 지정해서 전달한다. 결과가 이미지인 경우 result.json내에 "image_path"라는 필드로 전달하면 Web UI에서 인식해서 이미지로 표시할 수 있다. (presidio의 image_redact 또는 diffuser 예제 참고)

## 서버 모듈

* 기본 소스코드는 "server.py"로 지정된다. (appinfo.json에서 변경 가능)
* 사용자의 입력 파라메터는 "/api/run" POST 명령 핸들러로 전달된다.
* 데이터 및 이미지 파일 입력은 입력 데이터 경로인 "/data/input"에 저장되며 파일 명은 입력 파라메터로 전달되는 방식으로 구현한다. (mtcnn의 server.py 참조)
* 출력 데이터는 "/api/run" 핸들러의 리턴값으로 JSON 객체를 리턴하여 전달 할 수 있다.
* 출력 데이터에 파일을 추가하려면 출력 데이터 경로에 해당 파일을 저장하고 "/api/run"핸들러의 리턴값을 통해서 파일 이름을 전달하면 된다. (mtcnn의 server.py - 'anonymize' 처리부분 참조)
* 서버 모듈은 "POST /api/run" 핸들러 이외에 "GET /api/ping" 메소드를 처리해야 한다. 해당 메소드는 서버 모듈이 현재 정상 동작임을 표현하는 것으로 다음과 같은 데이터를 리턴하면 된다.
  ```json
  {
    "success": true
  }
  ```

### 서버 모듈로 구현이 필요한 경우
* 대부분의 경우 커맨드 라인 모듈로 구현하면 충분하다
* 모델 초기화에 시간이 많이 소요되고, 해당 앱 모듈이 자주 호출되는 경우, 커맨드 라인 모듈은 실행시마다 초기화가 필요하므로 동작에 문제가 발생한다. 이러한 경우 서버 모듈을 사용하면 한 번 초기화 된 모듈을 계속 사용할 수 있다.
* 웹 UI와의 연동은 커맨드 라인 모듈, 서버 모듈 모두 동일한 인터페이스로 수행된다.

## 웹 UI

앱 모듈은 각기 모듈마다 테스트를 위한 웹 페이지를 가진다. 이는 현재 앱 매니저 페이지 (/webroot/js/pages/app_manager.js)에 getAppRunLink() 함수에 정의되어 있다.

미리 정의되지 않은 경우 generic_test.html 페이지로 연결된다.

## 제네릭 웹 UI

제네릭 웹 UI는 미리 정의된 테스트 페이지가 없는 앱 모듈을 위한 테스트 페이지를 의미한다. 현재는 diffuser 모듈에서만 사용하고 있다.

현재 제네릭 웹 UI의 기능은 다음과 같다.

* 입력 파라메터를 텍스트로 직접 입력할 수 있다.
* 실행 중 stdout 출력 로그를 확인할 수 있다.
* 출력 파라메터를 텍스트로 확인할 수 있다.
* 출력 파라메터 중에 "image_path" 필드가 존재하는 경우 해당 이미지를 표시한다.

