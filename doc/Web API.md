# Docker Web API 목록

## 이미지 관리 API

### 이미지 생성

```jsx
POST /api/image/create
```

지정된 데이터로 새로운 이미지를 생성한다.

POST시 body에 아래와 같은 데이터로 구성된 json 객체를 전달해야 한다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| name | string | 생성할 이미지의 이름을 지정한다 |
| baseImage | string | 기본으로 사용할 Docker 이미지의 Tag를 지정한다 |
| update | boolean | 기본 이지미에 apt update && apt upgrade 명령을 수행할지 여부를 지정한다 |
| aptInstall | string | apt install 명령어로 추가로 설치할 패키지를 문자열로 지정한다. 패키지는 빈칸으로 구분하여 하나의 문자열로 넘겨준다 |
| pipInstall | string | pip install 명령어로 추가로 설치할 패키지를 문자열로 지정한다. 패키지는 빈칸으로 구분하여 하나의 문자열로 넘겨준다 |
| additionalCommand | string | 패키지 설치후 추가로 입력할 명령어를 지정한다. 명령어는 ‘\n’으로 구분하여 하나의 문자열로 넘겨준다. |

결과로 json 객체를 리턴하며 해당 객체는 success 필드값을 boolean으로 가진다.  리턴은 생성이 완료되기를 기다리지 않고 생성 프로세스를 시작하면 바로 리턴한다. 이후 생성 과정을 모니터링 하기 위해서는 해당 이미지 이름을 이용하여 생성 정보 얻기 API를 사용한다.

### 이미지 생성 정보 얻기

```jsx
GET /api/image/create/{name}
```

생성중인 이미지의 정보를 얻는다. URL의 파라메터 name에는 생성하는 이미지의 이름을 지정하면 된다.

결과로 json 객체를 리턴하며 다음과 같은 데이터를 가진다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| status | string | 이미지 생성 프로세스의 실행 상태로 ‘running’ 또는 ‘exited’ 상태 중의 하나의 값을 가진다. |
| lines | [string] | 이미지 생성 프로세스의 출력을 문자열의 리스트로 가진다. |

### 이미지 생성 정보 삭제

```jsx
DELETE /api/image/create/{name}
```

생성중인 이미지의 정보를 삭제한다. URL의 파라메터 name에는 생성하는 이미지의 이름을 지정하면 된다.

이미지 생성이 완료되 경우 (이미지 생성 정보를 얻었을때 status가 exited인 경우) 더 이상 해당 생성 과정을 모니터링하지 않으려는 경우에 호출한다. 이를 통해서 서버가 해당 이미지 생성 과정에 대한 데이터를 더 이상 유지하지 않게 해준다.

해당 API의 호출이 이미지 생성 과정을 중단하게 하지는 않는다.

추후 DB를 통해서 이미지 관련 정보를 저장하게 한다면 사용법이 달라질 수 있다.

### 이미지 목록 얻기

```jsx
GET /api/image/list
```

생성된 이미지의 목록을 얻는다.

결과로 json 객체의 배열을 리턴하며 각 객체는 docker-py에서 유지하는 모든 데이터를 포함한다. 그중 주요한 필드는 다음과 같다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| RepoTag | [string] | 이미지의 태그 이름 |
| Created | number | 생성된 시간 (Unix timestamp, 초단위) |
| Id | string | 이미지의 id |
| Size | number | 이미지의 크기, 바이트수 |

### 이미지 삭제

```jsx
DELETE /api/image/item/{name}
```

생성된 이미지를 삭제한다.  URL의 파라메터 name에는 삭제하려는 이미지의 이름을 지정하면 된다.

## 실행 관리 API

### 실행 수행

```jsx
POST /api/exec/create
```

새로운 실행을 시작한다.

POST시 body에 아래와 같은 데이터로 구성된 json 객체를 전달해야 한다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| srcPath | string | 실행할 소스코드가 위한 곳의 경로를 지정한다. 실행 컨테이너에서는 ‘/app’ 위치로 마운팅된다. |
| mainSrc | string | 소스코드에서 실행할 파이썬 메인 소스 파일을 지정한다. |
| imageTag | string | 실행할 이미지의 이름을 지정한다. |
| inputPath | string | 소스코드에서 참고할 input 데이터의 경로를 지정한다. 해당 경로는 소스코드에서 ‘./data’ 또는 ‘/app/data’로 참조할 수 있다. |
| outputPath | string | 소스코드에서 참고할 output 데이터의 경로를 지정한다. 해당 경로는 소스코드에서 ‘./output’ 또는 ‘/app/output’으로 참조할 수 있다. |

결과로 json 객체를 리턴하며 다음과 같은 데이터를 가진다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| exec_id | string | 새로 실행된 프로세스에 대한 id 값, 이 값을 통해 이후 상태 정보를 얻을 수 있다 |

### 실행 정보 얻기

```jsx
GET /api/exec/info/{exec_id}
```

실행 중인 프로세스의 정보를 얻는다. URL의 파라메터 exec_id에는 생성시에 리턴된 id값을 지정한다.

결과로 json 객체를 리턴하며, 객체는 docker-py에서 유지하는 모든 데이터를 포함한다. 그중 주요한 필드는 다음과 같다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| Args | [string] | 실행시에 커맨드 파라메터로 넘겨준 값의 목록 |
| Created | string | 실행 시작 시간으로 ISO문자열 형태 |
| Path | string | 실행 프로세스의 실행 파일 경로 |
| State.Running | boolean | 실행 중인지의 여부 |

### 실행 로그 얻기

```jsx
GET /api/exec/logs/{exec_id}
```

실행 프로세스의 출력 로그를 얻는다. URL의 파라메터 name에는 생성하는 이미지의 이름을 지정하면 된다.

결과로 json 객체를 리턴하며 다음과 같은 데이터를 가진다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| lines | [string] | 실행 프로세스의 출력을 문자열의 리스트로 가진다. |

### 실행 목록 얻기

```jsx
GET /api/exec/list
```

모든 실행의 목록을 얻는다. 목록의 각 이이템은 실행에 대한 정보를 가지고 있으며 각 필드의 의미는 실행 수행 명령의 설명을 참고한다. 추가로 실행 정보 얻기 (info)를 통해 얻을 수 있는 정보를 'container'라는 필드를 통해 접근 가능하다.

### 실행 중단

```jsx
PUT /api/exec/stop/{exec_id}
```

실행 프로세스를 중단한다. URL 파라메터인 exec_id에는 생성시에 리턴된 id값을 지정한다.

실행 중단은 강제로 프로세스를 종료하는 것으로 저장되지 않은 진행 정보는 손실된다.

### 실행 정보 삭제

```jsx
DELETE /api/exec/item/{exec_id}
```

실행 프로세스의 정보를 삭제한다. URL의 파라메터 exec_id에는 생성시에 리턴된 id값을 지정한다.

실행이 완료되 경우 (실행 정보를 얻었을때 State.Running이 false인 경우) 더 이상 해당 실행을 모니터링하지 않으려는 경우에 호출한다. 이를 통해서 서버가 해당 실행 과정에 대한 데이터를 더 이상 유지하지 않게 해준다.

해당 API의 호출이 실행을 중단하게 하지는 않는다.

추후 DB를 통해서 실행 관련 정보를 저장하게 한다면 사용법이 달라질 수 있다.

### 실행 진행 정보 얻기

```jsx
GET /api/exec/progress/{exec_id}
```

실행 프로세스의 진행 정보를 얻는다. URL의 파라메터 exec_id에는 생성시에 리턴된 id값을 지정한다.

진행 정보는 실행하는 소스 코드에서 진행도를 저장해주어야 지원이 된다. 각 소스 코드는 진행 정보를 도커내의 "/apprun/progress.json" 파일에 저장해야 한다. 파일은 다음과 같은 정보를 포함해야 진행도를 UI에 표시할 수 있다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| main_progress | number | 실행 프로세스의 진행 정도를 퍼센트로 지정한다. (0~100) |

프로젝트 진행에 따라서 정보 필드는 확장될 수 있다.


### 실행 결과 얻기

```jsx
GET /api/exec/result/{exec_id}
```

실행 프로세스의 실행 결과를 얻는다. URL의 파라메터 exec_id에는 생성시에 리턴된 id값을 지정한다.

결과는 실행하는 소스 코드에서 해당 정보를 저장해주어야 지원이 된다. 각 소스 코드는 결과 정보를 도커내의 "/apprun/result.json" 파일에 저장해야 한다. 파일은 다음과 같은 정보를 포함해야 결과를 UI에 표시할 수 있다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| loss | number | 학습의 최종 loss 값을 지정한다. |
| metric | number | 학습의 평가 실행의 metric 값을 지정한다. |
| metric_name | string | metric 방법을 문자열로 지정한다. (예: accuracy, f1-score, mse 등) |

프로젝트 진행에 따라서 정보 필드는 확장될 수 있다.

## 스토리지 관리 API

### 스토리지 목록 얻기

```jsx
GET /api/storage/list
```

등록된 스토리지의 목록을 얻는다. 스토리지는 시스템이 미리 정의된 위치로 웹 클라이언트로 접근 가능한 경로를 지정한다.

스토리지는 학습에 사용할 데이터셋, 전처리기 소스코드, 학습 모델 소스코드 등을 저장하고 사용할 수 있도록 해준다.

스토리지 목록의 각 아이템은 다음과 같은 필드값을 가진다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| id | string | 스토리지의 ID, 다른 스토리지 API를 사용할 때 필요 |
| name | string | 스토리지의 이름, UI상에 표시할 때 사용 |


### 스토리지 파일 목록 얻기

```jsx
GET /api/storage/list/{storage_id}
GET /api/storage/list/{storage_id}/{file_path}
GET /api/storage/list/{storage_id}?page={page}&count={count}
GET /api/storage/list/{storage_id}/{file_path}?page={page}&count={count}
```

스토리지내의 파일, 디렉토리의 목록을 얻는다. URL 파라메터의 stroage_id는 스토리지의 ID를 file_path에는 해당 스토리지내의 세부 경로를 지정한다. file_path를 지정하지 않으면 해당 스토리지의 기본 경로의 목록을 리턴한다.

URL의 쿼리 파라메터로 다음과 같은 페이지 파라메터를 추가할 수 있다.

| 필드이름 | 필드타입 | 설명 | 기본값 |
| --- | --- | --- | --- |
| page | number | 한 번에 리턴할 아이템의 개수를 제한하는 경우 몇 페이지의 아이템을 리턴할지를 지정한다. 페이지 번호는 0부터 시작한다 | 0 |
| count | number | 한 번에 리턴할 아이템의 개수, 0인 경우 전체 아이템을 리턴한다. | 0 |


리턴되는 정보는 다음과 같은 필드값을 가진다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| id | string | 스토리지의 ID |
| file_path | string | 스토리지의 경로 요청한 file_path 값과 동일한 값 |
| page | number | 페이지 번호로 요청한 page 값과 동일한 값 |
| count | number | 페이지 당 아이템 개수로 요청한 count 값과 동일한 값 |
| total_count | number | 해당 경로의 전체 아이템 개수 |
| items | object | 해당 경로의 폴더 및 파일에 대한 정보 |

items 목록의 각 항목이 가지는 필드값은 다음과 같다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| name | string | 파일 또는 폴더의 이름 |
| is_dir | boolean | 폴더인 경우 true, 파일인 경우 false 값을 가진다 |
| size | number | 파일의 크기 정보. 폴더의 경우 0값을 가진다 |
| mtime | number | 파일이 수정된 시각 정보. epoch로부터의 시간 (초단위, 부동소숫점) |

### 스토리지에 폴더 생성

```jsx
PUT /api/storage/dir/{storage_id}/{file_path}
```

스토리지에 폴더를 생성한다. URL 파라메터의 stroage_id는 스토리지의 ID를 file_path에는 생성할 경로를 지정한다.

#### 에러 코드

| 코드 | 텍스트 | 설명 |
| --- | --- | --- |
| 403 | File already exist | 해당 위치에 이미 폴더 또는 동일 이름의 파일이 존재하는 경우 |


### 스토리지의 파일 데이터 얻기

```jsx
GET /api/storage/file/{storage_id}/{file_path}
```

스토리지의 파일 데이터를 얻는다. URL 파라메터의 stroage_id는 스토리지의 ID를 file_path에는 파일의 경로를 지정한다.

파일 데이터는 별다른 처리 없이 리턴되기 때문에 해당 URL형식은 image 태그의 src에도 사용할 수 있다.

#### 에러 코드

| 코드 | 텍스트 | 설명 |
| --- | --- | --- |
| 404 | File not found | 해당 위치에 파일이 없는 경우 |
| 503 | File access not allowed | 해당 파일에 접근 권한이 없는 경우 |


### 스토리지에 파일 업로드

```jsx
POST /api/storage/file/{storage_id}/{file_path}
```

스토리지에 파일 데이터를 업로드한다. URL 파라메터의 stroage_id는 스토리지의 ID를 file_path에는 **파일을 업로드할 경로**를 지정한다.

폼의 파일 업로드를 이용하여 파일을 생성하는 경우에 사용한다.

#### 에러 코드

| 코드 | 텍스트 | 설명 |
| --- | --- | --- |
| 404 | Path not found | 업로드할 경로가 존재하지 않는 경우 |

### 스토리지에 파일 저장

```jsx
PUT /api/storage/file/{storage_id}/{file_path}
```

스토리지에 파일 데이터를 저장한다. URL 파라메터의 stroage_id는 스토리지의 ID를 file_path에는 저장할 파일의 경로를 지정한다.

POST를 통해 직접 파일을 생성하거나 내용을 변경하려는 경우에 사용한다. 파일이 해당 경로에 이미 존재하는 경우 에러없이 내용을 덮어쓰기 한다.

### 스토리지 아이템 (파일, 폴더) 삭제

```jsx
DELETE /api/storage/item/{storage_id}/{file_path}
```

스토리지의 파일 또는 폴더를 삭제한다. URL 파라메터의 stroage_id는 스토리지의 ID를 file_path에는 삭제할 아이템의 경로를 지정한다.

#### 에러 코드

| 코드 | 텍스트 | 설명 |
| --- | --- | --- |
| 400 | _unspecified_ | 삭제시 발생한 일반 오류 |
| 404 | File not found | 삭제할 파일이 존재하지 않는 경우 |

### 스토리지에 아이템 업로드

```jsx
POST /api/storage/upload_item
```

스토리지에 위치를 지정하지 않고 임시로 데이터를 업로드한다. 사용자의 코드를 일회성으로 실행하거나 하는 경우에 사용할 수 있다.

폼의 파일 업로드를 이용해서 호출해야 한다.

업로드 된 파일은 일반 스토리지가 아닌 업로드 아이템을 위한 특별한 위치에 저장되며 별도의 삭제 명령을 보낼 때까지 유지된다.

업로드시 zip 압축 파일인 경우 해당 zip 파일내에 포함된 파일 목록을 분석하여 리턴해 줄 수도 있다. (폼 제출시 데이터에 unzip=true를 포함)

업로드 완료 후 다음과 같은 정보를 리턴한다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| success | boolean | 업로드 성공 여부 |
| files | [string] | 업로드 된 파일 목록 |
| unzip_files | [string] | 압축 파일내의 파일 이름을 포함한 파일 목록, 폼 데이터에 unzip=true를 포함한 경우에만 동작함 |
| upload_id | string | 업로드 아이템의 ID |

### 스토리지의 업로드 아이템 삭제

```jsx
DELETE /api/storage/upload_item/{upload_id}
```

업로드 아이템 API로 업로드된 파일들을 삭제한다. 파라메터의 upload_id는 아이템 업로드의 결과로 리턴된 upload_id 값을 지정한다.

## 데이터셋 관리 API

### 데이터셋 목록 얻기

```jsx
GET /api/dataset/list
```

데이터셋의 목록을 얻는다. 데이터셋 정보는 다음과 같은 필드로 구성된다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| name | string | 데이터셋의 이름 |
| type | string | 데이터셋의 타입. Image, Text, Table과 같은 값을 가진다 |
| storageId | string | 데이터셋이 위치하는 스토리지 ID |
| storagePath | string | 데이터셋이 위치하는 경로 |
| description | string | 데이터셋의 설명 |
| family | string | 데이터셋 패밀리 |

### 데이터셋 목록 저장

```jsx
POST /api/dataset/list
```

데이터셋의 목록 전체를 저장한다. 기존의 데이터셋 목록이 대체되므로 주의해야 한다.

### 데이터셋 추가

```jsx
POST /api/dataset/item/{name}
```

데이터셋을 추가한다. 파라메터 URL의 name에는 추가할 데이터셋의 이름을 지정한다. 전송할 데이터는 데이터셋 목록에 정의한 필드값을 지정하면 된다.

### 데이터셋 삭제

```jsx
DELETE /api/dataset/item/{name}
```

데이터셋을 삭제한다. 파라메터 URL의 name에는 삭제할 데이터셋의 이름을 지정한다.

## 모델 관리 API

### 모델 목록 얻기

```jsx
GET /api/model/list
```

모델의 목록을 얻는다. 모델 정보는 다음과 같은 필드로 구성된다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| name | string | 모델의 이름 |
| type | string | 모델의 타입. Model, Preprocessor와 같은 값을 가진다 |
| storageId | string | 모델이 위치하는 스토리지 ID |
| storagePath | string | 모델이 위치하는 경로 |
| mainSrc | string | 모델의 메인 소스 파일 이름 |
| description | string | 모델의 설명 |
| family | string | 모델의 데이터셋 패밀리 |

### 모델 목록 저장

```jsx
POST /api/model/list
```

모델의 목록 전체를 저장한다. 기존의 모델 목록이 대체되므로 주의해야 한다.

### 모델 추가

```jsx
POST /api/model/item/{name}
```

모델을 추가한다. 파라메터 URL의 name에는 추가할 모델의 이름을 지정한다. 전송할 데이터는 모델 목록에 정의한 필드값을 지정하면 된다.

### 모델 삭제

```jsx
DELETE /api/dataset/item/{name}
```

모델을 삭제한다. 파라메터 URL의 name에는 삭제할 모델의 이름을 지정한다.

## 앱 모듈 관리 API

### 앱 모듈 목록 얻기

```jsx
GET /api/app/list
```

앱 모듈의 목록을 얻는다.

앱 모듈은 다음과 같은 필드로 구성된다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| id | string | 앱 모듈의 ID |
| name | string | 앱 모듈의 이름 |
| type | string | 앱 모듈의 타입으로 "server", "script" 값을 가질수 있다 |
| image | object | 이미지 정보 |
| execution | object | 앱 모듈 실행 정보 |

이미지 정보는 다음과 같은 필드로 구성된다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| tag | string | 이미지의 태그(이름) |
| build | object | 이미지 생성 정보 |

이미지의 build 정보는 다음과 같은 필드로 구성된다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| base | string | 이미지 빌드의 기본으로 사용할 이미지의 태그 이름 |
| update | boolean | 이미지 빌드시 소프트웨어 업데이트를 수행할지의 여부 |
| apt | string | 이미지 빌드시 설치할 모듈 목록으로 빈칸으로 구분하여 하나의 문자열로 표현한다 |
| pip | string | 이미지 빌드시 pip로 설치할 모듈 목록으로 빈칸으로 구분하여 하나의 문자열로 표현한다 |
| additional_command | [string] | 이미지 빌드시 실행할 명령어들로 여러개인 경우 문자열의 리스트로 입력할 수 있다 |

앱 모듈의 execution에 해당하는 정보는 다음과 같은 필드로 구성된다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| src | string | 실행할 소스코드의 경로 |
| command_params | [string] | 실행할 명령어를 문자열의 리스트로 지정한다. 하나의 명령어를 파라메터를 구분하여 리스트로 지정하는 것 |
| input | string | 입력 데이터의 경로 |
| output | string | 출력 데이터의 경로 |
| port | number | 서버로 실행하는 경우 사용할 포트 번호 |

### 앱 모듈 실행

```jsx
POST /api/app/run/{module_id}
```

앱 모듈을 실행한다. URL 파라메터의 module_id에는 실행할 모듈의 ID를 지정한다.

실행시 전달할 데이터의 포맷은 앱 모듈에 따라 달라진다. POST로 전달되는 데이터 객체는 서버의 경우 call_server 함수의 파라메터로 전달되고, 스크립트의 경우 입력 데이터 경로의 params.json 파일로 전달된다.

리턴되는 데이터의 포맷은 앱 모듈에 따라 달라진다. 서버의 경우 call_server 함수의 리턴값이 그대로 리턴되며, 스크립트의 경우 출력 데이터 경로의 result.json 파일의 내용이 리턴값으로 사용된다.

## 시스템 API

### 시스템 정보 얻기

```jsx
GET /api/system/info
```

시스템의 현재 상태 정보를 얻는다.

시스템 정보는 다음과 같은 필드를 포함한다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| cpu_info | string | CPU 명칭 |
| cpu_thread_count | number | CPU의 논리 프로세서의 개수 |
| cpu_core_count | number | CPU의 물리 프로세서의 개수 |
| cpu_util | number | CPU 사용량 (퍼센트) |
| total_memory | number | 메인 메모리의 전체 바이트 수 |
| available_memory | number | 사용 가능한 메인 메모리의 전체 바이트 수 |
| platform | string | 시스템 플랫폼의 이름 |
| node | string | 시스템의 네트워크 상의 노드 이름 |
| disk_total | number | 설치된 디스크의 크기 (바이트) |
| disk_used | number | 디스크의 사용된 용량의 바이트 수 |
| disk_free | number | 디스크의 사용 가능 용량의 바이트 수 |
| gpu_info | object | GPU 정보 |

GPU 정보는 다음과 같은 필드를 포함한다.

| 필드이름 | 필드타입 | 설명 |
| --- | --- | --- |
| name | string | GPU 제품명 |
| gpu_util | string | GPU 코어 사용률 |
| mem_util | string | GPU 메모리 사용률 |
| total_mem | string | GPU 전체 메모리 |
| used_mem | string | GPU 사용된 메모리 |
| free_mem | string | GPU 사용가능 메모리 |
| temp | string | GPU 온도 |
| power | string | GPU 전력 |
| power_limit | string | GPU 전력 한도 |
