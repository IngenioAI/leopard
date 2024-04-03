# HTML Util

Leopard 시스템에서 사용하는 HTLM 확장 및 템플릿 기능을 설명한다

- 확장 태그는 `<LP-` 또는 `<!LP-` 로 시작하고 `</LP-` 및 `</!LP-`로 끝난다. 두 태그간의 차이점은 없다.
- 확장 태그는 전처리기로 동작하기 때문에 HTML 구조와 무관한 위치에 존재할 수 있다. 그러나 대부분은 태그을 교체하는 방식이므로 기존 태그 구조에 맞춰서 위치한다.
- 단 `<LP-include-string` 의 경우 단순 문자열로 교체되므로 문자열 내부에 위치할 수 있으며, 문자열이 아닌 곳에도 위치할 수 있다.


## LP-include-html

해당 위치에 다른 html 파일을 삽입한다.

| 속성 이름 | 타입 | 설명 |
|---|---|---|
| src | String | 삽입할 html의 위치 |

`src` 속성의 파일 위치는 기본적으로 소스 트리상의 `ui/fragment` 아래에 위치하며, 해당 위치에 파일이 없는 경우 `ui/dialog` 아래에서 찾는다.

## LP-template

해당 파일에서 템플릿의 사용을 정의한다.

| 속성 이름 | 타입 | 설명 |
|---|---|---|
| use | String | 사용할 템플릿의 이름 |

`use` 속성에서 정의하는 템플릿은 `/ui/template` 아래 위치한 템플릿 중 하나에 해당한다. 템플릿도 `LP-include-html`과 마찬가지로 다른 html 파일의 내용으로 대체되는 것이지만 다음과 같은 차이점이 존재한다.

- `LP-include-html`은 대체하는 파일의 내용이 그대로 대체되지만, `LP-include-template`는 대체하는 내용의 일부를 `LP-content` 태그로 지정하여 대체할 수 있다.

### LP-content

`LP-cotent`는 템플릿에서 삽입할 컨텐츠를 정의한다. 컨텐츠는 해당 엘리먼트의 하위 엘리먼트로 정의하면 된다.
| 속성 이름 | 타입 | 설명 |
|---|---|---|
| name | String | 정의할 템플릿 컨텐츠의 이름 |


### LP-include-content

`LP-include-content`는 템플릿으로 사용되는 html에서 사용되는 태그로 `LP-content`로 지정한 컨텐츠중 하나를 선택해서 삽입할 수 있다. 컨텐츠의 선택은 `LP-content`에서 정의한 `name`과 같은 값으로 `LP-include-content`의 `name`에 지정하면 된다.

| 속성 이름 | 타입 | 설명 |
|---|---|---|
| name | String | 삽입할 템플릿 컨텐츠의 이름 |

### LP-param

템플릿 컨텐츠와 유사하나 태그가 아닌 간단한 문자열이나 값을 컨텐츠로 지정하고 싶은 경우에 사용한다.

| 속성 이름 | 타입 | 설명 |
|---|---|---|
| name | String | 정의할 템플릿 파라메터의 이름 |
| value | Stirng | 해당 템플릿 파라메터의 값 |

정의된 파라메터는 템플릿 정의 파일에서 `LP-include-string` 태그로 사용할 수 있다. 보통은 템플릿에서 각자 다른 값을 가져야하는 id 등을 정의하기 위해서 사용된다.

### LP-include-string

`LP-param`으로 정의한 템플릿 문자열 값을 가져와서 대체하는 태그이다.

| 속성 이름 | 타입 | 설명 |
|---|---|---|
| name | String | 삽입할 템플릿 파라메터의 이름 |

## LP-include-script

다른 스크립트(js) 파일을 해당 위치에 삽입는 것으로 `script` 태그와 거의 유사하게 동작한다. 단 `src` 속성 값을 주지 않으면서 사용할 수 있는데, 그러면 해당 html 파일명과 같은 이름의 js 파일을 `js/page` 위치에서 로딩하려고 시도한다.

| 속성 이름 | 타입 | 설명 |
|---|---|---|
| src | String | 삽입할 스크립트 파일 경로 |

## LP-include-dialog-script

기본적으로 `LP-include-script`와 동일하지만 `src` 속성이 없는 경우 파일을 로딩하는 위치가 `js/dialog`가 된다.