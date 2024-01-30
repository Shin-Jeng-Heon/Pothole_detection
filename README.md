# Excel Export


3개의 값을 입력받아 하나의 행으로 처리한 후, Excel파일로 만들어주는 모듈


## 설치 방법

`pandas`와 `openpyxl` 설치 혹은 `requirements.txt` 설치


```bash
pip install pandas openpyxl
```


Or


```bash
pip install requirements.txt
```

<br>


## 사용 방법


### 1. 모듈 불러오기


```python
from ExcelExport import Saver
```

<br>


### 2. 데이터 입력 방법


```python
Saver.add_data(image_name, longitude, latitude)
```


### `add_data(image_name, longitude, latitude)`

이 메서드는 새로운 데이터를 데이터 세트에 추가합니다.

#### 매개변수:

- `image_name` (`str`): 이미지 파일의 이름.
- `longitude` (`float`): 이미지가 촬영된 위치의 경도.
- `latitude` (`float`): 이미지가 촬영된 위치의 위도.

#### 예제 사용:

```python
Saver.add_data("example.jpg", 127.123456, 37.123456)
```


### 3. 추가한 데이터들 엑셀 파일로 저장하기


```python
Saver.save_to_excel()
```


이 코드를 추가하면 Detection Result폴더에 Excel파일이 하나씩 저장됨.


### 4. 예제 코드


input.test.py를 실행



#### 예시 입력

```bash
이름테스트 22 33
이름테스트2 33 44
```

위와 같이 예시 입력 후


```bash
1 1 1
```
을 입력하게 되면 Detection Result폴더에 Excel파일이 저장되는 모습을 볼 수 있음

input_test.py코드를 보면 모듈 사용방법을 알 수 있음