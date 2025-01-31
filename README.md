# Документация для сервиса Party Analyser

## Описание

Интерактивный сервис, который использует компьютерное зрение для анализа фотографий вечеринок и помогает пользователям принимать решение о том, стоит ли им идти на вечеринку, исходя из наличия на изображении пиццы и вина, а также пеперони на пицце.
Приложение включает в себя API для работы с изображениями и получения статистических ответов.

## Установка

1. Клонируйте репозиторий:
   
```
git clone https://github.com/tintubiel/PartyAnalyser.git
```
2. Установите зависимости:
    
 ```   
 make venv
 make install
```
3. Загрузите веса моделей:

```
make download_weights
```
## Запуск приложения

Для запуска приложения используйте команду:
```
make run_app
```
## Маршруты API

### 1. GET /

- Описание: Возвращает приветственное сообщение.
- Ответ:{"message": "Добро пожаловать в Party Analyser"}
### 2. GET /answers

- Описание: Возвращает список доступных ответов.
- Ответ: {"answers": [...]}


### 3. POST /predict

- Описание: Принимает изображение и возвращает предсказание.
- Параметры:
  - image: Изображение в байтовом формате.
- Ответ:{"answer": "<предсказание>"}

## Модель PizzaWineAnalytics

### Описание

Класс PizzaWineAnalytics выполняет анализ изображения с целью определения наличия пиццы, пеперони и вина. Он использует модели детекции и классификации для обработки входного изображения и предоставляет информацию о том, стоит ли идти на вечеринку.

### Конструктор
```
def __init__(self, detection_model: PizzaPepperoniDetector, classification_model: WineClassificator)
```
- Параметры:
  - detection_model: Экземпляр модели PizzaPepperoniDetector, отвечающей за обнаружение пиццы и пеперони в изображении.
  - classification_model: Экземпляр модели WineClassificator, предназначенной для классификации наличия вина.

### Свойства

#### answers
```
@property
def answers(self)
```
- Описание: Возвращает набор предопределенных сообщений о наличии пиццы и вина. 

### Методы

#### predict
```
def predict(self, image: np.ndarray) -> Dict[str, bool]
```
- Описание: Выполняет предсказание наличия пиццы, пеперони и вина по предоставленному изображению.

#### _process_results
```
def _process_results(self, results) -> str
```
- Описание: Обрабатывает результаты предсказания и возвращает соответствующее сообщение о решении о вечеринке.

## Пример использования
```
    if __name__ == "__main__":
        classificator = WineClassificator()
        detector = PizzaPepperoniDetector()
        analyser = PizzaWineAnalytics(detector, classificator)
        image_path = '/Users/tintubiel/Desktop/img-20181208-205558-largejpg.jpg'
        image = Image.open(image_path)
        image = np.array(image)
        print(analyser.predict(image))
```
