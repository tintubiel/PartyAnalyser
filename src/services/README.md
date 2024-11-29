    
## Использование модели детекции напрямую
    
    detector = PizzaPepperoniDetector()
Детекция из изображения по массиву

    image_path = '/Users/tintubiel/Desktop/images.jpeg'
    image = Image.open(image_path)
    image = np.array(image)
    pizza_found, pepperoni_found = detector.detect_from_image(image)
    print(f'Pizza found: {pizza_found}, Pepperoni found: {pepperoni_found}')
Детекция из изображения по пути

    image_path = '/Users/tintubiel/Desktop/images.jpeg'
    pizza_found, pepperoni_found = detector.detect_from_image_path(image_path)
    print(f'Pizza found: {pizza_found}, Pepperoni found: {pepperoni_found}')

Детекция из изображения по URL

    url = "https://img.freepik.com/fotos-premium/grande-jantar-com-pizza-salada-e-batata-frita-aniversario-natal-ano-novo-jantar-festivo-uma-mesa-conceito-de-reuniao-ou-celebracao_102783-257.jpg?w=740"
    pizza_found, pepperoni_found = detector.detect_from_url(url)
    print(f'Pizza found from URL: {pizza_found}, Pepperoni found from URL: {pepperoni_found}')

Оценка вероятностей наличия пиццы и пеперони на изображения
   
    image_path = '/Users/tintubiel/Desktop/images.jpeg'
    pizza_found, pepperoni_found = detector.detect_image_proba(image_path)
    print(f'Pizza found: {pizza_found}%, Pepperoni found: {pepperoni_found}%')

_________________________________________________
## Использование модели классификации напрямую

    classificator = WineClassificator()
    
Классификация изображения по пути

    image_path = '/Users/tintubiel/Desktop/123.jpg'
    wine_found = classificator.classificate_from_image_path(image_path)
    print(f'Wine found: {wine_found}')

Классификация изображения по URL

    url = "https://img.freepik.com/fotos-premium/grande-jantar-com-pizza-salada-e-batata-frita-aniversario-natal-ano-novo-jantar-festivo-uma-mesa-conceito-de-reuniao-ou-celebracao_102783-257.jpg?w=740"
    wine_found = classificator.classificate_from_url(url)
    print(f'Wine found: {wine_found}')


