APP_PORT := 2444
DOCKER_TAG := latest



venv:
	python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

.PHONY: run_app
run_app:
	python3 -m uvicorn app:app --host='0.0.0.0' --port=$(APP_PORT)

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: download_weights
download_weights:
	wget -O weights/wine_classifier.ckpt https://disk.yandex.ru/d/SapdELfD1n2LdA
	wget -O weights/pizza_detector.pt https://disk.yandex.ru/d/vEZhuhi9jjbTVw
