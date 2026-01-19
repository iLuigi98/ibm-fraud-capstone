install:
	pip3 install -r requirements.txt

pull-data:
	dvc pull

push-data:
	dvc push

lint:
	python3 -m compileall src
