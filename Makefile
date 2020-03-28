clean:
	rm -rf build; rm -rf dist; rm -rf *egg-info

build:
	python3 setup.py sdist bdist_wheel

upload:
	python3 -m twine upload dist/* --verbose

release:
	make clean; make build; make upload

install:
	pip3 install mpi4tf
