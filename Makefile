all: build install

build:
	python3 setup.py bdist_wheel

build_l4t:
	python3 setup_l4t.py bdist_wheel
	auditwheel repair dist/nvidia_tao_deploy-*.whl
	rm -rf dist/nvidia_tao_deploy-*.whl
	mv wheelhouse/*.whl dist/
	rm -rf wheelhouse/

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

install:
	pip3 install dist/nvidia_tao_deploy-*.whl

uninstall:
	pip3 uninstall -y nvidia-tao-deploy

