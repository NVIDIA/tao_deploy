all: build install

build:
	bash release/docker/obfuscate_source_code.sh
	python3 setup.py bdist_wheel
	bash release/docker/revert_obfuscation.sh

build_l4t:
	bash release/docker/obfuscate_source_code.sh
	python3 setup_l4t.py bdist_wheel
	auditwheel repair dist/nvidia_tao_deploy-*.whl
	rm -rf dist/nvidia_tao_deploy-*.whl
	mv wheelhouse/*.whl dist/
	rm -rf wheelhouse/
	bash release/docker/revert_obfuscation.sh

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

install:
	pip3 install dist/nvidia_tao_deploy-*.whl

uninstall:
	pip3 uninstall -y nvidia-tao-deploy

