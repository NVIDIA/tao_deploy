#!/bin/bash
# Description: Script responsible for generation of an obf_src wheel using pyarmor package.

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/workspace/tao-deploy' || REPO_ROOT="${WORKSPACE}"
echo "Building from ${REPO_ROOT}"

echo "Installing required packages"
pip3 install --upgrade pip setuptools
pip3 install pyarmor==8.5.8 pyinstaller pybind11

echo "Registering pyarmor"
pyarmor -d reg ${REPO_ROOT}/release/docker/pyarmor-regfile-1219.zip || exit $?

echo "Clearing build and dists"
python3 ${REPO_ROOT}/setup.py clean --all
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo "Obfuscating the code using pyarmor"
# This makes sure the non-py files are retained.
pyarmor cfg data_files=*
pyarmor -d gen --recursive --output /obf_src/ ${REPO_ROOT}/nvidia_tao_deploy/ || exit $?

echo "Migrating codebase"
# Move sources to orig_src
rm -rf /orig_src
mkdir /orig_src
mv ${REPO_ROOT}/nvidia_tao_deploy/* /orig_src/

# Move obf_src files to src
mv /obf_src/* ${REPO_ROOT}/
