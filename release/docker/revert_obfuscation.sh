#!/bin/bash
# Description: Script responsible for reverting obfuscation.

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/workspace/tao-deploy' || REPO_ROOT="${WORKSPACE}"

echo "Restoring the original project structure"
# Move the obf_src files.
rm -rf ${REPO_ROOT}/nvidia_tao_deploy/*

# Move back the original files
mv /orig_src/* ${REPO_ROOT}/nvidia_tao_deploy/

# Remove the tmp folders.
rm -rf /orig_src
rm -rf /obf_src
rm -rf ${REPO_ROOT}/pyarmor_runtime_001219
