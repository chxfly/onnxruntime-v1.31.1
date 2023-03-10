#!/bin/bash

# The script is to generate all supported versions of onnx models which will be tested by onnx_test_runner
# in the end of ci build pipeline. The purpose is to make sure latest onnxruntime has no regressions. Note
# that the order of installation must be onnx123, onnx130, onnx141, onnx150 and onnxtip since we want
# to keep the tip of master on script exit for onnx backend test which is also a part of build pipeline.
# One possible improvement here is to keep the models saved to some public storage instead of generating
# on the fly every time.

# The script build onnx from source because the prebuilt package doesn't conform to manylinux1 standard.

set -e
PYTHON_VER=$1

if [[ "$PYTHON_VER" = "3.5" && -d "/opt/python/cp35-cp35m"  ]]; then
   PYTHON_EXE="/opt/python/cp35-cp35m/bin/python3.5"
   export PATH=/opt/python/cp35-cp35m/bin:$PATH
elif [[ "$PYTHON_VER" = "3.6" && -d "/opt/python/cp36-cp36m"  ]]; then
   PYTHON_EXE="/opt/python/cp36-cp36m/bin/python3.6"
   export PATH=/opt/python/cp36-cp36m/bin:$PATH
elif [[ "$PYTHON_VER" = "3.7" && -d "/opt/python/cp37-cp37m"  ]]; then
   PYTHON_EXE="/opt/python/cp37-cp37m/bin/python3.7"
   export PATH=/opt/python/cp37-cp37m/bin:$PATH
else
   PYTHON_EXE="/usr/bin/python${PYTHON_VER}"
fi

version2tag=(5af210ca8a1c73aa6bae8754c9346ec54d0a756e-onnx123
             bae6333e149a59a3faa9c4d9c44974373dcf5256-onnx130
             9e55ace55aad1ada27516038dfbdc66a8a0763db-onnx141
             7d7bc83d29a328233d3e8affa4c4ea8b3e3599ef-onnx150
             553df22c67bee5f0fe6599cff60f1afc6748c635-onnxtip)
for v2t in ${version2tag[*]}; do
  onnx_version="$(cut -d'-' -f1<<<${v2t})"
  onnx_tag="$(cut -d'-' -f2<<<${v2t})"
  if [ -z ${lastest_onnx_version+x} ]; then
    echo "first pass";
  else
    echo "deleting old onnx-${lastest_onnx_version}";
    ${PYTHON_EXE} -m pip uninstall -y onnx
  fi
  lastest_onnx_version=$onnx_version
  GetFile https://github.com/onnx/onnx/archive/$onnx_version.tar.gz /tmp/src/$onnx_version.tar.gz
  tar -xf /tmp/src/$onnx_version.tar.gz -C /tmp/src
  cd /tmp/src/onnx-$onnx_version
  if [ ! -d "third_party/pybind11/pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git third_party/pybind11
  fi 
  ${PYTHON_EXE} -m pip install .
  mkdir -p /data/onnx/${onnx_tag}
  backend-test-tools generate-data -o /data/onnx/$onnx_tag
done
