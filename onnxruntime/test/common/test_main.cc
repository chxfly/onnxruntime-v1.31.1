// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/test_environment.h"

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const OrtApi* Ort::g_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;

  try {
    const bool create_default_logger = false;
    onnxruntime::test::TestEnvironment environment{argc, argv, create_default_logger};

    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }

  return status;
}
