.DEFAULT_GOAL := all

BAZEL ?= bazel
BAZEL_FLAGS ?=
JOBS ?= 16
DOCKER ?= docker
OVMS_MEDIA_IMAGE_TAG ?= local
IMAGE ?= mediapipe-openvino:$(OVMS_MEDIA_IMAGE_TAG)
DEMOS_IMAGE ?= mediapipe-openvino-demos:$(OVMS_MEDIA_IMAGE_TAG)

ifeq ($(OS),Windows_NT)
SHELL := bash.exe
.SHELLFLAGS := -c
BAZEL_STARTUP_FLAGS ?= --output_user_root=C:/b
else
SHELL := /bin/bash
BAZEL_STARTUP_FLAGS ?=
endif

BAZEL_CMD = $(BAZEL) $(BAZEL_STARTUP_FLAGS)
CALCULATORS := //mediapipe/calculators/openvino:openvino_calculators
EXAMPLE := //mediapipe/examples/desktop/openvino_classification:openvino_classification
UNIT_TESTS := //mediapipe/calculators/openvino:openvino_session_calculator_test \
              //mediapipe/calculators/openvino:openvino_calculators_concurrency_test
EXAMPLE_TEST := //mediapipe/examples/desktop/openvino_classification:openvino_classification_test
PYTHON_DEMO := //mediapipe/python/openvino:openvino_classification_demo

.PHONY: all build tests test example-tests python clean docker_build ovms_demos_image tests_in_docker run_demos_in_docker

all: build tests example-tests

build:
	$(BAZEL_CMD) build --jobs=$(JOBS) $(BAZEL_FLAGS) $(CALCULATORS) $(EXAMPLE)

tests:
	$(BAZEL_CMD) test --jobs=$(JOBS) --test_output=errors $(BAZEL_FLAGS) $(UNIT_TESTS)

test: tests example-tests

example-tests:
	$(BAZEL_CMD) test --jobs=$(JOBS) --test_output=errors $(BAZEL_FLAGS) $(EXAMPLE_TEST)

python:
	$(BAZEL_CMD) build --jobs=$(JOBS) $(BAZEL_FLAGS) $(PYTHON_DEMO)

clean:
	$(BAZEL_CMD) clean

docker_build:
	$(DOCKER) build --target builder --build-arg JOBS=$(JOBS) -t $(IMAGE) .

ovms_demos_image:
	$(DOCKER) build --target demos --build-arg JOBS=$(JOBS) -t $(DEMOS_IMAGE) .

tests_in_docker:
	mkdir -p test-results
	$(DOCKER) run --rm -v "$(CURDIR)/test-results:/test-results" -e TEST_RESULTS_DIR=/test-results $(IMAGE) bash ci/run_tests.sh tests

run_demos_in_docker:
	mkdir -p test-results
	$(DOCKER) run --rm -v "$(CURDIR)/test-results:/test-results" -e TEST_RESULTS_DIR=/test-results $(DEMOS_IMAGE) bash ci/run_tests.sh example-tests