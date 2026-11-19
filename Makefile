.DEFAULT_GOAL := all

BAZEL ?= bazel
BAZEL_FLAGS ?=
JOBS ?= 16
DOCKER ?= docker
OVMS_MEDIA_IMAGE_TAG ?= local
IMAGE ?= mediapipe-openvino:$(OVMS_MEDIA_IMAGE_TAG)
DEMOS_IMAGE ?= mediapipe-openvino-demos:$(OVMS_MEDIA_IMAGE_TAG)
RUNTIME_IMAGE ?= mediapipe-openvino-runtime:$(OVMS_MEDIA_IMAGE_TAG)

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

.PHONY: all build tests test example-tests python clean docker_build ovms_demos_image release_image test_release_image tests_in_docker run_demos_in_docker

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

release_image:
	$(DOCKER) build --target release --build-arg JOBS=$(JOBS) -t $(RUNTIME_IMAGE) .

test_release_image:
	mkdir -p test-results
	$(DOCKER) run --rm -v "$(CURDIR)/test-results:/test-results" $(IMAGE) \
	  ffmpeg -hide_banner -loglevel error -f lavfi -i 'testsrc=size=64x48:rate=5:duration=1' \
	  -frames:v 3 -c:v mjpeg -pix_fmt yuvj420p -y /test-results/example_video.avi
	$(DOCKER) run --rm -v "$(CURDIR)/test-results/example_video.avi:/mediapipe/testdata/example_video.avi:ro" $(RUNTIME_IMAGE) \
	  --model_path=/mediapipe/testdata/identity.xml \
	  --input_video_path=/mediapipe/testdata/example_video.avi \
	  --device=CPU --input_width=2 --input_height=1 --max_frames=2 --top_k=0 \
	  > test-results/release-smoke.log 2>&1
	grep -Fq 'frame 0 output[0]' test-results/release-smoke.log
	grep -Fq 'frame 1 output[0]' test-results/release-smoke.log
	! grep -Fq 'frame 2 output[0]' test-results/release-smoke.log

tests_in_docker:
	mkdir -p test-results
	$(DOCKER) run --rm -v "$(CURDIR)/test-results:/test-results" -e TEST_RESULTS_DIR=/test-results $(IMAGE) bash ci/run_tests.sh tests

run_demos_in_docker:
	mkdir -p test-results
	$(DOCKER) run --rm -v "$(CURDIR)/test-results:/test-results" -e TEST_RESULTS_DIR=/test-results $(DEMOS_IMAGE) bash ci/run_tests.sh example-tests