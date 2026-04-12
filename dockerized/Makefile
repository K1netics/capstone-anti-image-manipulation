ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

API_IMAGE ?= photoguard-api
WEB_IMAGE ?= photoguard-web
WEB_PORT ?= 8080

.PHONY: build run run-gpu stop package load push-ghcr pull-ghcr

build:
	"$(ROOT_DIR)/build-images.sh"

run:
	"$(ROOT_DIR)/run-stack.sh"

run-gpu:
	"$(ROOT_DIR)/run-stack-gpu.sh"

stop:
	"$(ROOT_DIR)/stop-stack.sh"

package:
	"$(ROOT_DIR)/package-images.sh"

load:
	"$(ROOT_DIR)/load-images.sh"

push-ghcr:
	"$(ROOT_DIR)/push-ghcr.sh"

pull-ghcr:
	"$(ROOT_DIR)/pull-ghcr.sh"
