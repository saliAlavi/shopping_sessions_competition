IMAGE ?= shopping-sessions:latest
CPUS  ?= 8
MEM   ?= 24g
PLATFORM ?= linux/amd64
DEV   ?= dev
ANSWER ?= dev-answer
export IMAGE CPUS MEM PLATFORM

.PHONY: native fixtures-native build fixtures dev run-dev check-dev

## Build the binary on the host (needs g++ >= 11)
native:
	mkdir -p build && g++ -O2 -march=x86-64-v3 -std=c++17 -o build/pipeline src/main.cpp -lpthread

## Run all 36 fixtures against the host binary
fixtures-native: native
	python3 scripts/run_fixtures.py fixtures -- build/pipeline

## Build the image for the scorer's platform
build:
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE) .

## Run the image on all 36 edge-case fixtures and diff against the expected tables
fixtures: build
	python3 scripts/run_fixtures.py fixtures -- scripts/docker_run.sh

## Make the 1M-event dev set with the published generator (needs `generate` on PATH)
dev:
	generate --seed 1 --events 1000000 --clicks 250000 --output $(DEV)

## Run on the dev set with scorer flags and time it
run-dev: build
	rm -rf out && time scripts/docker_run.sh $(DEV) out

## Compare the dev output with the published dev answer (dev-answer.zip unzipped here)
check-dev:
	python3 scripts/compare.py out $(ANSWER)
