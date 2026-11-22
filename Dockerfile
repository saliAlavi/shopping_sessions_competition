# Shopping Sessions: native C++ pipeline, built as one static binary.
# The scorers are linux/amd64 Intel Xeon (Cascade Lake or newer); the binary
# targets x86-64-v3 (AVX2, BMI2), which every such machine supports.
FROM debian:bookworm-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ /src/
RUN g++ -O2 -march=x86-64-v3 -std=c++17 -static -o /pipeline main.cpp -lpthread

FROM debian:bookworm-slim
COPY --from=build /pipeline /app/pipeline
# The scorer runs as uid 1000 with a read-only root filesystem and no network.
# Reads $DATA_DIR (/data), writes $OUTPUT_DIR (/output); nothing else is touched.
CMD ["/app/pipeline"]
