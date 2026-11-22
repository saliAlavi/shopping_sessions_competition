# Shopping Sessions: native C++ pipeline

An entry for the Databricks **Shopping Sessions** track of the Builders Buildathon.
It turns the messy click-stream CSVs under `/data` into the two exact daily Parquet
tables, `/output/user_daily/` and `/output/campaign_daily/`, following `rules.md`.

The whole pipeline is one dependency-free C++17 file, `src/main.cpp`, compiled to a
static binary. It uses no CSV library, no Parquet library and no query engine.

## How it works

Every phase runs on all `NCPU` threads.

1. **Map.** Every input file is `mmap`ed into one reserved virtual region, with
   zero padding after each file, so the parser never copies and can read a few
   bytes past any field safely.
2. **Links.** Link lines are parsed into a lock-free open-addressing hash table of
   user IDs (64-bit hash to dense id). A lock-free union-find with CAS linking and
   path halving joins them. Each person is named by the byte-wise smallest of its
   IDs, found with a CAS-min over each component.
3. **Events and clicks.** These are parsed with an AVX2 field splitter that
   implements the CSV dialect exactly: quotes only around whole fields, exact field
   counts, and `\r` as an ordinary byte. Timestamps (F1/F2/F3) are validated and
   converted by hand.
   * Each event copy becomes a 24-byte record: the event_id hash, `ingest_ts`, the
     person, and a packed `ts | type | category | price` (or `INVALID`). Records
     are scattered into partitions by the event_id hash.
   * Each valid click becomes a 16-byte record: person, ts, and a campaign and
     click_id reference. Clicks are scattered by person. Click copies are identical,
     so deduplicating them cannot change a result, and it is skipped.
4. **Dedup.** In each event_id partition, a small hash table keeps the copy with
   the smallest `ingest_ts`. The survivor is kept only if that copy is valid. It is
   then scattered by person.
5. **Sessions.** In each person partition, a counting sort groups records by
   person, and each person's events are sorted by `ts`. Sessions split on gaps over
   30 minutes. A two-pointer sliding 7-day click window gives the credited click
   (latest `ts`, then smallest click_id) and the distinct assisting campaigns.
   Rows are aggregated per (person, day) and per (campaign, day).
6. **Output.** A minimal hand-written Parquet writer (Thrift compact metadata,
   PLAIN pages, `DATE` logical type) writes one `user_daily` file per thread and one
   `campaign_daily` file.

## Correctness

* `make fixtures-native` runs all 36 edge-case fixtures (36/36 pass).
* On the 1M-event dev set (seed 1), the output matches `dev-answer` exactly.

## Run locally

```sh
make native                       # g++ >= 11, x86-64-v3
make fixtures-native              # 36 fixtures against the host binary
DATA_DIR=dev OUTPUT_DIR=out build/pipeline && python3 scripts/compare.py out dev-answer

make build && make fixtures       # the Docker image, run with the scorer's flags
```

Set `VERBOSE=1` to get per-phase timings on stderr.
