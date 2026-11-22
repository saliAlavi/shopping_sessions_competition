# Shopping Sessions: native C++ pipeline

An entry for the Databricks **Shopping Sessions** track of the Builders Buildathon.
It turns the messy click-stream CSVs under `/data` into the two exact daily Parquet
tables, `/output/user_daily/` and `/output/campaign_daily/`, following `rules.md`.

The whole pipeline is one dependency-free C++17 file, `src/main.cpp`, compiled to a
static binary (x86-64-v3). It uses no CSV library, no Parquet library and no query
engine. The run image is `FROM scratch` and holds only that binary.

## How it works

Every phase runs on all `NCPU` threads, with dynamic work distribution.

1. **Map.** Every input file is `mmap`ed into one reserved virtual region, with zero
   padding after each file, so field parsing never copies and may read a little past
   any field safely.
2. **Links.** Link lines go into a lock-free open-addressing hash table of user IDs
   (64-bit hash to dense id). The table is pre-faulted with sequential parallel
   sweeps. A lock-free union-find with CAS linking and path halving joins the IDs.
   Each person is named by the byte-wise smallest of its IDs, found with a CAS-min
   on 8-byte big-endian prefixes. Products, campaigns and size estimates are loaded
   on one thread while the others parse links.
3. **Events, then clicks.**
   * A line-anchored AVX2 splitter classifies the 128 bytes at each line start once
     and peels the delimiters off a two-word mask. Fields split on `,` and `\n` only
     (no value may hold either). Quotes are checked arithmetically: each field is
     either unquoted or `"..."`, and the line's quote count must equal twice the
     number of quoted fields. That rule is exact under the spec. Lines longer than
     128 bytes take an exact scalar path.
   * The input text is software-prefetched about 1.5 KB ahead, because the hardware
     prefetcher restarts at every 4 KB page of the mapped files.
   * Timestamps are validated and converted with SIMD (`pshufb`/`maddubs`) and a
     table indexed by (year, month).
   * User-table lookups are software-pipelined: they are prefetched 7 lines ahead.
     The table is kept under 25% full, since a first-probe miss is a mispredicted
     branch.
   * Each event copy becomes a 24-byte record: the event_id hash, `ingest_ts`, the
     person, and packed `ts | type | category | price`, or `INVALID`. Records are
     scattered by event_id hash through **write-combining staging buffers flushed
     with non-temporal stores**.
   * Clicks become 16-byte records scattered by person. Click copies are identical,
     so dedup cannot change a result and is skipped.
   * Partition capacities come from a histogram of the event pass, and overflow
     goes to fixed blocks.
4. **Dedup.** In each event_id partition, a small hash table keeps the copy with
   the smallest `ingest_ts` and marks the other copies, and the kept copy survives
   only if it is valid. Survivors stream into person partitions, which are ID ranges.
   Their blocks reuse the memory of event partitions that are already processed.
5. **Sessions.** Within each person partition:
   * **No exact sort of events is needed.** Sessions never split inside a 2^20 ms
     (17.5 min < 30 min) bucket, so one counting pass on `ts >> 20` and a stable
     person scatter are enough. The session walk treats a bucket as a unit, and a
     gap is the next bucket's minimum minus the running maximum.
   * Clicks get an exact LSD radix sort per partition, then a stable scatter by
     person.
   * A two-pointer 7-day window gives the credited click (latest `ts`, then the
     smallest click_id) and the distinct assisting campaigns.
   * Rows are aggregated per (person, day) straight into Parquet column buffers, and
     per (campaign, day) counters are kept per thread.
6. **Output.** A minimal hand-written Parquet writer (Thrift compact metadata,
   PLAIN pages, `DATE` logical type) writes one `user_daily` file per thread with
   `writev`. All large mappings are released with parallel `MADV_DONTNEED` before
   `_exit`, so process teardown is not serial.

## Correctness

* All 36 edge-case fixtures pass (`make fixtures-native`).
* The 1M-event dev set (seed 1) matches `dev-answer` exactly.
* Five more seeds, including a window containing 2028-02-29, match an independent
  pure-Python reference implementation of `rules.md` exactly.
* On 60M events, the output is identical across very different thread counts,
  partition counts and buffer capacities, including runs that force every overflow
  path.
* `make fuzz` checks the fast line splitter against the exact scalar splitter on
  27 million random lines (stray quotes, wrong field counts, `\r`, long lines).

## Run locally

```sh
make native                       # g++ >= 11
make fixtures-native              # 36 fixtures against the host binary
DATA_DIR=dev OUTPUT_DIR=out build/pipeline && python3 scripts/compare.py out dev-answer

make build && make fixtures       # the Docker image, run with the scorer's flags
```

Set `VERBOSE=1` to get per-phase timings on stderr.
