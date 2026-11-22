// Shopping Sessions: native C++ pipeline.
//
// Phases (all parallel over NCPU threads):
//   0. mmap every input file into one padded virtual buffer (zero-copy)
//   1. links  -> concurrent user-ID hash table + edge lists
//   2. lock-free union-find; each person named by its byte-wise smallest ID
//   3. events -> 24-byte records scattered by event_id hash
//      clicks -> 16-byte records scattered by person
//   4. dedup each event_id partition (smallest ingest_ts wins, then validity)
//      and scatter the survivors by person
//   5. per person partition: sort, sessionize, credit, assists, aggregate
//   6. write Parquet (hand-rolled writer, one file per thread)
#include <immintrin.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std;
typedef uint64_t u64;
typedef uint32_t u32;
typedef int64_t i64;
typedef int32_t i32;
typedef uint8_t u8;
typedef uint16_t u16;

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

static double now_s() {
  return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}
static double T0;
static bool VERBOSE = false;
#define LOG(...)                                         \
  do {                                                   \
    if (VERBOSE) {                                       \
      fprintf(stderr, "[%7.3f] ", now_s() - T0);         \
      fprintf(stderr, __VA_ARGS__);                      \
      fputc('\n', stderr);                               \
    }                                                    \
  } while (0)

[[noreturn]] static void die(const char* m) {
  fprintf(stderr, "fatal: %s\n", m);
  exit(1);
}

// ------------------------------------------------------------------ memory
static vector<pair<char*, size_t>> REGIONS;  // big mappings, released in parallel before exit
static void* vmalloc(size_t n, bool huge) {
  n = (n + 4095) & ~(size_t)4095;
  if (n == 0) n = 4096;
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) die("mmap anon");
  madvise(p, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  if (n >= (1u << 20)) {
    static atomic_flag lk = ATOMIC_FLAG_INIT;
    while (lk.test_and_set(memory_order_acquire)) _mm_pause();
    REGIONS.push_back({(char*)p, n});
    lk.clear(memory_order_release);
  }
  return p;
}

static int NT = 8;
template <class F>
static void parallel(F f) {
  vector<thread> th;
  for (int t = 1; t < NT; t++) th.emplace_back([&, t] { f(t); });
  f(0);
  for (auto& x : th) x.join();
}

// ------------------------------------------------------------------ hashing
static inline u64 ld64(const char* p) {
  u64 v;
  memcpy(&v, p, 8);
  return v;
}
static inline u32 ld32(const char* p) {
  u32 v;
  memcpy(&v, p, 4);
  return v;
}
static inline u64 mum(u64 a, u64 b) {
  __uint128_t r = (__uint128_t)a * b;
  return (u64)r ^ (u64)(r >> 64);
}
// Reads up to 7 bytes past the end of the string: every buffer is padded.
__attribute__((always_inline)) static inline u64 hash_str(const char* p, u32 n) {
  const u64 s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull, s2 = 0x8ebc6af09c88c6e3ull,
            s3 = 0x589965cc75374cc3ull;
  u64 seed = s0 ^ ((u64)n * s3);
  u64 a, b;
  if (LIKELY(n <= 16)) {
    if (n <= 8) {
      a = n ? (ld64(p) & (~0ull >> (64 - 8 * n))) : 0;
      b = 0;
    } else {
      a = ld64(p);
      b = ld64(p + 8) & (~0ull >> (128 - 8 * n));
    }
    return mum(a ^ s1, b ^ seed);
  } else {
    u32 i = n;
    const char* q = p;
    while (i > 16) {
      seed = mum(ld64(q) ^ s1, ld64(q + 8) ^ seed);
      q += 16;
      i -= 16;
    }
    // last 16 bytes (overlapping), always in bounds
    a = ld64(p + n - 16);
    b = ld64(p + n - 8);
  }
  return mum(s1 ^ n, mum(a ^ s1, b ^ seed));
}

// ------------------------------------------------------------------ CSV
// First ',', '\n' or '"' at or after p. Needs 32 readable bytes past the line end.
static inline const char* fdelim(const char* p) {
  const __m256i c1 = _mm256_set1_epi8(','), c2 = _mm256_set1_epi8('\n'), c3 = _mm256_set1_epi8('"');
  for (;;) {
    __m256i v = _mm256_loadu_si256((const __m256i*)p);
    u32 m = (u32)_mm256_movemask_epi8(_mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(v, c1), _mm256_cmpeq_epi8(v, c2)), _mm256_cmpeq_epi8(v, c3)));
    if (m) return p + __builtin_ctz(m);
    p += 32;
  }
}
static inline const char* next_line(const char* p) {
  const __m256i c2 = _mm256_set1_epi8('\n');
  for (;;) {
    __m256i v = _mm256_loadu_si256((const __m256i*)p);
    u32 m = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, c2));
    if (m) return p + __builtin_ctz(m) + 1;
    p += 32;
  }
}

struct Fld {
  const char* p;
  u32 n;
};

// Split one line into exactly K fields. Returns the start of the next line;
// ok=false when the line is malformed.
template <int K>
static inline const char* split(const char* p, Fld* f, bool& ok) {
  const char* q = p;
  for (int i = 0; i < K; i++) {
    if (*q == '"') {
      const char* e = fdelim(q + 1);
      if (UNLIKELY(*e != '"')) {
        ok = false;
        return next_line(e);
      }
      f[i].p = q + 1;
      f[i].n = (u32)(e - q - 1);
      q = e + 1;
    } else {
      const char* e = fdelim(q);
      if (UNLIKELY(*e == '"')) {
        ok = false;
        return next_line(e);
      }
      f[i].p = q;
      f[i].n = (u32)(e - q);
      q = e;
    }
    char c = *q;
    if (i < K - 1) {
      if (UNLIKELY(c != ',')) {
        ok = false;
        return next_line(q);
      }
      q++;
    } else {
      if (UNLIKELY(c != '\n')) {
        ok = false;
        return next_line(q);
      }
      ok = true;
      return q + 1;
    }
  }
  ok = false;
  return next_line(q);
}

// Continuous delimiter iterator: every byte of a chunk is classified once,
// 64 bytes at a time; next() yields the positions of ',', '\n' and '"' in order.
struct DIt {
  const char* blk;
  u64 m;
  static inline u64 scan(const char* b) {
    const __m256i c1 = _mm256_set1_epi8(','), c2 = _mm256_set1_epi8('\n'), c3 = _mm256_set1_epi8('"');
    __m256i v0 = _mm256_loadu_si256((const __m256i*)b), v1 = _mm256_loadu_si256((const __m256i*)(b + 32));
    u32 a = (u32)_mm256_movemask_epi8(_mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(v0, c1), _mm256_cmpeq_epi8(v0, c2)), _mm256_cmpeq_epi8(v0, c3)));
    u32 b2 = (u32)_mm256_movemask_epi8(_mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(v1, c1), _mm256_cmpeq_epi8(v1, c2)), _mm256_cmpeq_epi8(v1, c3)));
    return (u64)a | ((u64)b2 << 32);
  }
  inline void init(const char* p) {
    blk = p;
    m = scan(p);
  }
  inline const char* next() {
    while (UNLIKELY(m == 0)) {
      blk += 64;
      m = scan(blk);
    }
    const char* r = blk + __builtin_ctzll(m);
    m &= m - 1;
    return r;
  }
};

// Split the line starting at p into exactly K fields using the iterator.
// On return p is the start of the next line. false = malformed line.
template <int K>
static inline bool split_it(const char*& p, DIt& it, Fld* f) {
  const char* s = p;
  const char* d;
  for (int i = 0; i < K; i++) {
    d = it.next();
    if (*s == '"') {
      // d == s: the opening quote
      const char* q = it.next();
      if (UNLIKELY(*q != '"')) {
        d = q;
        goto bad;
      }
      f[i].p = s + 1;
      f[i].n = (u32)(q - s - 1);
      d = it.next();
      if (UNLIKELY(d != q + 1)) goto bad;
    } else {
      if (UNLIKELY(*d == '"')) goto bad;
      f[i].p = s;
      f[i].n = (u32)(d - s);
    }
    if (i < K - 1) {
      if (UNLIKELY(*d != ',')) goto bad;
      s = d + 1;
    } else {
      if (UNLIKELY(*d != '\n')) goto bad;
      p = d + 1;
      return true;
    }
  }
bad:
  while (*d != '\n') d = it.next();
  p = d + 1;
  return false;
}

// Delimiter iterator over ',' and '\n' only, plus a running count of '"'.
// Fields are split by commas alone (no value holds a comma), quotes are checked
// arithmetically: each field is either unquoted with no '"' or "...", and the
// line's quote count must be exactly twice the number of quoted fields.
struct DIt2 {
  const char* blk;
  u64 m, q;
  i64 qcum;
  static inline void scan(const char* b, u64& d, u64& qq) {
    const __m256i c1 = _mm256_set1_epi8(','), c2 = _mm256_set1_epi8('\n'), c3 = _mm256_set1_epi8('"');
    __m256i v0 = _mm256_loadu_si256((const __m256i*)b), v1 = _mm256_loadu_si256((const __m256i*)(b + 32));
    u32 d0 = (u32)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(v0, c1), _mm256_cmpeq_epi8(v0, c2)));
    u32 d1 = (u32)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(v1, c1), _mm256_cmpeq_epi8(v1, c2)));
    u32 q0 = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, c3));
    u32 q1 = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, c3));
    d = (u64)d0 | ((u64)d1 << 32);
    qq = (u64)q0 | ((u64)q1 << 32);
  }
  inline void init(const char* p) {
    blk = p;
    qcum = 0;
    scan(p, m, q);
  }
  inline const char* next() {
    while (UNLIKELY(m == 0)) {
      qcum += __builtin_popcountll(q);
      blk += 64;
      scan(blk, m, q);
    }
    const char* r = blk + __builtin_ctzll(m);
    m &= m - 1;
    return r;
  }
  // quotes in [init position, x), x inside the current block
  inline i64 qbefore(const char* x) const {
    u32 o = (u32)(x - blk);
    return qcum + __builtin_popcountll(q & ((1ull << o) - 1));
  }
};

template <int K>
static inline bool split2(const char*& p, DIt2& it, i64& qstart, Fld* __restrict f) {
  // iterator state in registers (f[] stores cannot alias it)
  const char* blk = it.blk;
  u64 m = it.m, q = it.q;
  i64 qcum = it.qcum;
  const char* ls = p;
  const char* s = p;
  const char* d = nullptr;
  u32 bad = 0;
  u32 nq = 0;
  u8 cor = 0;
#pragma GCC unroll 8
  for (int i = 0; i < K; i++) {
    while (UNLIKELY(m == 0)) {
      qcum += __builtin_popcountll(q);
      blk += 64;
      DIt2::scan(blk, m, q);
    }
    d = blk + __builtin_ctzll(m);
    m &= m - 1;
    // delimiters are only ',' (0x2C) or '\n' (0x0A): OR-ing the separators must give exactly ','
    if (i < K - 1)
      cor |= (u8)*d;
    else
      bad |= (u32)(*d != '\n');
    u32 qs = (u32)(s[0] == '"');
    u32 qe = (u32)(d[-1] == '"');
    u32 len = (u32)(d - s);
    bad |= (qs ^ qe) | (qs & (u32)(len < 2));
    f[i].p = s + qs;
    f[i].n = len - 2 * qs;
    nq += qs;
    s = d + 1;
  }
  if (K > 1) bad |= (u32)(cor != ',');
  u32 o = (u32)(d - blk);
  i64 qend = qcum + __builtin_popcountll(q & ((1ull << o) - 1));
  bad |= (u32)((qend - qstart) != (i64)(2 * nq));
  it.blk = blk;
  it.m = m;
  it.q = q;
  it.qcum = qcum;
  if (LIKELY(!bad)) {
    qstart = qend;
    p = d + 1;
    return true;
  }
  // malformed (or misaligned by a wrong field count): resync at the true line end
  const char* le = next_line(ls);
  it.init(le);
  qstart = 0;
  p = le;
  return false;
}

// ------------------------------------------------------------------ timestamps
static i32 YDAYS[131];      // days from 1970-01-01 to Jan 1 of 1970 + i
static u16 MDAYS[2][13];    // days before month m
static i32 YMD[131 * 16];  // days from 1970-01-01 to (1970+i, m, 1)
static u8 YDIM[131 * 16];  // days in month
static void init_date_tables2();
static void init_date_tables() {
  init_date_tables2();
  i32 acc = 0;
  for (int i = 0; i < 131; i++) {
    YDAYS[i] = acc;
    int y = 1970 + i;
    acc += ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 366 : 365;
  }
  static const int dm[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  for (int l = 0; l < 2; l++) {
    int s = 0;
    for (int m = 1; m <= 12; m++) {
      MDAYS[l][m] = (u16)s;
      s += dm[m] + (m == 2 ? l : 0);
    }
  }
}
static inline i64 days_from_civil(i64 y, unsigned m, unsigned d);
static void init_date_tables2() {
  static const int dm[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  i32 acc = 0;
  for (int i = 0; i < 131; i++) {
    int y = 1970 + i;
    bool lp = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    for (int m = 1; m <= 12; m++) {
      YMD[i * 16 + m] = acc;
      YDIM[i * 16 + m] = (u8)(dm[m] + (m == 2 && lp));
      acc += YDIM[i * 16 + m];
    }
  }
}
static inline i64 civil_days(unsigned y, unsigned m, unsigned d, bool lp) {
  if (LIKELY(y >= 1970 && y < 2101)) return YDAYS[y - 1970] + MDAYS[lp][m] + d - 1;
  return days_from_civil(y, m, d);
}

static inline i64 days_from_civil(i64 y, unsigned m, unsigned d) {
  y -= m <= 2;
  const i64 era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (i64)doe - 719468;
}
static inline bool isdig(char c) { return (unsigned)(c - '0') < 10; }
static inline unsigned d2(const char* p) { return (unsigned)(p[0] - '0') * 10 + (unsigned)(p[1] - '0'); }
static inline bool leap(unsigned y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static const u8 DIM[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

// Parse a ts field (F1, F2 or F3) into epoch ms. false = bad timestamp.
static inline bool parse_ts(const char* p, u32 n, i64& out) {
  if ((n == 20 || n == 25) && p[4] == '-') {
    if (p[7] != '-' || p[10] != 'T' || p[13] != ':' || p[16] != ':') return false;
    if (!(isdig(p[0]) && isdig(p[1]) && isdig(p[2]) && isdig(p[3]) && isdig(p[5]) && isdig(p[6]) &&
          isdig(p[8]) && isdig(p[9]) && isdig(p[11]) && isdig(p[12]) && isdig(p[14]) && isdig(p[15]) &&
          isdig(p[17]) && isdig(p[18])))
      return false;
    unsigned y = d2(p) * 100 + d2(p + 2), mo = d2(p + 5), d = d2(p + 8), h = d2(p + 11), mi = d2(p + 14),
             s = d2(p + 17);
    if (mo < 1 || mo > 12 || d < 1 || h > 23 || mi > 59 || s > 59) return false;
    unsigned dim = DIM[mo] + (mo == 2 && leap(y));
    if (d > dim) return false;
    i64 off = 0;
    if (n == 20) {
      if (p[19] != 'Z') return false;
    } else {
      char sg = p[19];
      if ((sg != '+' && sg != '-') || p[22] != ':' || !isdig(p[20]) || !isdig(p[21]) || !isdig(p[23]) ||
          !isdig(p[24]))
        return false;
      unsigned oh = d2(p + 20), om = d2(p + 23);
      if (oh > 23 || om > 59) return false;
      off = (i64)(oh * 60 + om) * 60;
      if (sg == '-') off = -off;
    }
    out = (days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s - off) * 1000;
    return true;
  }
  if (n == 0) return false;
  u32 i = 0;
  while (i < n && p[i] == '0') i++;
  if (n - i > 18) return false;  // a non-digit is bad, a huge number is outside the window
  i64 v = 0;
  for (; i < n; i++) {
    unsigned c = (unsigned)(p[i] - '0');
    if (c > 9) return false;
    v = v * 10 + c;
  }
  out = v;
  return true;
}
// ingest_ts is always valid F1: epoch seconds.
__attribute__((always_inline)) static inline i64 parse_f1_sec(const char* p) {
  __m128i t = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)p), _mm_set1_epi8('0'));
  __m128i g = _mm_shuffle_epi8(t, _mm_setr_epi8(0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1));
  __m128i pr = _mm_maddubs_epi16(g, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0));
  unsigned y = (unsigned)_mm_extract_epi16(pr, 0) * 100u + (unsigned)_mm_extract_epi16(pr, 1);
  unsigned mo = (unsigned)_mm_extract_epi16(pr, 2), d = (unsigned)_mm_extract_epi16(pr, 3);
  unsigned h = (unsigned)_mm_extract_epi16(pr, 4), mi = (unsigned)_mm_extract_epi16(pr, 5);
  unsigned s = (unsigned)(p[17] - '0') * 10 + (unsigned)(p[18] - '0');
  unsigned yi = y - 1970;
  if (LIKELY(yi < 131 && mo - 1 < 12)) return (i64)(YMD[yi * 16 + mo] + (i32)d - 1) * 86400 + h * 3600 + mi * 60 + s;
  return civil_days(y, mo, d, leap(y)) * 86400 + h * 3600 + mi * 60 + s;
}

// ---- SIMD versions
alignas(16) static const u8 KEEPMASK[32] = {0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                            0,    0,    0,    0,    0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
// F3: n digits (1..16) right-aligned; reads 16 bytes ending at p+n.
__attribute__((always_inline)) static inline bool f3_simd(const char* p, u32 n, i64& out) {
  __m128i v = _mm_loadu_si128((const __m128i*)(p + n - 16));
  __m128i t = _mm_sub_epi8(v, _mm_set1_epi8('0'));
  __m128i keep = _mm_loadu_si128((const __m128i*)(KEEPMASK + n));
  t = _mm_and_si128(t, keep);
  if (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(t, _mm_set1_epi8(9)), t)) != 0xFFFF) return false;
  __m128i m1 = _mm_maddubs_epi16(t, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1));
  __m128i m2 = _mm_madd_epi16(m1, _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1));
  __m128i m3 = _mm_packus_epi32(m2, m2);
  __m128i m4 = _mm_madd_epi16(m3, _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1));
  u64 hi = (u32)_mm_cvtsi128_si32(m4), lo = (u32)_mm_extract_epi32(m4, 1);
  out = (i64)(hi * 100000000ull + lo);
  return true;
}
// "YYYY-MM-DDTHH:MM:SS" at p, validated; returns local seconds or -1.
// digits at 0-3,5,6,8,9,11,12,14,15,17,18; separators at 4,7,10,13,16
__attribute__((always_inline)) static inline i64 dt19_simd(const char* p) {
  __m256i v = _mm256_loadu_si256((const __m256i*)p);
  __m256i t = _mm256_sub_epi8(v, _mm256_set1_epi8('0'));
  u32 dig = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_min_epu8(t, _mm256_set1_epi8(9)), t));
  const __m256i sep = _mm256_setr_epi8(0, 0, 0, 0, '-', 0, 0, '-', 0, 0, 'T', 0, 0, ':', 0, 0, ':', 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  u32 sp = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, sep));
  const u32 DIGM = 0x6DB6Fu;  // bits 0-3,5,6,8,9,11,12,14,15,17,18
  const u32 SEPM = 0x12490u;  // bits 4,7,10,13,16
  if (((dig & DIGM) | (sp & SEPM)) != (DIGM | SEPM)) return -1;
  // pairs: (0,1)(2,3)(5,6)(8,9)(11,12)(14,15) from the low 16 bytes
  __m128i lo = _mm256_castsi256_si128(t);
  __m128i g = _mm_shuffle_epi8(lo, _mm_setr_epi8(0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1));
  __m128i pr = _mm_maddubs_epi16(g, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0));
  unsigned y = (unsigned)_mm_extract_epi16(pr, 0) * 100u + (unsigned)_mm_extract_epi16(pr, 1);
  unsigned mo = (unsigned)_mm_extract_epi16(pr, 2), d = (unsigned)_mm_extract_epi16(pr, 3);
  unsigned h = (unsigned)_mm_extract_epi16(pr, 4), mi = (unsigned)_mm_extract_epi16(pr, 5);
  unsigned s = (unsigned)(p[17] - '0') * 10 + (unsigned)(p[18] - '0');
  if (mo - 1 > 11 || h > 23 || mi > 59 || s > 59) return -1;
  unsigned yi = y - 1970;
  if (LIKELY(yi < 131)) {
    unsigned ix = yi * 16 + mo;
    if (d - 1 >= YDIM[ix]) return -1;
    return (i64)(YMD[ix] + (i32)d - 1) * 86400 + h * 3600 + mi * 60 + s;
  }
  bool lp = leap(y);
  if (d - 1 >= (unsigned)(DIM[mo] + (mo == 2 && lp))) return -1;
  return civil_days(y, mo, d, lp) * 86400 + h * 3600 + mi * 60 + s;
}
__attribute__((always_inline)) static inline bool parse_ts_fast(const char* p, u32 n, i64& out) {
  if (n == 20) {
    if (p[4] == '-') {
      if (p[19] != 'Z') return false;
      i64 s = dt19_simd(p);
      if (s < 0) return false;
      out = s * 1000;
      return true;
    }
  } else if (n == 25) {
    if (p[4] == '-') {
      char sg = p[19];
      if ((sg != '+' && sg != '-') || p[22] != ':' || !isdig(p[20]) || !isdig(p[21]) || !isdig(p[23]) ||
          !isdig(p[24]))
        return false;
      unsigned oh = d2(p + 20), om = d2(p + 23);
      if (oh > 23 || om > 59) return false;
      i64 s = dt19_simd(p);
      if (s < 0) return false;
      i64 off = (i64)(oh * 60 + om) * 60;
      if (sg == '-') off = -off;
      out = (s - off) * 1000;
      return true;
    }
  }
  if (n >= 1 && n <= 16) return f3_simd(p, n, out);
  return parse_ts(p, n, out);
}

// ------------------------------------------------------------------ files
struct File {
  string path;
  size_t size = 0;  // effective size (ends with '\n' when non-empty)
  char* data = nullptr;
};
static char* BASE = nullptr;

static vector<File> list_csv(const string& dir) {
  vector<File> out;
  DIR* d = opendir(dir.c_str());
  if (!d) return out;
  while (dirent* e = readdir(d)) {
    string n = e->d_name;
    if (n.size() < 4 || n[0] == '.' || n.compare(n.size() - 4, 4, ".csv") != 0) continue;
    string full = dir + "/" + n;
    struct stat st;
    if (stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
    File f;
    f.path = full;
    f.size = (size_t)st.st_size;
    out.push_back(f);
  }
  closedir(d);
  sort(out.begin(), out.end(), [](const File& a, const File& b) { return a.path < b.path; });
  return out;
}

// Map all files into one reserved region; each file is followed by >= 64 zero bytes.
static void map_files(vector<File*>& fs) {
  size_t total = 4096;
  for (auto* f : fs) total += ((f->size + 64 + 4095) & ~(size_t)4095);
  BASE = (char*)vmalloc(total, false);
  size_t off = 0;
  for (auto* f : fs) {
    f->data = BASE + off;
    if (f->size) {
      int fd = open(f->path.c_str(), O_RDONLY);
      if (fd < 0) die("open");
      void* p = mmap(BASE + off, f->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, 0);
      if (p == MAP_FAILED) die("mmap file");
      close(fd);
      if (f->data[f->size - 1] != '\n') {
        f->data[f->size] = '\n';
        f->size++;
      }
    }
    off += ((f->size + 64 + 4095) & ~(size_t)4095);
  }
}

static string read_small(const string& path) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) return string();
  string s;
  char buf[65536];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, fp)) > 0) s.append(buf, k);
  fclose(fp);
  return s;
}

// ------------------------------------------------------------------ small string maps
struct SMap {
  vector<u64> keys;
  vector<u32> vals;
  vector<string> strs;
  u64 mask = 0;
  void init(size_t n) {
    size_t c = 16;
    while (c < n * 2) c <<= 1;
    keys.assign(c, 0);
    vals.assign(c, ~0u);
    strs.assign(c, string());
    mask = c - 1;
  }
  // returns existing value or inserts v
  u32 insert(const char* p, u32 n, u32 v) {
    u64 h = hash_str(p, n) | 1;
    u64 i = (h >> 17) & mask;
    for (;;) {
      if (keys[i] == 0) {
        keys[i] = h;
        vals[i] = v;
        strs[i].assign(p, n);
        return v;
      }
      if (keys[i] == h && strs[i].size() == n && memcmp(strs[i].data(), p, n) == 0) return vals[i];
      i = (i + 1) & mask;
    }
  }
  inline u32 find(const char* p, u32 n) const {
    u64 h = hash_str(p, n) | 1;
    u64 i = (h >> 17) & mask;
    for (;;) {
      u64 k = keys[i];
      if (k == 0) return ~0u;
      if (k == h && strs[i].size() == n && memcmp(strs[i].data(), p, n) == 0) return vals[i];
      i = (i + 1) & mask;
    }
  }
};

// ------------------------------------------------------------------ globals
static i64 WS, WE, CWS;  // window start/end, click window start (ms)
static const i64 D7 = 7LL * 86400000;
static const i64 GAP = 1800000;
static i64 DAY0;  // epoch day of WS
static int ND;    // number of days

static SMap PROD;  // product -> category index
static SMap CAMP;  // campaign -> index

// Fast exact lookup for IDs of the form <prefix><decimal without leading zero>.
// If every ID in the file has that form, a string not of that form cannot match.
struct DirectIds {
  bool ok = false;
  string pre;
  vector<u32> val;  // number -> value, ~0u if absent
  static bool canon(const char* p, u32 n, const string& pre, u64& num) {
    u32 k = (u32)pre.size();
    if (n <= k || n > k + 9 || memcmp(p, pre.data(), k) != 0) return false;
    if (p[k] == '0') return false;
    u64 v = 0;
    for (u32 i = k; i < n; i++) {
      unsigned c = (unsigned)(p[i] - '0');
      if (c > 9) return false;
      v = v * 10 + c;
    }
    num = v;
    return true;
  }
  void build(const vector<pair<string, u32>>& ids, const string& prefix) {
    pre = prefix;
    u64 mx = 0;
    for (auto& e : ids) {
      u64 v;
      if (!canon(e.first.data(), (u32)e.first.size(), pre, v)) return;
      mx = max(mx, v);
    }
    if (mx > 50000000) return;
    val.assign(mx + 1, ~0u);
    for (auto& e : ids) {
      u64 v;
      canon(e.first.data(), (u32)e.first.size(), pre, v);
      if (val[v] == ~0u) val[v] = e.second;
    }
    ok = true;
  }
  // returns 1 found (out set), 0 absent, -1 unknown (fall back to the map)
  inline int find(const char* p, u32 n, u32& out) const {
    if (!ok) return -1;
    u64 v;
    if (!canon(p, n, pre, v) || v >= val.size()) return 0;
    out = val[v];
    return out != ~0u;
  }
};
static DirectIds PRODD, CAMPD;
static inline u32 prod_cat(const char* p, u32 n) {
  u32 v;
  int r = PRODD.find(p, n, v);
  if (r < 0) return PROD.find(p, n);
  return r ? v : ~0u;
}
static inline u32 camp_idx(const char* p, u32 n) {
  u32 v;
  int r = CAMPD.find(p, n, v);
  if (r < 0) return CAMP.find(p, n);
  return r ? v : ~0u;
}
static vector<string> CATS;
static vector<string> CATLP;  // length-prefixed category names
static vector<string> CHANNELS, CAMPIDS;
static u32 UNKNOWN_CAT;
static u32 NCAMP;

// user table
struct Slot {
  atomic<u64> key;
  atomic<u32> pid;  // person id + 1; 0 while pending
  u32 pad;
};
static Slot* UT;
static u64 UMASK;
static int USHIFT;
static int UBITS;
static atomic<u32> NEXT_ID{0};
static const char** NAMEP;
static u32* NAMEN;
static u32* SLOT_OF;
static u64* PFX;  // big-endian 8-byte prefix of each link-phase id, for fast byte-wise compares
static u64 MAXID;

struct IdAlloc {
  u32 cur = 0, end = 0;
  inline u32 next() {
    if (UNLIKELY(cur == end)) {
      cur = NEXT_ID.fetch_add(1024);
      end = cur + 1024;
      if ((u64)end > MAXID) die("id space exhausted");
    }
    return cur++;
  }
};

template <bool LINK>
static inline u32 uget(u64 h, const char* p, u32 n, IdAlloc& al) {
  u64 i = h >> USHIFT;
  for (;;) {
    Slot& s = UT[i];
    u64 k = s.key.load(memory_order_acquire);
    if (k == h) {
      u32 v;
      while ((v = s.pid.load(memory_order_acquire)) == 0) _mm_pause();
      return v - 1;
    }
    if (k == 0) {
      u64 e = 0;
      if (s.key.compare_exchange_strong(e, h, memory_order_acq_rel)) {
        u32 id = al.next();
        NAMEP[id] = p;
        NAMEN[id] = n;
        if (LINK) {
          SLOT_OF[id] = (u32)i;
          u64 w = n >= 8 ? ld64(p) : (n ? (ld64(p) & (~0ull >> (64 - 8 * n))) : 0);
          PFX[id] = __builtin_bswap64(w);
        }
        s.pid.store(id + 1, memory_order_release);
        return id;
      }
      if (e == h) continue;
    }
    i = (i + 1) & UMASK;
  }
}

static inline bool name_less(u32 a, u32 b) {
  u32 na = NAMEN[a], nb = NAMEN[b];
  int c = memcmp(NAMEP[a], NAMEP[b], na < nb ? na : nb);
  if (c != 0) return c < 0;
  return na < nb;
}

// ------------------------------------------------------------------ record buffers
// thread-local bump arena for overflow blocks (released with the other regions)
static thread_local char* AR_P = nullptr;
static thread_local size_t AR_LEFT = 0;
static void* arena_alloc(size_t n) {
  n = (n + 63) & ~(size_t)63;
  if (AR_LEFT < n) {
    size_t sz = max(n, (size_t)64 << 20);
    AR_P = (char*)vmalloc(sz, true);
    AR_LEFT = sz;
  }
  void* r = AR_P;
  AR_P += n;
  AR_LEFT -= n;
  return r;
}

template <class T>
struct Buf {
  // a fixed slice, then fixed-size overflow blocks from a thread-local bump arena
  static const u32 OB = 4096;
  T* d = nullptr;
  u32 n = 0, cap = 0;
  vector<T*> ob;
  u32 on = 0;
  inline void push(const T& x) {
    if (LIKELY(n < cap))
      d[n++] = x;
    else
      push_ov(x);
  }
  void push_ov(const T& x) {
    if (ob.empty() || on == OB) {
      ob.push_back((T*)arena_alloc(sizeof(T) * OB));
      on = 0;
    }
    ob.back()[on++] = x;
  }
  inline size_t ovsize() const { return ob.empty() ? 0 : (ob.size() - 1) * (size_t)OB + on; }
  inline size_t size() const { return n + ovsize(); }
  inline T& at(size_t i) { return i < n ? d[i] : ob[(i - n) / OB][(i - n) % OB]; }
  template <class F>
  inline void each(F f) const {
    for (u32 i = 0; i < n; i++) f(d[i]);
    for (size_t b = 0; b < ob.size(); b++) {
      u32 m = b + 1 == ob.size() ? on : OB;
      const T* q = ob[b];
      for (u32 i = 0; i < m; i++) f(q[i]);
    }
  }
};

// Write-combining scatter: G records per partition are staged in a small cache-resident
// area and flushed with non-temporal stores (no read-for-ownership of the destination).
template <class T, int G>
struct WC {
  static_assert((sizeof(T) * G) % 64 == 0, "group must be whole cache lines");
  vector<Buf<T>>* bufs = nullptr;
  T* stage = nullptr;  // [parts][G], 64-byte aligned
  u8* cnt = nullptr;
  size_t parts = 0;
  void init(vector<Buf<T>>& b) {
    bufs = &b;
    parts = b.size();
    stage = (T*)aligned_alloc(64, ((sizeof(T) * G * parts) + 63) & ~(size_t)63);
    cnt = (u8*)calloc(parts, 1);
  }
  inline void push(size_t k, const T& x) {
    u32 c = cnt[k];
    T* sg = stage + k * G;
    sg[c] = x;
    if (UNLIKELY(++c == G)) {
      flush_full(k, sg);
      c = 0;
    }
    cnt[k] = (u8)c;
  }
  void flush_full(size_t k, T* sg) {
    Buf<T>& b = (*bufs)[k];
    if (LIKELY(b.ob.empty() && b.n + G <= b.cap)) {
      const __m256i* src = (const __m256i*)sg;
      __m256i* dst = (__m256i*)(b.d + b.n);
      for (size_t i = 0; i < sizeof(T) * G / 32; i++) _mm256_stream_si256(dst + i, _mm256_load_si256(src + i));
      b.n += G;
    } else {
      for (int i = 0; i < G; i++) b.push(sg[i]);
    }
  }
  void finish() {
    for (size_t k = 0; k < parts; k++) {
      T* sg = stage + k * G;
      for (u32 i = 0; i < cnt[k]; i++) (*bufs)[k].push(sg[i]);
      cnt[k] = 0;
    }
    _mm_sfence();
    free(stage);
    free(cnt);
    stage = nullptr;
    cnt = nullptr;
  }
};

// variable capacity per partition
template <class T>
static void init_bufs_var(vector<Buf<T>>& v, const vector<size_t>& caps) {
  size_t parts = caps.size(), tot = 0;
  v.resize(parts);
  for (size_t c : caps) tot += (c + 15) & ~(size_t)15;
  T* mem = (T*)vmalloc(sizeof(T) * tot + 64, true);
  size_t o = 0;
  for (size_t i = 0; i < parts; i++) {
    size_t c = (caps[i] + 15) & ~(size_t)15;
    v[i].d = mem + o;
    v[i].cap = (u32)c;
    o += c;
  }
}

template <class T>
static void init_bufs(vector<Buf<T>>& v, size_t parts, size_t cap) {
  cap = (cap + 15) & ~(size_t)15;  // whole write-combining groups keep streamed stores aligned
  v.resize(parts);
  T* mem = (T*)vmalloc(sizeof(T) * cap * parts + 64, true);
  for (size_t i = 0; i < parts; i++) {
    v[i].d = mem + i * cap;
    v[i].cap = (u32)cap;
  }
}

struct ERec {  // event copy, partitioned by event_id hash
  u64 eh;
  u32 ingest;
  u32 person;
  u64 val;  // (ts_off << 32) | payload, or INVALID
};
static const u64 INVALID = ~0ull;
struct __attribute__((packed)) PE {  // surviving event, partitioned by person
  u32 person;
  u64 val;
};
struct CRec {  // valid click, partitioned by person
  u32 person;
  u32 ts;    // ms since CWS
  u64 ref;   // (offset of click_id in BASE) << 16 | campaign
};
// payload: [type:2][cat:6][price:24]
static inline u32 mkpay(u32 type, u32 cat, u32 price) { return (type << 30) | (cat << 24) | price; }

static int P1B, P2S;  // log2 event partitions; persons per partition = 1 << P2S (range partitioning)
static u32 P1, P2, P2USED;

struct TL {  // thread-local state
  vector<Buf<ERec>> eb;
  vector<Buf<PE>> pb;
  vector<Buf<CRec>> cb;
  WC<ERec, 8> ebw;
  WC<PE, 16> pbw;
  WC<CRec, 4> cbw;
  vector<pair<u32, u32>> edges;
  vector<u32> phist;  // valid event copies per person partition
  IdAlloc al;
  // outputs
  // user_daily columns, built in place (PLAIN encoding)
  struct BB {
    char* p = nullptr;
    size_t n = 0, cap = 0;
    inline void need(size_t k) {
      if (UNLIKELY(n + k > cap)) {
        size_t c = cap ? cap * 2 : (1 << 16);
        while (c < n + k) c *= 2;
        p = (char*)realloc(p, c);
        cap = c;
      }
    }
    inline void put32(u32 v) { memcpy(p + n, &v, 4); n += 4; }
    inline void put64(u64 v) { memcpy(p + n, &v, 8); n += 8; }
    inline void put(const char* s, size_t k) { memcpy(p + n, s, k); n += k; }
  };
  BB o_uid, o_day, o_sess, o_ev, o_rev, o_top, o_def;
  u64 o_rows = 0;
  struct CC {
    u32 sess, psess, assist, pad;
    i64 rev;
  };
  vector<CC> cc;  // [(camp)*ND + day]
  i64 n_ev_lines = 0, n_cl_lines = 0;
};
static vector<TL> TLS;

// ------------------------------------------------------------------ chunks
struct Chunk {
  File* f;
  size_t b, e;
  int kind;  // 0 links, 1 events, 2 clicks
};

static void make_chunks(vector<File>& fs, int kind, size_t csz, vector<Chunk>& out) {
  for (auto& f : fs) {
    if (!f.size) continue;
    for (size_t b = 0; b < f.size; b += csz) out.push_back(Chunk{&f, b, min(f.size, b + csz), kind});
  }
}

// iterate the lines whose first byte lies in [b, e); the header line is skipped
template <class F>
static inline void for_lines(const Chunk& c, F fn) {
  const char* d = c.f->data;
  const char* p = d + c.b;
  const char* end = d + c.e;
  if (c.b == 0)
    p = next_line(p);
  else if (p[-1] != '\n')
    p = next_line(p);
  while (p < end) p = fn(p);
}

// ------------------------------------------------------------------ links
// first line start at or after chunk begin (the header line is skipped)
static inline const char* chunk_start(const Chunk& c) {
  const char* d = c.f->data;
  const char* p = d + c.b;
  if (c.b == 0 || p[-1] != '\n') p = next_line(p);
  return p;
}

static void parse_links_chunk(const Chunk& c, TL& tl) {
  const char* p = chunk_start(c);
  const char* end = c.f->data + c.e;
  if (p >= end) return;
  DIt2 it;
  it.init(p);
  i64 qs0 = 0;
  struct Pend {
    u64 ha, hb;
    const char *pa, *pb;
    u32 na, nb;
  } ring[8];
  u32 head = 0, tail = 0;
  auto finish = [&](const Pend& q) {
    u32 a = uget<true>(q.ha, q.pa, q.na, tl.al);
    u32 b = uget<true>(q.hb, q.pb, q.nb, tl.al);
    if (a != b) tl.edges.push_back({a, b});
  };
  while (p < end) {
    Fld f[2];
    bool ok = split2<2>(p, it, qs0, f);
    if (!ok || !f[0].n || !f[1].n) continue;
    Pend q;
    q.ha = hash_str(f[0].p, f[0].n) | 1;
    q.hb = hash_str(f[1].p, f[1].n) | 1;
    q.pa = f[0].p;
    q.pb = f[1].p;
    q.na = f[0].n;
    q.nb = f[1].n;
    _mm_prefetch((const char*)&UT[q.ha >> USHIFT], _MM_HINT_T0);
    _mm_prefetch((const char*)&UT[q.hb >> USHIFT], _MM_HINT_T0);
    if (tail - head == 7) finish(ring[head++ & 7]);
    ring[tail++ & 7] = q;
  }
  while (head != tail) finish(ring[head++ & 7]);
}

static atomic<u32>* PARENT;
static atomic<u32>* MINN;
static inline u32 uf_find(u32 x) {
  for (;;) {
    u32 p = PARENT[x].load(memory_order_relaxed);
    if (p == x) return x;
    u32 g = PARENT[p].load(memory_order_relaxed);
    if (g == p) return p;
    PARENT[x].store(g, memory_order_relaxed);
    x = g;
  }
}
static inline void uf_union(u32 a, u32 b) {
  for (;;) {
    a = uf_find(a);
    b = uf_find(b);
    if (a == b) return;
    if (a < b) swap(a, b);
    u32 e = a;
    if (PARENT[a].compare_exchange_strong(e, b, memory_order_acq_rel)) return;
  }
}

// ------------------------------------------------------------------ events / clicks
static void parse_events_chunk(const Chunk& c, TL& tl) {
  const u32 p1shift = 64 - P1B;
  const char* p = chunk_start(c);
  const char* end = c.f->data + c.e;
  if (p >= end) return;
  DIt2 it;
  it.init(p);
  i64 qs0 = 0;
  struct Pend {
    u64 uh;
    const char* up;
    u32 un;
    ERec r;
  } ring[8];
  u32 head = 0, tail = 0;
  auto finish = [&](Pend& q) {
    q.r.person = uget<false>(q.uh, q.up, q.un, tl.al);
    tl.phist[q.r.person >> P2S]++;
    tl.ebw.push(q.r.eh >> p1shift, q.r);
  };
  i64 lines = 0;
  while (p < end) {
    Fld f[7];
    bool ok = split2<7>(p, it, qs0, f);
    lines++;
    if (UNLIKELY(!ok || f[0].n == 0)) continue;
    ERec r;
    r.eh = hash_str(f[0].p, f[0].n);
    r.person = 0;
    r.val = INVALID;
    i64 ing = parse_f1_sec(f[3].p);
    r.ingest = (u32)ing;
    do {
      if (UNLIKELY(f[1].n == 0 || f[5].n == 0)) break;
      i64 ts;
      if (UNLIKELY(!parse_ts_fast(f[2].p, f[2].n, ts))) break;
      if (UNLIKELY(ts < WS || ts >= WE)) break;
      if (UNLIKELY(ing * 1000 - ts > 86400000)) break;
      u32 pay;
      const Fld& ty = f[4];
      if (ty.n == 4 && ld32(ty.p) == 0x77656976u) {  // "view"
        pay = mkpay(0, 0, 0);
      } else if (ty.n == 4 && ld32(ty.p) == 0x74726163u) {  // "cart"
        pay = mkpay(1, 0, 0);
      } else if (ty.n == 8 && ld64(ty.p) == 0x6573616863727570ull) {  // "purchase"
        const Fld& pr = f[6];
        if (pr.n == 0) break;
        u32 i = 0;
        while (i < pr.n && pr.p[i] == '0') i++;
        bool good = true;
        for (u32 k = i; k < pr.n; k++)
          if (!isdig(pr.p[k])) {
            good = false;
            break;
          }
        if (!good || pr.n - i > 8) break;
        u32 v = 0;
        for (u32 k = i; k < pr.n; k++) v = v * 10 + (u32)(pr.p[k] - '0');
        if (v < 1 || v > 10000000) break;
        u32 cat = prod_cat(f[5].p, f[5].n);
        if (cat == ~0u) cat = UNKNOWN_CAT;
        pay = mkpay(2, cat, v);
      } else {
        break;
      }
      r.val = ((u64)(u32)(ts - WS) << 32) | pay;
    } while (0);
    if (r.val == INVALID) {
      tl.ebw.push(r.eh >> p1shift, r);
      continue;
    }
    Pend q;
    q.uh = hash_str(f[1].p, f[1].n) | 1;
    _mm_prefetch((const char*)&UT[q.uh >> USHIFT], _MM_HINT_T0);
    q.up = f[1].p;
    q.un = f[1].n;
    q.r = r;
    if (tail - head == 7) finish(ring[head++ & 7]);
    ring[tail++ & 7] = q;
  }
  while (head != tail) finish(ring[head++ & 7]);
  tl.n_ev_lines += lines;
}

static void parse_clicks_chunk(const Chunk& c, TL& tl) {
  const char* p = chunk_start(c);
  const char* end = c.f->data + c.e;
  if (p >= end) return;
  DIt2 it;
  it.init(p);
  i64 qs0 = 0;
  struct Pend {
    u64 uh;
    const char* up;
    u32 un;
    CRec r;
  } ring[8];
  u32 head = 0, tail = 0;
  auto finish = [&](Pend& q) {
    u32 person = uget<false>(q.uh, q.up, q.un, tl.al);
    q.r.person = person;
    tl.cbw.push(person >> P2S, q.r);
  };
  i64 lines = 0;
  while (p < end) {
    Fld f[4];
    bool ok = split2<4>(p, it, qs0, f);
    lines++;
    if (UNLIKELY(!ok || f[0].n == 0 || f[1].n == 0)) continue;
    i64 ts;
    if (!parse_ts_fast(f[3].p, f[3].n, ts)) continue;
    if (ts < CWS || ts >= WE) continue;
    u32 cmp = camp_idx(f[2].p, f[2].n);
    if (cmp == ~0u) continue;
    Pend q;
    q.uh = hash_str(f[1].p, f[1].n) | 1;
    _mm_prefetch((const char*)&UT[q.uh >> USHIFT], _MM_HINT_T0);
    q.up = f[1].p;
    q.un = f[1].n;
    q.r.ts = (u32)(ts - CWS);
    q.r.ref = ((u64)(f[0].p - BASE) << 16) | cmp;
    if (tail - head == 7) finish(ring[head++ & 7]);
    ring[tail++ & 7] = q;
  }
  while (head != tail) finish(ring[head++ & 7]);
  tl.n_cl_lines += lines;
}

// ------------------------------------------------------------------ sorting helpers
static void radix_sort_ts(u64* a, u64* tmp, size_t n) {
  // sort by the high 32 bits (ts), 3 passes of 11 bits
  for (int pass = 0; pass < 3; pass++) {
    int sh = 32 + pass * 11;
    u32 cnt[2048];
    memset(cnt, 0, sizeof cnt);
    for (size_t i = 0; i < n; i++) cnt[(a[i] >> sh) & 2047]++;
    u32 s = 0;
    for (int i = 0; i < 2048; i++) {
      u32 c = cnt[i];
      cnt[i] = s;
      s += c;
    }
    for (size_t i = 0; i < n; i++) tmp[cnt[(a[i] >> sh) & 2047]++] = a[i];
    memcpy(a, tmp, n * sizeof(u64));
  }
}
static inline void sort_u64(u64* a, size_t n, vector<u64>& tmp) {
  if (n <= 1) return;
  if (n <= 32) {
    for (size_t i = 1; i < n; i++) {
      u64 x = a[i];
      size_t j = i;
      while (j > 0 && a[j - 1] > x) {
        a[j] = a[j - 1];
        j--;
      }
      a[j] = x;
    }
  } else if (n < 512) {
    sort(a, a + n);
  } else {
    if (tmp.size() < n) tmp.resize(n);
    radix_sort_ts(a, tmp.data(), n);
  }
}

struct Cl {
  u32 ts;
  u32 camp;
  u64 off;
};

static inline int clickid_cmp(u64 oa, u64 ob) {
  const char* a = BASE + oa;
  const char* b = BASE + ob;
  u32 na = (u32)(fdelim(a) - a), nb = (u32)(fdelim(b) - b);
  int c = memcmp(a, b, na < nb ? na : nb);
  if (c) return c;
  return na < nb ? -1 : (na > nb ? 1 : 0);
}

// ------------------------------------------------------------------ sessions
struct SessCtx {
  vector<u32> ccnt;     // per campaign count in window
  vector<u32> active;   // active campaign list
  vector<u32> apos;     // position in active list
  vector<i64> catsum;   // per category, current day
  vector<u32> touched;
  vector<u64> sorttmp;
  vector<u64> ev;
  vector<Cl> cl;
  vector<u8> ra, rb;  // radix scratch (12-byte records)
  vector<u32> cnt, ccount;
};

static void process_partition(u32 p, TL& tl, SessCtx& sc, u32 NL) {
  const u32 LMASK = NL - 1;
  // gather events, counting sort by local person
  vector<u32>& cnt = sc.cnt;
  cnt.assign(NL + 1, 0);
  size_t ne = 0;
  for (int t = 0; t < NT; t++) ne += TLS[t].pb[p].size();
  if (ne == 0) return;
  if (sc.ev.size() < ne) sc.ev.resize(ne);
  u64* ev = sc.ev.data();
  // LSD radix sort of (person, ts): three passes on ts, then a stable counting
  // scatter by person, so every person's events come out sorted by ts.
  struct __attribute__((packed)) RE {
    u64 val;
    u32 loc;
  };
  if (sc.ra.size() < ne * sizeof(RE)) {
    sc.ra.resize(ne * sizeof(RE) + 64);
    sc.rb.resize(ne * sizeof(RE) + 64);
  }
  RE* A = (RE*)sc.ra.data();
  RE* B = (RE*)sc.rb.data();
  // Sessions never split inside a 2^20 ms (17.5 min < 30 min) bucket, so events only
  // need grouping by (person, ts >> 20): one counting pass on the bucket, then the
  // stable person scatter. The walk below handles buckets as units.
  static thread_local u32 h0[2048];
  memset(h0, 0, sizeof h0);
  {
    size_t i = 0;
    for (int t = 0; t < NT; t++)
      TLS[t].pb[p].each([&](const PE& x) {
        u64 v = x.val;
        u32 l = x.person & LMASK;
        A[i].val = v;
        A[i].loc = l;
        i++;
        h0[(v >> 52) & 2047]++;
        cnt[l]++;
      });
  }
  {
    u32 a = 0;
    for (int i = 0; i < 2048; i++) {
      u32 x = h0[i];
      h0[i] = a;
      a += x;
    }
  }
  for (size_t i = 0; i < ne; i++) B[h0[(A[i].val >> 52) & 2047]++] = A[i];
  u32 s = 0;
  for (u32 i = 0; i < NL; i++) {
    u32 c = cnt[i];
    cnt[i] = s;
    s += c;
  }
  cnt[NL] = s;
  for (size_t i = 0; i < ne; i++) ev[cnt[B[i].loc]++] = B[i].val;
  // now cnt[i] = end of local i; start = cnt[i-1] (end of previous)

  vector<u32>& cc = sc.ccount;
  cc.assign(NL + 1, 0);
  size_t nc = 0;
  for (int t = 0; t < NT; t++) {
    const Buf<CRec>& b = TLS[t].cb[p];
    b.each([&](const CRec& x) { cc[x.person & LMASK]++; });
    nc += b.size();
  }
  s = 0;
  for (u32 i = 0; i < NL; i++) {
    u32 c = cc[i];
    cc[i] = s;
    s += c;
  }
  cc[NL] = s;
  if (sc.cl.size() < nc) sc.cl.resize(nc);
  Cl* cl = sc.cl.data();
  for (int t = 0; t < NT; t++)
    TLS[t].cb[p].each([&](const CRec& x) {
      cl[cc[x.person & LMASK]++] = Cl{x.ts, (u32)(x.ref & 0xFFFF), x.ref >> 16};
    });

  u32* ccnt = sc.ccnt.data();
  u32* apos = sc.apos.data();
  vector<u32>& active = sc.active;
  i64* catsum = sc.catsum.data();
  vector<u32>& touched = sc.touched;
  TL::CC* CCA = tl.cc.data();

  u32 eprev = 0, cprev = 0;
  for (u32 li = 0; li < NL; li++) {
    u32 eend = cnt[li], cend = cc[li];
    u32 eb = eprev, cb = cprev;
    eprev = eend;
    cprev = cend;
    if (eend == eb) continue;
    u32 person = (p << P2S) | li;
    const char* nptr = NAMEP[person];
    u32 nlen = NAMEN[person];
    u64* E = ev + eb;
    u32 n = eend - eb;
    Cl* C = cl + cb;
    u32 m = cend - cb;
    if (m > 1) {
      if (m <= 16) {
        for (u32 i = 1; i < m; i++) {
          Cl x = C[i];
          u32 j = i;
          while (j > 0 && C[j - 1].ts > x.ts) {
            C[j] = C[j - 1];
            j--;
          }
          C[j] = x;
        }
      } else {
        sort(C, C + m, [](const Cl& a, const Cl& b) { return a.ts < b.ts; });
      }
    }
    // walk sessions
    i32 curday = -1;
    i32 d_sess = 0, d_ev = 0;
    i64 d_rev = 0;
    u32 lo = 0, hi = 0;
    u32 i = 0;
    auto flush = [&]() {
      if (curday < 0) return;
      int8_t top = -1;
      i64 best = 0;
      for (u32 c : touched) {
        i64 v = catsum[c];
        if (v > best || (v == best && v > 0 && (int)c < top)) {
          best = v;
          top = (int8_t)c;
        }
        catsum[c] = 0;
      }
      touched.clear();
      tl.o_uid.need(4 + nlen);
      tl.o_uid.put32(nlen);
      tl.o_uid.put(nptr, nlen);
      tl.o_day.need(4);
      tl.o_day.put32((u32)(i32)(DAY0 + curday));
      tl.o_sess.need(4);
      tl.o_sess.put32((u32)d_sess);
      tl.o_ev.need(4);
      tl.o_ev.put32((u32)d_ev);
      tl.o_rev.need(8);
      tl.o_rev.put64((u64)d_rev);
      if ((tl.o_rows & 7) == 0) {
        tl.o_def.need(1);
        tl.o_def.p[tl.o_def.n++] = 0;
      }
      if (top >= 0) {
        tl.o_def.p[tl.o_def.n - 1] |= (char)(1 << (tl.o_rows & 7));
        const string& cs = CATLP[(size_t)top];
        tl.o_top.need(cs.size());
        tl.o_top.put(cs.data(), cs.size());
      }
      tl.o_rows++;
    };
    // bucket [i, j): events with equal ts >> 20, in any order; returns j, sets min/max
    auto bucket = [&](u32 b0, u32& mn, u32& mx) -> u32 {
      u32 t = (u32)(E[b0] >> 32);
      u32 key = t >> 20;
      mn = mx = t;
      u32 j = b0 + 1;
      while (j < n) {
        u32 u = (u32)(E[j] >> 32);
        if ((u >> 20) != key) break;
        mn = u < mn ? u : mn;
        mx = u > mx ? u : mx;
        j++;
      }
      return j;
    };
    u32 nb_min = 0, nb_max = 0, nb_end = 0;
    if (n) nb_end = bucket(0, nb_min, nb_max);
    while (i < n) {
      // a session starts with the pending bucket [i, nb_end)
      u32 st = nb_min;
      i32 day = (i32)((WS + st) / 86400000 - DAY0);
      if (day != curday) {
        flush();
        curday = day;
        d_sess = 0;
        d_ev = 0;
        d_rev = 0;
      }
      u32 last = nb_max;
      u32 j = nb_end;
      while (j < n) {
        u32 bmn, bmx;
        u32 e2 = bucket(j, bmn, bmx);
        if (bmn - last > GAP) {
          nb_min = bmn;
          nb_max = bmx;
          nb_end = e2;
          break;
        }
        last = bmx;
        j = e2;
      }
      i64 rev = 0;
      bool purch = false;
      for (u32 k = i; k < j; k++) {
        u32 pl = (u32)E[k];
        if (UNLIKELY((pl >> 30) == 2)) {
          u32 price = pl & 0xFFFFFF, cat = (pl >> 24) & 63;
          rev += price;
          purch = true;
          if (catsum[cat] == 0) touched.push_back(cat);
          catsum[cat] += price;
        }
      }
      d_sess++;
      d_ev += (i32)(j - i);
      d_rev += rev;
      i = j;
      // attribution: clicks with st <= c.ts <= st + D7 (click offsets)
      u64 hiT = (u64)st + (u64)D7;
      while (hi < m && C[hi].ts <= hiT) {
        u32 cp = C[hi].camp;
        if (ccnt[cp]++ == 0) {
          apos[cp] = (u32)active.size();
          active.push_back(cp);
        }
        hi++;
      }
      while (lo < hi && C[lo].ts < st) {
        u32 cp = C[lo].camp;
        if (--ccnt[cp] == 0) {
          u32 pos = apos[cp];
          u32 last_c = active.back();
          active[pos] = last_c;
          apos[last_c] = pos;
          active.pop_back();
        }
        lo++;
      }
      u32 credit;
      if (lo < hi) {
        u32 best = hi - 1;
        u32 k = hi - 1;
        while (k > lo && C[k - 1].ts == C[hi - 1].ts) {
          k--;
          if (clickid_cmp(C[k].off, C[best].off) < 0) best = k;
        }
        credit = C[best].camp;
        for (u32 cp : active) CCA[(size_t)cp * ND + day].assist++;
      } else {
        credit = NCAMP;  // organic
      }
      size_t ix = (size_t)credit * ND + day;
      TL::CC& e = CCA[ix];
      e.sess++;
      e.psess += purch;
      e.rev += rev;
    }
    flush();
    // clear window state
    for (u32 cp : active) ccnt[cp] = 0;
    active.clear();
  }
}

// ------------------------------------------------------------------ parquet
struct TW {
  string b;
  vector<int> last{0};
  void byte(u8 x) { b.push_back((char)x); }
  void varint(u64 x) {
    while (x >= 0x80) {
      byte((u8)(x | 0x80));
      x >>= 7;
    }
    byte((u8)x);
  }
  void zz(i64 x) { varint((u64)((x << 1) ^ (x >> 63))); }
  void fh(int id, int type) {
    int d = id - last.back();
    if (d > 0 && d <= 15)
      byte((u8)((d << 4) | type));
    else {
      byte((u8)type);
      zz(id);
    }
    last.back() = id;
  }
  void i32f(int id, i32 v) {
    fh(id, 5);
    zz(v);
  }
  void i64f(int id, i64 v) {
    fh(id, 6);
    zz(v);
  }
  void strf(int id, const string& s) {
    fh(id, 8);
    varint(s.size());
    b += s;
  }
  void sbeg(int id) {
    fh(id, 12);
    last.push_back(0);
  }
  void send() {
    byte(0);
    last.pop_back();
  }
  void lbeg(int id, int et, int n) {
    fh(id, 9);
    if (n < 15)
      byte((u8)((n << 4) | et));
    else {
      byte((u8)(0xF0 | et));
      varint(n);
    }
  }
  void ebeg() { last.push_back(0); }
  void eend() {
    byte(0);
    last.pop_back();
  }
};

enum PType { PT_INT32 = 1, PT_INT64 = 2, PT_BYTES = 6 };
enum LType { LT_NONE, LT_STRING, LT_DATE };
struct PCol {
  string name;
  int ptype;
  int ltype;
  bool optional;
  vector<pair<const char*, size_t>> parts;  // page payload pieces (def levels + values)
};

static void put32(string& s, u32 v) { s.append((const char*)&v, 4); }
static void put64(string& s, u64 v) { s.append((const char*)&v, 8); }

static void write_parquet(const string& path, u64 nrows, vector<PCol>& cols) {
  vector<string> hdr(cols.size());
  vector<i64> offs(cols.size()), sizes(cols.size());
  i64 off = 4;
  for (size_t ci = 0; ci < cols.size(); ci++) {
    PCol& c = cols[ci];
    size_t dsz = 0;
    for (auto& pr : c.parts) dsz += pr.second;
    TW ph;
    ph.i32f(1, 0);  // DATA_PAGE
    ph.i32f(2, (i32)dsz);
    ph.i32f(3, (i32)dsz);
    ph.sbeg(5);
    ph.i32f(1, (i32)nrows);
    ph.i32f(2, 0);  // PLAIN
    ph.i32f(3, 3);  // RLE
    ph.i32f(4, 3);  // RLE
    ph.send();
    ph.byte(0);
    hdr[ci] = ph.b;
    offs[ci] = off;
    sizes[ci] = (i64)(ph.b.size() + dsz);
    off += sizes[ci];
  }
  TW m;
  m.i32f(1, 1);
  m.lbeg(2, 12, (int)cols.size() + 1);
  m.ebeg();
  m.strf(4, "schema");
  m.i32f(5, (i32)cols.size());
  m.eend();
  for (auto& c : cols) {
    m.ebeg();
    m.i32f(1, c.ptype);
    m.i32f(3, c.optional ? 1 : 0);
    m.strf(4, c.name);
    if (c.ltype == LT_STRING) {
      m.i32f(6, 0);  // UTF8
      m.sbeg(10);
      m.sbeg(1);
      m.send();
      m.send();
    } else if (c.ltype == LT_DATE) {
      m.i32f(6, 6);  // DATE
      m.sbeg(10);
      m.sbeg(6);
      m.send();
      m.send();
    }
    m.eend();
  }
  m.i64f(3, (i64)nrows);
  if (nrows > 0) {
    m.lbeg(4, 12, 1);
    m.ebeg();
    m.lbeg(1, 12, (int)cols.size());
    i64 total = 0;
    for (size_t ci = 0; ci < cols.size(); ci++) {
      PCol& c = cols[ci];
      total += sizes[ci];
      m.ebeg();
      m.i64f(2, offs[ci]);
      m.sbeg(3);
      m.i32f(1, c.ptype);
      m.lbeg(2, 5, 2);
      m.zz(0);
      m.zz(3);
      m.lbeg(3, 8, 1);
      m.varint(c.name.size());
      m.b += c.name;
      m.i32f(4, 0);
      m.i64f(5, (i64)nrows);
      m.i64f(6, sizes[ci]);
      m.i64f(7, sizes[ci]);
      m.i64f(9, offs[ci]);
      m.send();
      m.eend();
    }
    m.i64f(2, total);
    m.i64f(3, (i64)nrows);
    m.eend();
  } else {
    m.lbeg(4, 12, 0);
  }
  m.strf(6, "shopping-sessions-native");
  m.byte(0);
  string tail = m.b;
  put32(tail, (u32)m.b.size());
  tail += "PAR1";
  vector<iovec> iov;
  iov.push_back({(void*)"PAR1", 4});
  if (nrows > 0)
    for (size_t ci = 0; ci < cols.size(); ci++) {
      iov.push_back({(void*)hdr[ci].data(), hdr[ci].size()});
      for (auto& pr : cols[ci].parts)
        if (pr.second) iov.push_back({(void*)pr.first, pr.second});
    }
  iov.push_back({(void*)tail.data(), tail.size()});
  if (nrows == 0) {
    // no row group: drop the (unreferenced) pages, keep offsets irrelevant
  }
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die("open output");
  size_t i = 0;
  while (i < iov.size()) {
    int cnt = (int)min((size_t)512, iov.size() - i);
    ssize_t k = writev(fd, &iov[i], cnt);
    if (k < 0) die("write output");
    size_t done = (size_t)k;
    while (done > 0 && i < iov.size()) {
      if (done >= iov[i].iov_len) {
        done -= iov[i].iov_len;
        i++;
      } else {
        iov[i].iov_base = (char*)iov[i].iov_base + done;
        iov[i].iov_len -= done;
        done = 0;
      }
    }
    while (i < iov.size() && iov[i].iov_len == 0) i++;
  }
  close(fd);
}

// ------------------------------------------------------------------ main
static i64 parse_manifest_ts(const string& js, const char* key) {
  size_t k = js.find(key);
  if (k == string::npos) die("manifest key");
  size_t q1 = js.find('"', k + strlen(key) + 1);
  q1 = js.find('"', q1);
  // find the value string: after ':'
  size_t colon = js.find(':', k + strlen(key));
  size_t a = js.find('"', colon);
  size_t b = js.find('"', a + 1);
  string v = js.substr(a + 1, b - a - 1);
  v.append(64, '\0');
  i64 out;
  if (!parse_ts(v.data(), (u32)(b - a - 1), out)) die("manifest ts");
  return out;
}

static void load_small_table(const string& path, vector<pair<string, string>>& rows) {
  string s = read_small(path);
  if (s.empty()) return;
  if (s.back() != '\n') s.push_back('\n');
  size_t real = s.size();
  s.append(64, '\0');
  const char* p = s.data();
  const char* end = s.data() + real;
  p = next_line(p);
  while (p < end) {
    Fld f[2];
    bool ok;
    const char* nx = split<2>(p, f, ok);
    if (ok) rows.push_back({string(f[0].p, f[0].n), string(f[1].p, f[1].n)});
    p = nx;
  }
}

int main() {
  T0 = now_s();
  mallopt(M_MMAP_THRESHOLD, 1 << 20);  // big blocks go straight back to the kernel when freed
  init_date_tables();
  VERBOSE = getenv("VERBOSE") != nullptr;
  const char* e;
  string DATA = (e = getenv("DATA_DIR")) ? e : "/data";
  string OUT = (e = getenv("OUTPUT_DIR")) ? e : "/output";
  NT = (e = getenv("NCPU")) ? atoi(e) : (int)thread::hardware_concurrency();
  if (NT < 1) NT = 1;
  if (NT > 64) NT = 64;

  // manifest
  {
    string js = read_small(DATA + "/manifest.json");
    WS = parse_manifest_ts(js, "\"window_start\"");
    WE = parse_manifest_ts(js, "\"window_end\"");
    CWS = WS - D7;
    DAY0 = WS / 86400000;
    ND = (int)((WE - 1) / 86400000 - DAY0 + 1);
    if (WE - CWS >= (1LL << 32)) die("window too long");
  }

  // files
  vector<File> lf = list_csv(DATA + "/links"), ef = list_csv(DATA + "/events"), cf = list_csv(DATA + "/clicks");
  vector<File*> all;
  size_t lb = 0, ebytes = 0, cbytes = 0;
  for (auto& f : lf) all.push_back(&f), lb += f.size;
  for (auto& f : ef) all.push_back(&f), ebytes += f.size;
  for (auto& f : cf) all.push_back(&f), cbytes += f.size;
  map_files(all);
  LOG("mapped %zu files: links %zu, events %zu, clicks %zu bytes", all.size(), lb, ebytes, cbytes);

  // user table + id space
  {
    double lines = ebytes / 75.0 + cbytes / 50.0 + (double)lb / 10.0;
    u64 want = (u64)(lines / 8) + 65536;
    int bits = 16;
    while ((1ull << bits) < want) bits++;
    UT = (Slot*)vmalloc(sizeof(Slot) << bits, true);
    {
      // fault the table in with sequential parallel sweeps (random first-touch faults are far slower)
      size_t nb = sizeof(Slot) << bits;
      parallel([&](int t) {
        size_t per = ((nb / NT) + 4095) & ~(size_t)4095;
        for (size_t o = (size_t)t * per; o < min(nb, (size_t)(t + 1) * per); o += 4096) ((volatile char*)UT)[o] = 0;
      });
    }
    UBITS = bits;
    UMASK = (1ull << bits) - 1;
    USHIFT = 64 - bits;
    MAXID = (u64)(ebytes / 30 + cbytes / 8 + lb / 2) + (u64)NT * 4096 + 65536;
    if (MAXID > 0xFFFFFFF0ull) MAXID = 0xFFFFFFF0ull;
    NAMEP = (const char**)vmalloc(sizeof(char*) * MAXID, false);
    NAMEN = (u32*)vmalloc(sizeof(u32) * MAXID, false);
  }

  // small tables and size estimates: done by thread 0 while the others parse links
  double est_ev = 0, est_cl = 0;
  auto small_job = [&]() {
  // products & categories
  {
    vector<pair<string, string>> rows;
    load_small_table(DATA + "/products.csv", rows);
    vector<string> cats;
    for (auto& r : rows) cats.push_back(r.second);
    cats.push_back("unknown");
    sort(cats.begin(), cats.end());
    cats.erase(unique(cats.begin(), cats.end()), cats.end());
    if (cats.size() > 64) die("too many categories");
    CATS = cats;
    for (auto& c : cats) {
      string lp(4, '\0');
      u32 n = (u32)c.size();
      memcpy(&lp[0], &n, 4);
      CATLP.push_back(lp + c);
    }
    UNKNOWN_CAT = (u32)(lower_bound(cats.begin(), cats.end(), string("unknown")) - cats.begin());
    vector<pair<string, u32>> ids;
    ids.reserve(rows.size());
    for (auto& r : rows) {
      u32 ci = (u32)(lower_bound(cats.begin(), cats.end(), r.second) - cats.begin());
      ids.push_back({r.first, ci});
    }
    PRODD.build(ids, "p");
    if (!PRODD.ok) {  // general fallback map, only when ids are not all canonical
      PROD.init(rows.size() + 1);
      for (auto& r : ids) {
        string key = r.first;
        key.append(16, '\0');
        PROD.insert(key.data(), (u32)r.first.size(), r.second);
      }
    }
  }
  // campaigns
  {
    vector<pair<string, string>> rows;
    load_small_table(DATA + "/campaigns.csv", rows);
    CAMP.init(rows.size() + 1);
    for (auto& r : rows) {
      string key = r.first;
      key.append(16, '\0');
      u32 idx = (u32)CAMPIDS.size();
      if (CAMP.insert(key.data(), (u32)r.first.size(), idx) == idx) {
        CAMPIDS.push_back(r.first);
        CHANNELS.push_back(r.second);
      }
    }
    NCAMP = (u32)CAMPIDS.size();
    {
      vector<pair<string, u32>> ids;
      for (u32 i = 0; i < NCAMP; i++) ids.push_back({CAMPIDS[i], i});
      CAMPD.build(ids, "c_");
    }
    if (NCAMP >= 65535) die("too many campaigns");
  }
  // estimates
  double ev_line = 75, cl_line = 50;
  {
    // sample average line lengths
    auto sample = [](vector<File>& fs, double def) {
      size_t bytes = 0, lines = 0;
      for (auto& f : fs) {
        size_t n = min(f.size, (size_t)(1 << 20));
        for (size_t i = 0; i < n; i++) lines += f.data[i] == '\n';
        bytes += n;
        if (bytes > (4 << 20)) break;
      }
      return lines > 100 ? (double)bytes / lines : def;
    };
    ev_line = sample(ef, ev_line);
    cl_line = sample(cf, cl_line);
  }
  est_ev = ebytes / ev_line + 1000;
  est_cl = cbytes / cl_line + 1000;
  P1B = 4;
  while (P1B < 13 && (est_ev / (1u << P1B)) > 40000) P1B++;
  if (getenv("P1B")) P1B = atoi(getenv("P1B"));
  P1 = 1u << P1B;

  {
    // person partitions are id ranges of 1 << P2S; ids never exceed the table capacity
    // (plus partially used allocation blocks), so P2 partitions cover every id.
    double est_ids = est_ev / 12.0 + 1024;
    double want_parts = est_ev / 24000.0;
    P2S = 6;
    while (P2S < 20 && est_ids / (double)(1u << P2S) > want_parts) P2S++;
    if (getenv("P2S")) P2S = atoi(getenv("P2S"));
    P2 = (u32)((((1ull << UBITS) + (u64)NT * 4096 + 65536) >> P2S) + 1);
    P2USED = (u32)(est_ids / (double)(1u << P2S)) + 1;
    LOG("P1=%u P2=%u (used ~%u, 2^%d persons each) utable=2^%d est_ev=%.0f est_cl=%.0f", P1, P2, P2USED, P2S, UBITS,
        est_ev, est_cl);
  }
    LOG("small tables: %zu cats, %u campaigns, %d days", CATS.size(), NCAMP, ND);
  };

  TLS.resize(NT);

  // ---------------- phase 1: links
  {
    u64 maxl = (u64)(lb / 2) + (u64)NT * 2048 + 1024;
    SLOT_OF = (u32*)vmalloc(sizeof(u32) * maxl, false);
    PFX = (u64*)vmalloc(sizeof(u64) * maxl, false);
    vector<Chunk> ch;
    make_chunks(lf, 0, 1 << 20, ch);
    atomic<size_t> next{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      if (t == 0 || NT == 1) small_job();
      for (;;) {
        size_t i = next.fetch_add(1);
        if (i >= ch.size()) break;
        parse_links_chunk(ch[i], tl);
      }
      tl.al.cur = tl.al.end = 0;  // drop the rest of the block
    });
  }
  u32 L = NEXT_ID.load();
  LOG("links parsed: %u link ids", L);

  // ---------------- phase 2: union-find
  {
    PARENT = (atomic<u32>*)vmalloc(sizeof(u32) * (L + 1), true);
    MINN = (atomic<u32>*)vmalloc(sizeof(u32) * (L + 1), true);
    atomic<size_t> nx{0};
    parallel([&](int t) {
      size_t per = (L + NT - 1) / NT;
      for (size_t i = t * per; i < min((size_t)L, (t + 1) * per); i++) {
        PARENT[i].store((u32)i, memory_order_relaxed);
        MINN[i].store((u32)i, memory_order_relaxed);
      }
    });
    LOG("uf: init");
    parallel([&](int t) {
      for (auto& e2 : TLS[t].edges) uf_union(e2.first, e2.second);
    });
    LOG("uf: union");
    parallel([&](int t) {
      size_t per = (L + NT - 1) / NT;
      for (size_t i = t * per; i < min((size_t)L, (t + 1) * per); i++) {
        if (!NAMEP[i]) continue;
        u32 r = uf_find((u32)i);
        if (r == i) continue;
        UT[SLOT_OF[i]].pid.store(r + 1, memory_order_relaxed);
        u64 mine = PFX[i];
        u32 cur = MINN[r].load(memory_order_relaxed);
        for (;;) {
          u64 other = PFX[cur];
          bool less = mine < other || (mine == other && name_less((u32)i, cur));
          if (!less) break;
          if (MINN[r].compare_exchange_weak(cur, (u32)i, memory_order_relaxed)) break;
        }
      }
    });
    LOG("uf: minname+slots");
    parallel([&](int t) {
      size_t per = (L + NT - 1) / NT;
      for (size_t i = t * per; i < min((size_t)L, (t + 1) * per); i++) {
        if (!NAMEP[i]) continue;
        if (PARENT[i].load(memory_order_relaxed) == i) {
          u32 mn = MINN[i].load(memory_order_relaxed);
          if (mn != i) {
            NAMEP[i] = NAMEP[mn];
            NAMEN[i] = NAMEN[mn];
          }
        }
      }
    });
    for (auto& tl : TLS) vector<pair<u32, u32>>().swap(tl.edges);
  }
  LOG("union-find done");

  // ---------------- phase 3: events, then clicks
  vector<size_t> vpart(P2, 0);  // valid event copies per person partition
  {
    const double capf = getenv("CAPF") ? atof(getenv("CAPF")) : 1.0;  // test knob: shrink to exercise overflow paths
    size_t ecap = (size_t)(est_ev / NT / P1 * 1.3 * capf) + 64;
    vector<Chunk> ch;
    make_chunks(ef, 1, 4 << 20, ch);
    atomic<size_t> next{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      init_bufs(tl.eb, P1, ecap);
      tl.ebw.init(tl.eb);
      tl.phist.assign(P2, 0);
      for (;;) {
        size_t i = next.fetch_add(1);
        if (i >= ch.size()) break;
        parse_events_chunk(ch[i], tl);
      }
      tl.ebw.finish();
    });
    size_t tot = 0;
    for (auto& tl : TLS)
      for (u32 k = 0; k < P2; k++) vpart[k] += tl.phist[k];
    for (size_t v : vpart) tot += v;
    LOG("events parsed");
    // clicks: per-partition capacity follows the event distribution
    vector<size_t> ccaps(P2);
    for (u32 k = 0; k < P2; k++)
      ccaps[k] = (size_t)(est_cl * ((double)vpart[k] / (double)(tot + 1)) / NT * 1.5 * capf) + 32;
    vector<Chunk> cch;
    make_chunks(cf, 2, 4 << 20, cch);
    atomic<size_t> cnext{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      init_bufs_var(tl.cb, ccaps);
      tl.cbw.init(tl.cb);
      for (;;) {
        size_t i = cnext.fetch_add(1);
        if (i >= cch.size()) break;
        parse_clicks_chunk(cch[i], tl);
      }
      tl.cbw.finish();
    });
  }
  u32 NIDS = NEXT_ID.load();
  LOG("events/clicks parsed: ids=%u", NIDS);

  // ---------------- phase 4: dedup
  {
    size_t est_surv = (size_t)(est_ev * 0.97);
    const double capf = getenv("CAPF") ? atof(getenv("CAPF")) : 1.0;
    vector<size_t> pcaps(P2);
    for (u32 k = 0; k < P2; k++) pcaps[k] = (size_t)((double)vpart[k] / NT * 1.2 * capf) + 32;
    atomic<u32> next{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      init_bufs_var(tl.pb, pcaps);
      tl.pbw.init(tl.pb);
      vector<u64> tab;  // (tag << 32) | (thread << 26 | index); empty = ~0
      auto rec = [&](u32 p, u32 code) -> ERec& {
        Buf<ERec>& b = TLS[code >> 26].eb[p];
        u32 i = code & ((1u << 26) - 1);
        return b.at(i);
      };
      for (;;) {
        u32 p = next.fetch_add(1);
        if (p >= P1) break;
        size_t n = 0;
        for (int s = 0; s < NT; s++) n += TLS[s].eb[p].size();
        size_t cap = 64;
        while (cap < n * 2) cap <<= 1;
        tab.assign(cap, ~0ull);
        u64 mask = cap - 1;
        // pass 1: keep the smallest ingest_ts per event_id, mark every other copy INVALID
        for (int s = 0; s < NT; s++) {
          Buf<ERec>& b = TLS[s].eb[p];
          size_t tot = b.size();
          for (size_t i = 0; i < tot; i++) {
            ERec& r = b.at(i);
            u32 tag = (u32)(r.eh >> 20);
            u64 k = r.eh & mask;
            u64 me = ((u64)tag << 32) | ((u32)s << 26) | (u32)i;
            for (;;) {
              u64 e = tab[k];
              if (e == ~0ull) {
                tab[k] = me;
                break;
              }
              if ((u32)(e >> 32) == tag) {
                ERec& o = rec(p, (u32)e);
                if (o.eh == r.eh) {
                  if (r.ingest < o.ingest) {
                    o.val = INVALID;
                    tab[k] = me;
                  } else {
                    r.val = INVALID;
                  }
                  break;
                }
              }
              k = (k + 1) & mask;
            }
          }
        }
        // pass 2: sequential scatter of the surviving valid copies by person range
        for (int s = 0; s < NT; s++)
          TLS[s].eb[p].each([&](const ERec& r) {
            if (r.val != INVALID) tl.pbw.push(r.person >> P2S, PE{r.person, r.val});
          });
      }
      tl.pbw.finish();
    });
  }
  LOG("dedup done");
  if (VERBOSE) {
    size_t eo = 0, et = 0, co = 0, ct = 0, po = 0, pt = 0;
    for (auto& tl : TLS) {
      for (auto& b : tl.eb) eo += b.ovsize(), et += b.size();
      for (auto& b : tl.cb) co += b.ovsize(), ct += b.size();
      for (auto& b : tl.pb) po += b.ovsize(), pt += b.size();
    }
    LOG("overflow: events %zu/%zu clicks %zu/%zu persons %zu/%zu", eo, et, co, ct, po, pt);
  }

  // ---------------- phase 5: sessions
  {
    u32 NL = 1u << P2S;
    if (NIDS > ((u64)P2 << P2S)) die("id range");
    vector<pair<size_t, u32>> order(P2);
    for (u32 p = 0; p < P2; p++) {
      size_t n = 0;
      for (int t = 0; t < NT; t++) n += TLS[t].pb[p].size();
      order[p] = {n, p};
    }
    sort(order.begin(), order.end(), [](auto& a, auto& b) { return a.first > b.first; });
    atomic<u32> next{0};
    size_t csz = (size_t)(NCAMP + 1) * ND;
    parallel([&](int t) {
      TL& tl = TLS[t];
      tl.cc.assign(csz, TL::CC{0, 0, 0, 0, 0});
      SessCtx sc;
      sc.ccnt.assign(NCAMP + 1, 0);
      sc.apos.assign(NCAMP + 1, 0);
      sc.catsum.assign(64, 0);
      for (;;) {
        u32 i = next.fetch_add(1);
        if (i >= P2) break;
        process_partition(order[i].second, tl, sc, NL);
      }
    });
  }
  LOG("sessions done");

  // ---------------- phase 6: output
  {
    string ud = OUT + "/user_daily", cd = OUT + "/campaign_daily";
    mkdir(OUT.c_str(), 0777);
    mkdir(ud.c_str(), 0777);
    mkdir(cd.c_str(), 0777);
    atomic<int> wrote{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      u64 n = tl.o_rows;
      if (n == 0 && t != 0) return;
      vector<PCol> cols(6);
      cols[0] = PCol{"user_id", PT_BYTES, LT_STRING, false, {{tl.o_uid.p, tl.o_uid.n}}};
      cols[1] = PCol{"day", PT_INT32, LT_DATE, false, {{tl.o_day.p, tl.o_day.n}}};
      cols[2] = PCol{"sessions", PT_INT32, LT_NONE, false, {{tl.o_sess.p, tl.o_sess.n}}};
      cols[3] = PCol{"events", PT_INT32, LT_NONE, false, {{tl.o_ev.p, tl.o_ev.n}}};
      cols[4] = PCol{"revenue_cents", PT_INT64, LT_NONE, false, {{tl.o_rev.p, tl.o_rev.n}}};
      // def levels: [u32 len][varint (groups<<1)|1][bit-packed groups]
      string dh;
      {
        u64 groups = (n + 7) / 8;
        u64 hv = (groups << 1) | 1;
        string v;
        while (hv >= 0x80) {
          v.push_back((char)(hv | 0x80));
          hv >>= 7;
        }
        v.push_back((char)hv);
        put32(dh, (u32)(v.size() + groups));
        dh += v;
      }
      cols[5] = PCol{"top_category", PT_BYTES, LT_STRING, true,
                     {{dh.data(), dh.size()}, {tl.o_def.p, (size_t)((n + 7) / 8)}, {tl.o_top.p, tl.o_top.n}}};
      write_parquet(ud + "/part-" + to_string(t) + ".parquet", n, cols);
      // drop the column buffers now, in parallel, instead of at exit
      for (TL::BB* bb : {&tl.o_uid, &tl.o_day, &tl.o_sess, &tl.o_ev, &tl.o_rev, &tl.o_top, &tl.o_def}) {
        uintptr_t a0 = ((uintptr_t)bb->p + 4095) & ~(uintptr_t)4095, a1 = ((uintptr_t)bb->p + bb->cap) & ~(uintptr_t)4095;
        if (bb->p && a1 > a0) madvise((void*)a0, a1 - a0, MADV_DONTNEED);
      }
      wrote++;
    });
    // campaign_daily
    size_t csz = (size_t)(NCAMP + 1) * ND;
    vector<i64> cs(csz, 0), cps(csz, 0), crev(csz, 0), cas(csz, 0);
    for (auto& tl : TLS)
      for (size_t i = 0; i < csz; i++) {
        cs[i] += tl.cc[i].sess;
        cps[i] += tl.cc[i].psess;
        crev[i] += tl.cc[i].rev;
        cas[i] += tl.cc[i].assist;
      }
    vector<string> cdat(7);
    u64 rows = 0;
    for (u32 c = 0; c <= NCAMP; c++)
      for (int d = 0; d < ND; d++) {
        size_t i = (size_t)c * ND + d;
        if (cs[i] == 0 && cas[i] == 0) continue;
        const string id = c == NCAMP ? string("organic") : CAMPIDS[c];
        const string ch = c == NCAMP ? string("organic") : CHANNELS[c];
        put32(cdat[0], (u32)id.size());
        cdat[0] += id;
        put32(cdat[1], (u32)ch.size());
        cdat[1] += ch;
        put32(cdat[2], (u32)(i32)(DAY0 + d));
        put64(cdat[3], (u64)cs[i]);
        put64(cdat[4], (u64)cps[i]);
        put64(cdat[5], (u64)crev[i]);
        put64(cdat[6], (u64)cas[i]);
        rows++;
      }
    const char* names[7] = {"campaign_id", "channel", "day", "sessions", "purchasing_sessions", "revenue_cents",
                            "assisted_sessions"};
    const int pts[7] = {PT_BYTES, PT_BYTES, PT_INT32, PT_INT64, PT_INT64, PT_INT64, PT_INT64};
    const int lts[7] = {LT_STRING, LT_STRING, LT_DATE, LT_NONE, LT_NONE, LT_NONE, LT_NONE};
    vector<PCol> cols(7);
    for (int k = 0; k < 7; k++) cols[k] = PCol{names[k], pts[k], lts[k], false, {{cdat[k].data(), cdat[k].size()}}};
    write_parquet(cd + "/part-0.parquet", rows, cols);
  }
  LOG("output written");
  // Tearing ~10 GB of mappings down at exit is serial in the kernel; dropping the
  // pages with MADV_DONTNEED (mmap_lock held for read) runs on every thread at once.
  {
    const size_t SL = 64ull << 20;
    vector<pair<char*, size_t>> slices;
    for (auto& r : REGIONS)
      for (size_t o = 0; o < r.second; o += SL) slices.push_back({r.first + o, min(SL, r.second - o)});
    atomic<size_t> nx{0};
    parallel([&](int t) {
      for (;;) {
        size_t i = nx.fetch_add(1);
        if (i >= slices.size()) break;
        madvise(slices[i].first, slices[i].second, MADV_DONTNEED);
      }
    });
  }
  LOG("released");
  if (VERBOSE) {
    string st = read_small("/proc/self/status");
    size_t a = st.find("VmRSS");
    size_t b = st.find("RssFile");
    if (a != string::npos) fprintf(stderr, "%s", st.substr(a, st.find('\n', st.find("RssShmem")) - a).c_str());
    (void)b;
    fputc('\n', stderr);
    if (getenv("SMAPS")) {
      string sm = read_small("/proc/self/smaps");
      FILE* fo = fopen(getenv("SMAPS"), "w");
      fwrite(sm.data(), 1, sm.size(), fo);
      fclose(fo);
    }
  }
  fflush(stderr);
  _exit(0);
}
