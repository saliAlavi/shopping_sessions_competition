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
static void* vmalloc(size_t n, bool huge) {
  n = (n + 4095) & ~(size_t)4095;
  if (n == 0) n = 4096;
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) die("mmap anon");
  madvise(p, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
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
static inline u64 hash_str(const char* p, u32 n) {
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

// ------------------------------------------------------------------ timestamps
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
static inline i64 parse_f1_sec(const char* p) {
  unsigned y = d2(p) * 100 + d2(p + 2), mo = d2(p + 5), d = d2(p + 8), h = d2(p + 11), mi = d2(p + 14),
           s = d2(p + 17);
  return days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s;
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
static vector<string> CATS;
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
static atomic<u32> NEXT_ID{0};
static const char** NAMEP;
static u32* NAMEN;
static u32* SLOT_OF;
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
        if (LINK) SLOT_OF[id] = (u32)i;
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
template <class T>
struct Buf {
  T* d = nullptr;
  u32 n = 0, cap = 0;
  vector<T> ov;
  inline void push(const T& x) {
    if (LIKELY(n < cap))
      d[n++] = x;
    else
      ov.push_back(x);
  }
  template <class F>
  inline void each(F f) const {
    for (u32 i = 0; i < n; i++) f(d[i]);
    for (const T& x : ov) f(x);
  }
  size_t size() const { return n + ov.size(); }
};

template <class T>
static void init_bufs(vector<Buf<T>>& v, size_t parts, size_t cap) {
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

static int P1B, P2B;  // log2 partitions
static u32 P1, P2;

struct TL {  // thread-local state
  vector<Buf<ERec>> eb;
  vector<Buf<PE>> pb;
  vector<Buf<CRec>> cb;
  vector<pair<u32, u32>> edges;
  IdAlloc al;
  // outputs
  vector<u32> o_person;
  vector<i32> o_day, o_sess, o_ev;
  vector<i64> o_rev;
  vector<int8_t> o_top;
  vector<i64> c_sess, c_psess, c_rev, c_assist;  // [(camp)*ND + day]
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
static void parse_links_chunk(const Chunk& c, TL& tl) {
  for_lines(c, [&](const char* p) {
    Fld f[2];
    bool ok;
    const char* nx = split<2>(p, f, ok);
    if (ok && f[0].n && f[1].n) {
      u32 a = uget<true>(hash_str(f[0].p, f[0].n) | 1, f[0].p, f[0].n, tl.al);
      u32 b = uget<true>(hash_str(f[1].p, f[1].n) | 1, f[1].p, f[1].n, tl.al);
      if (a != b) tl.edges.push_back({a, b});
    }
    return nx;
  });
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
  i64 lines = 0;
  for_lines(c, [&](const char* p) {
    Fld f[7];
    bool ok;
    const char* nx = split<7>(p, f, ok);
    lines++;
    if (UNLIKELY(!ok || f[0].n == 0)) return nx;
    u64 eh = hash_str(f[0].p, f[0].n);
    ERec r;
    r.eh = eh;
    r.person = 0;
    r.val = INVALID;
    i64 ing = parse_f1_sec(f[3].p);
    r.ingest = (u32)ing;
    do {
      if (UNLIKELY(f[1].n == 0 || f[5].n == 0)) break;
      u64 uh = hash_str(f[1].p, f[1].n) | 1;
      _mm_prefetch((const char*)&UT[uh >> USHIFT], _MM_HINT_T0);
      i64 ts;
      if (UNLIKELY(!parse_ts(f[2].p, f[2].n, ts))) break;
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
        u32 cat = PROD.find(f[5].p, f[5].n);
        if (cat == ~0u) cat = UNKNOWN_CAT;
        pay = mkpay(2, cat, v);
      } else {
        break;
      }
      r.person = uget<false>(uh, f[1].p, f[1].n, tl.al);
      r.val = ((u64)(u32)(ts - WS) << 32) | pay;
    } while (0);
    tl.eb[eh >> p1shift].push(r);
    return nx;
  });
  tl.n_ev_lines += lines;
}

static void parse_clicks_chunk(const Chunk& c, TL& tl) {
  i64 lines = 0;
  for_lines(c, [&](const char* p) {
    Fld f[4];
    bool ok;
    const char* nx = split<4>(p, f, ok);
    lines++;
    if (UNLIKELY(!ok || f[0].n == 0 || f[1].n == 0)) return nx;
    u64 uh = hash_str(f[1].p, f[1].n) | 1;
    _mm_prefetch((const char*)&UT[uh >> USHIFT], _MM_HINT_T0);
    i64 ts;
    if (!parse_ts(f[3].p, f[3].n, ts)) return nx;
    if (ts < CWS || ts >= WE) return nx;
    u32 cmp = CAMP.find(f[2].p, f[2].n);
    if (cmp == ~0u) return nx;
    u32 person = uget<false>(uh, f[1].p, f[1].n, tl.al);
    CRec r;
    r.person = person;
    r.ts = (u32)(ts - CWS);
    r.ref = ((u64)(f[0].p - BASE) << 16) | cmp;
    tl.cb[person & (P2 - 1)].push(r);
    return nx;
  });
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
  if (n <= 24) {
    for (size_t i = 1; i < n; i++) {
      u64 x = a[i];
      size_t j = i;
      while (j > 0 && a[j - 1] > x) {
        a[j] = a[j - 1];
        j--;
      }
      a[j] = x;
    }
  } else if (n < 4096) {
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
  vector<u32> cnt, ccount;
};

static void process_partition(u32 p, TL& tl, SessCtx& sc, u32 NL) {
  // gather events, counting sort by local person
  vector<u32>& cnt = sc.cnt;
  cnt.assign(NL + 1, 0);
  size_t ne = 0;
  for (int t = 0; t < NT; t++) {
    const Buf<PE>& b = TLS[t].pb[p];
    b.each([&](const PE& x) { cnt[x.person >> P2B]++; });
    ne += b.size();
  }
  if (ne == 0) return;
  u32 s = 0;
  for (u32 i = 0; i < NL; i++) {
    u32 c = cnt[i];
    cnt[i] = s;
    s += c;
  }
  cnt[NL] = s;
  if (sc.ev.size() < ne) sc.ev.resize(ne);
  u64* ev = sc.ev.data();
  for (int t = 0; t < NT; t++) TLS[t].pb[p].each([&](const PE& x) { ev[cnt[x.person >> P2B]++] = x.val; });
  // now cnt[i] = end of local i; start = cnt[i-1] (end of previous)

  vector<u32>& cc = sc.ccount;
  cc.assign(NL + 1, 0);
  size_t nc = 0;
  for (int t = 0; t < NT; t++) {
    const Buf<CRec>& b = TLS[t].cb[p];
    b.each([&](const CRec& x) { cc[x.person >> P2B]++; });
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
      cl[cc[x.person >> P2B]++] = Cl{x.ts, (u32)(x.ref & 0xFFFF), x.ref >> 16};
    });

  u32* ccnt = sc.ccnt.data();
  u32* apos = sc.apos.data();
  vector<u32>& active = sc.active;
  i64* catsum = sc.catsum.data();
  vector<u32>& touched = sc.touched;
  i64* c_sess = tl.c_sess.data();
  i64* c_psess = tl.c_psess.data();
  i64* c_rev = tl.c_rev.data();
  i64* c_assist = tl.c_assist.data();

  u32 eprev = 0, cprev = 0;
  for (u32 li = 0; li < NL; li++) {
    u32 eend = cnt[li], cend = cc[li];
    u32 eb = eprev, cb = cprev;
    eprev = eend;
    cprev = cend;
    if (eend == eb) continue;
    u32 person = (li << P2B) | p;
    u64* E = ev + eb;
    u32 n = eend - eb;
    sort_u64(E, n, sc.sorttmp);
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
      tl.o_person.push_back(person);
      tl.o_day.push_back((i32)(DAY0 + curday));
      tl.o_sess.push_back(d_sess);
      tl.o_ev.push_back(d_ev);
      tl.o_rev.push_back(d_rev);
      tl.o_top.push_back(top);
    };
    while (i < n) {
      u32 st = (u32)(E[i] >> 32);
      i32 day = (i32)((WS + st) / 86400000 - DAY0);
      if (day != curday) {
        flush();
        curday = day;
        d_sess = 0;
        d_ev = 0;
        d_rev = 0;
      }
      u32 last = st;
      i64 rev = 0;
      bool purch = false;
      u32 j = i;
      while (j < n) {
        u64 x = E[j];
        u32 t = (u32)(x >> 32);
        if (t - last > GAP) break;
        last = t;
        u32 pl = (u32)x;
        if ((pl >> 30) == 2) {
          u32 price = pl & 0xFFFFFF, cat = (pl >> 24) & 63;
          rev += price;
          purch = true;
          if (catsum[cat] == 0) touched.push_back(cat);
          catsum[cat] += price;
        }
        j++;
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
        for (u32 cp : active) c_assist[(size_t)cp * ND + day]++;
      } else {
        credit = NCAMP;  // organic
      }
      size_t ix = (size_t)credit * ND + day;
      c_sess[ix]++;
      if (purch) c_psess[ix]++;
      c_rev[ix] += rev;
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
  string data;  // page payload (def levels + values)
};

static void put32(string& s, u32 v) { s.append((const char*)&v, 4); }
static void put64(string& s, u64 v) { s.append((const char*)&v, 8); }

static void write_parquet(const string& path, u64 nrows, vector<PCol>& cols) {
  string out;
  out.reserve(64);
  out += "PAR1";
  vector<i64> offs(cols.size()), sizes(cols.size());
  for (size_t ci = 0; ci < cols.size(); ci++) {
    PCol& c = cols[ci];
    TW ph;
    ph.i32f(1, 0);  // DATA_PAGE
    ph.i32f(2, (i32)c.data.size());
    ph.i32f(3, (i32)c.data.size());
    ph.sbeg(5);
    ph.i32f(1, (i32)nrows);
    ph.i32f(2, 0);  // PLAIN
    ph.i32f(3, 3);  // RLE
    ph.i32f(4, 3);  // RLE
    ph.send();
    ph.byte(0);
    offs[ci] = (i64)out.size();
    out += ph.b;
    out += c.data;
    sizes[ci] = (i64)(ph.b.size() + c.data.size());
    string().swap(c.data);
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
  out += m.b;
  put32(out, (u32)m.b.size());
  out += "PAR1";
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die("open output");
  size_t w = 0;
  while (w < out.size()) {
    ssize_t k = write(fd, out.data() + w, out.size() - w);
    if (k <= 0) die("write output");
    w += (size_t)k;
  }
  close(fd);
}

static string deflevels(const vector<int8_t>& present) {
  // RLE/bit-packed hybrid, bit width 1, one bit-packed run; 4-byte length prefix
  size_t n = present.size();
  size_t groups = (n + 7) / 8;
  string r;
  u64 hdr = ((u64)groups << 1) | 1;
  string v;
  while (hdr >= 0x80) {
    v.push_back((char)(hdr | 0x80));
    hdr >>= 7;
  }
  v.push_back((char)hdr);
  size_t base = v.size();
  v.resize(base + groups, 0);
  for (size_t i = 0; i < n; i++)
    if (present[i]) v[base + i / 8] |= (char)(1 << (i % 8));
  put32(r, (u32)v.size());
  r += v;
  return r;
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
    UNKNOWN_CAT = (u32)(lower_bound(cats.begin(), cats.end(), string("unknown")) - cats.begin());
    PROD.init(rows.size() + 1);
    for (auto& r : rows) {
      u32 ci = (u32)(lower_bound(cats.begin(), cats.end(), r.second) - cats.begin());
      string key = r.first;
      key.append(16, '\0');
      PROD.insert(key.data(), (u32)r.first.size(), ci);
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
    if (NCAMP >= 65535) die("too many campaigns");
  }
  LOG("small tables: %zu cats, %u campaigns, %d days", CATS.size(), NCAMP, ND);

  // files
  vector<File> lf = list_csv(DATA + "/links"), ef = list_csv(DATA + "/events"), cf = list_csv(DATA + "/clicks");
  vector<File*> all;
  size_t lb = 0, ebytes = 0, cbytes = 0;
  for (auto& f : lf) all.push_back(&f), lb += f.size;
  for (auto& f : ef) all.push_back(&f), ebytes += f.size;
  for (auto& f : cf) all.push_back(&f), cbytes += f.size;
  map_files(all);
  LOG("mapped %zu files: links %zu, events %zu, clicks %zu bytes", all.size(), lb, ebytes, cbytes);

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
  double est_ev = ebytes / ev_line + 1000, est_cl = cbytes / cl_line + 1000;
  P1B = 4;
  while (P1B < 13 && (est_ev / (1u << P1B)) > 16384) P1B++;
  P2B = 4;
  while (P2B < 12 && (est_ev / (1u << P2B)) > 32768) P2B++;
  P1 = 1u << P1B;
  P2 = 1u << P2B;

  // user table + id space
  {
    double lines = est_ev + est_cl + (double)lb / 10.0;
    u64 want = (u64)(lines / 5) + 65536;
    int bits = 16;
    while ((1ull << bits) < want) bits++;
    UT = (Slot*)vmalloc(sizeof(Slot) << bits, true);
    UMASK = (1ull << bits) - 1;
    USHIFT = 64 - bits;
    MAXID = (u64)(ebytes / 30 + cbytes / 8 + lb / 2) + (u64)NT * 4096 + 65536;
    if (MAXID > 0xFFFFFFF0ull) MAXID = 0xFFFFFFF0ull;
    NAMEP = (const char**)vmalloc(sizeof(char*) * MAXID, false);
    NAMEN = (u32*)vmalloc(sizeof(u32) * MAXID, false);
    LOG("P1=%u P2=%u utable=2^%d est_ev=%.0f est_cl=%.0f", P1, P2, bits, est_ev, est_cl);
  }

  TLS.resize(NT);

  // ---------------- phase 1: links
  {
    u64 maxl = (u64)(lb / 2) + (u64)NT * 2048 + 1024;
    SLOT_OF = (u32*)vmalloc(sizeof(u32) * maxl, false);
    vector<Chunk> ch;
    make_chunks(lf, 0, 1 << 20, ch);
    atomic<size_t> next{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
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
    parallel([&](int t) {
      for (auto& e2 : TLS[t].edges) uf_union(e2.first, e2.second);
    });
    parallel([&](int t) {
      size_t per = (L + NT - 1) / NT;
      for (size_t i = t * per; i < min((size_t)L, (t + 1) * per); i++) {
        if (!NAMEP[i]) continue;
        u32 r = uf_find((u32)i);
        if (r == i) continue;
        u32 cur = MINN[r].load(memory_order_relaxed);
        while (name_less((u32)i, cur)) {
          if (MINN[r].compare_exchange_weak(cur, (u32)i, memory_order_relaxed)) break;
        }
      }
    });
    parallel([&](int t) {
      size_t per = (L + NT - 1) / NT;
      for (size_t i = t * per; i < min((size_t)L, (t + 1) * per); i++) {
        if (!NAMEP[i]) continue;
        u32 r = uf_find((u32)i);
        if (r != i) UT[SLOT_OF[i]].pid.store(r + 1, memory_order_relaxed);
      }
    });
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

  // ---------------- phase 3: events + clicks
  {
    size_t ecap = (size_t)(est_ev / NT / P1 * 1.3) + 64;
    size_t ccap = (size_t)(est_cl / NT / P2 * 1.3) + 64;
    parallel([&](int t) {
      init_bufs(TLS[t].eb, P1, ecap);
      init_bufs(TLS[t].cb, P2, ccap);
    });
    vector<Chunk> ch;
    make_chunks(ef, 1, 4 << 20, ch);
    make_chunks(cf, 2, 4 << 20, ch);
    atomic<size_t> next{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      for (;;) {
        size_t i = next.fetch_add(1);
        if (i >= ch.size()) break;
        if (ch[i].kind == 1)
          parse_events_chunk(ch[i], tl);
        else
          parse_clicks_chunk(ch[i], tl);
      }
    });
  }
  u32 NIDS = NEXT_ID.load();
  LOG("events/clicks parsed: ids=%u", NIDS);

  // ---------------- phase 4: dedup
  {
    size_t est_surv = (size_t)(est_ev * 0.97);
    size_t pcap = (size_t)(est_surv / NT / P2 * 1.3) + 64;
    atomic<u32> next{0};
    parallel([&](int t) {
      TL& tl = TLS[t];
      init_bufs(tl.pb, P2, pcap);
      struct DT {
        u64 eh;
        const ERec* r;
      };
      vector<DT> tab;
      for (;;) {
        u32 p = next.fetch_add(1);
        if (p >= P1) break;
        size_t n = 0;
        for (int s = 0; s < NT; s++) n += TLS[s].eb[p].size();
        size_t cap = 64;
        while (cap < n * 2) cap <<= 1;
        tab.assign(cap, DT{0, nullptr});
        u64 mask = cap - 1;
        for (int s = 0; s < NT; s++) {
          TLS[s].eb[p].each([&](const ERec& r) {
            u64 i = r.eh & mask;
            for (;;) {
              DT& d = tab[i];
              if (!d.r) {
                d.eh = r.eh;
                d.r = &r;
                break;
              }
              if (d.eh == r.eh) {
                if (r.ingest < d.r->ingest) d.r = &r;
                break;
              }
              i = (i + 1) & mask;
            }
          });
        }
        for (size_t i = 0; i < cap; i++) {
          const ERec* r = tab[i].r;
          if (r && r->val != INVALID) tl.pb[r->person & (P2 - 1)].push(PE{r->person, r->val});
        }
      }
    });
  }
  LOG("dedup done");

  // ---------------- phase 5: sessions
  {
    u32 NL = (NIDS >> P2B) + 1;
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
      tl.c_sess.assign(csz, 0);
      tl.c_psess.assign(csz, 0);
      tl.c_rev.assign(csz, 0);
      tl.c_assist.assign(csz, 0);
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
      size_t n = tl.o_person.size();
      if (n == 0 && t != 0) return;
      vector<PCol> cols(6);
      cols[0] = PCol{"user_id", PT_BYTES, LT_STRING, false, string()};
      cols[1] = PCol{"day", PT_INT32, LT_DATE, false, string()};
      cols[2] = PCol{"sessions", PT_INT32, LT_NONE, false, string()};
      cols[3] = PCol{"events", PT_INT32, LT_NONE, false, string()};
      cols[4] = PCol{"revenue_cents", PT_INT64, LT_NONE, false, string()};
      cols[5] = PCol{"top_category", PT_BYTES, LT_STRING, true, string()};
      size_t namebytes = 0;
      for (size_t i = 0; i < n; i++) namebytes += NAMEN[tl.o_person[i]] + 4;
      string& c0 = cols[0].data;
      c0.reserve(namebytes);
      for (size_t i = 0; i < n; i++) {
        u32 pid = tl.o_person[i];
        put32(c0, NAMEN[pid]);
        c0.append(NAMEP[pid], NAMEN[pid]);
      }
      cols[1].data.assign((const char*)tl.o_day.data(), n * 4);
      cols[2].data.assign((const char*)tl.o_sess.data(), n * 4);
      cols[3].data.assign((const char*)tl.o_ev.data(), n * 4);
      cols[4].data.assign((const char*)tl.o_rev.data(), n * 8);
      vector<int8_t> present(n);
      for (size_t i = 0; i < n; i++) present[i] = tl.o_top[i] >= 0;
      string& c5 = cols[5].data;
      c5 = deflevels(present);
      for (size_t i = 0; i < n; i++)
        if (tl.o_top[i] >= 0) {
          const string& s = CATS[(size_t)tl.o_top[i]];
          put32(c5, (u32)s.size());
          c5 += s;
        }
      write_parquet(ud + "/part-" + to_string(t) + ".parquet", n, cols);
      wrote++;
    });
    // campaign_daily
    size_t csz = (size_t)(NCAMP + 1) * ND;
    vector<i64> cs(csz, 0), cps(csz, 0), crev(csz, 0), cas(csz, 0);
    for (auto& tl : TLS)
      for (size_t i = 0; i < csz; i++) {
        cs[i] += tl.c_sess[i];
        cps[i] += tl.c_psess[i];
        crev[i] += tl.c_rev[i];
        cas[i] += tl.c_assist[i];
      }
    vector<PCol> cols(7);
    cols[0] = PCol{"campaign_id", PT_BYTES, LT_STRING, false, string()};
    cols[1] = PCol{"channel", PT_BYTES, LT_STRING, false, string()};
    cols[2] = PCol{"day", PT_INT32, LT_DATE, false, string()};
    cols[3] = PCol{"sessions", PT_INT64, LT_NONE, false, string()};
    cols[4] = PCol{"purchasing_sessions", PT_INT64, LT_NONE, false, string()};
    cols[5] = PCol{"revenue_cents", PT_INT64, LT_NONE, false, string()};
    cols[6] = PCol{"assisted_sessions", PT_INT64, LT_NONE, false, string()};
    u64 rows = 0;
    for (u32 c = 0; c <= NCAMP; c++)
      for (int d = 0; d < ND; d++) {
        size_t i = (size_t)c * ND + d;
        if (cs[i] == 0 && cas[i] == 0) continue;
        const string& id = c == NCAMP ? string("organic") : CAMPIDS[c];
        const string& ch = c == NCAMP ? string("organic") : CHANNELS[c];
        put32(cols[0].data, (u32)id.size());
        cols[0].data += id;
        put32(cols[1].data, (u32)ch.size());
        cols[1].data += ch;
        put32(cols[2].data, (u32)(i32)(DAY0 + d));
        put64(cols[3].data, (u64)cs[i]);
        put64(cols[4].data, (u64)cps[i]);
        put64(cols[5].data, (u64)crev[i]);
        put64(cols[6].data, (u64)cas[i]);
        rows++;
      }
    write_parquet(cd + "/part-0.parquet", rows, cols);
  }
  LOG("output written");
  // skip destructors / munmap of the big mappings
  fflush(stderr);
  _exit(0);
}
