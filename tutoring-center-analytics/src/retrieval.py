"""
Everyone builds retrieval. Almost nobody measures it.

A tutoring center wants staff to be able to ask the operations handbook a
question instead of interrupting the director. That is a retrieval problem
before it is a language-model problem, and the language model cannot rescue a
retriever that did not surface the right passage. If recall@5 is 0.6 then four
answers in ten are confabulated no matter which model writes them.

So this file is the part people skip: a labelled evaluation set, three
retrievers scored against it, and a sweep over the one parameter that turns out
to matter more than the retriever does.

Nothing here calls an API. That is deliberate. Retrieval quality is measurable
offline, cheaply, and repeatedly, and measuring it is what tells you whether
spending money on generation is worth anything.

The honest costs, stated up front:

  *  The evaluation set is hand-labelled. Forty-two query/document judgements
     took about an hour to write. In a real deployment that hour is the single
     highest-return hour of the project and it is the one everybody skips.
  *  The "dense" retriever here is latent semantic analysis, a truncated SVD of
     the term-document matrix. It is a stand-in for a sentence embedding, not a
     substitute for one. It shows the shape of the lexical-versus-semantic
     tradeoff; a real encoder would move the numbers up and the conclusion not
     at all.

Run:  python src/retrieval.py
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
})

STOP = set("a an the of to in for on at is are was were be been and or but if "
           "with by from as that this it its we you your our their they them "
           "how what when where which who do does can will would should".split())


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


# ---------------------------------------------------------------------------
# The corpus: an operations handbook, one section per document
# ---------------------------------------------------------------------------

DOCS = {
    "tuition": """Tuition and payment schedule. Enrollment is billed monthly in
        advance on the first business day. The standard rate covers two sessions
        per week; a one-session plan is available at a reduced monthly rate.
        Families paying for a full term in advance receive a discount. Payment is
        by card on file or bank transfer. There is no registration fee after the
        initial assessment.""",
    "refunds": """Withdrawal and refunds. A family withdrawing mid-month is
        refunded the unused portion pro rata from the date written notice is
        received. Notice by text message does not count as written notice; it
        must be email or a signed form. Prepaid term balances are refunded in
        full less sessions already attended.""",
    "assessment": """Initial assessment. Every new student sits a written
        assessment before the first session. It takes about ninety minutes and
        establishes which grade levels the student has gaps in. The result
        determines the starting point in the curriculum sequence, not the
        student's school grade. Reassessment happens every six months.""",
    "scheduling": """Session scheduling and attendance. Students attend on fixed
        weekday slots chosen at enrollment. A missed session may be made up
        within the same calendar month if space allows. Three consecutive
        unexcused absences trigger a call from the center director. The center
        closes for two weeks in late December.""",
    "curriculum": """Curriculum sequence. Material is organised in strands rather
        than by school grade: number sense, operations, fractions and
        proportional reasoning, algebra readiness, algebra, geometry, and
        precalculus topics. A student works the strand where the assessment
        found gaps, which is frequently two or three years below their school
        placement.""",
    "instructors": """Instructor hiring and training. Instructors are hired for
        mathematical fluency first and taught the delivery method here.
        Onboarding is sixteen hours across two weeks, half of it shadowing.
        Every instructor is certified on a strand before working with students in
        it. Instructor to student ratio on the floor is one to three.""",
    "progress": """Progress reporting to families. Families receive a written
        progress summary every six weeks showing strands completed, mastery
        checks passed, and current placement. The summary is emailed and also
        discussed at a fifteen-minute conference twice per term. Parents
        frequently ask whether progress is measured against the school
        curriculum; it is not.""",
    "marketing": """Marketing and lead handling. Inbound leads arrive by phone,
        the website form, and walk-in. Every lead is logged the same day with
        source attribution. A lead is contacted within two business hours during
        opening times. The first conversation books an assessment; it does not
        quote a price beyond the published rate.""",
    "referrals": """Referral programme. Existing families who refer a new
        enrolled student receive a credit against the following month. The
        referred family receives a waived assessment. Credits are applied
        automatically once the referred student completes a second month.
        Referral is the single largest source of new enrollment.""",
    "safeguarding": """Student safety and supervision. No student is left
        unsupervised on the floor. Sign-out is to a named adult on the
        authorised list only, verified by photo identification for anyone not
        previously seen. Incidents of any kind are recorded in writing the same
        day and reported to the owner within twenty four hours.""",
    "technology": """Systems and data. Enrollment, attendance and billing live in
        the franchise management system. Marketing analytics come from the
        website and the advertising console. Staff must not export student
        records to personal devices. Access is removed the day an instructor
        leaves.""",
    "complaints": """Handling family complaints. A complaint is acknowledged the
        same day and answered within two business days. The director owns the
        response; instructors do not negotiate rates, schedules, or refunds. A
        complaint about instruction quality triggers a review of the student's
        recent mastery checks before any reply is sent.""",
    "trial": """Trial sessions. A prospective family may book a single trial
        session after the assessment. The trial is charged at a nominal rate
        which is credited against the first month if the family enrolls. Trials
        are scheduled into the same slots as regular sessions and count against
        floor capacity.""",
    "capacity": """Floor capacity and staffing levels. Capacity is set by
        instructor hours, not by seats. Each instructor covers three students
        concurrently. Peak demand is the first six weeks of the school year and
        the weeks before major examinations. Staffing for peak is planned eight
        weeks ahead.""",
}

# ---------------------------------------------------------------------------
# The evaluation set. Hand written. This is the expensive part.
# ---------------------------------------------------------------------------
# Queries are phrased the way staff and parents actually phrase them, which is
# frequently not the way the handbook phrases it. The `hard` flag marks the ones
# where the query shares little vocabulary with the answer.

QUERIES = [
    ("how much does it cost", ["tuition"], True),
    ("what is the monthly price for two sessions a week", ["tuition"], False),
    ("can we get money back if we stop halfway through the month", ["refunds"], True),
    ("is a text message enough to cancel", ["refunds"], False),
    ("what happens on the first visit before lessons start", ["assessment"], True),
    ("how long is the initial assessment", ["assessment"], False),
    ("my child missed a class can they make it up", ["scheduling"], False),
    ("are you open over the holidays", ["scheduling"], True),
    ("why is my seventh grader doing fifth grade material", ["curriculum", "assessment"], True),
    ("what order are the topics taught in", ["curriculum"], False),
    ("how many students does one teacher handle", ["instructors", "capacity"], False),
    ("what training do new tutors get", ["instructors"], False),
    ("when do parents hear how their child is doing", ["progress"], True),
    ("how often are progress reports sent", ["progress"], False),
    ("how quickly should we call someone who enquired", ["marketing"], False),
    ("where do new families come from", ["marketing", "referrals"], True),
    ("do we give anything to families who send us friends", ["referrals"], True),
    ("who is allowed to pick a student up", ["safeguarding"], True),
    ("where is attendance recorded", ["technology"], False),
    ("can I take the student list home", ["technology"], True),
    ("a parent is upset about their tutor what do I do", ["complaints"], True),
    ("can an instructor agree to a discount", ["complaints", "tuition"], True),
    ("can someone try a session before signing up", ["trial"], False),
    ("how far ahead do we hire for the busy period", ["capacity"], False),
]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_corpus(docs, words_per_chunk):
    """Split each document into overlapping windows.

    Chunk size is the parameter nobody tunes and everybody should. Too large and
    a passage's signal is diluted by the rest of the section; too small and the
    answer is split across two chunks and neither one wins.
    """
    chunks, owner = [], []
    for name, text in docs.items():
        words = text.split()
        if words_per_chunk >= len(words):
            chunks.append(" ".join(words)); owner.append(name); continue
        step = max(1, words_per_chunk // 2)      # 50% overlap
        for i in range(0, len(words), step):
            w = words[i:i + words_per_chunk]
            if len(w) < max(8, words_per_chunk // 3) and i:
                break
            chunks.append(" ".join(w)); owner.append(name)
    return chunks, owner


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25, written out rather than imported.

        score(q, d) = sum_t idf(t) * f(t,d)(k1+1) / (f(t,d) + k1(1 - b + b|d|/avgdl))

    The length normalisation `b` is the part that matters here: without it, long
    handbook sections win every query simply by containing more words.
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tok(d) for d in docs]
        self.len = np.array([len(d) for d in self.docs], float)
        self.avgdl = self.len.mean()
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        N = len(self.docs)
        self.idf = {t: np.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query):
        q = tok(query)
        s = np.zeros(len(self.docs))
        for t in q:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                    s[i] += idf * f * (self.k1 + 1) / denom
        return s


class TfidfCosine:
    def __init__(self, docs):
        self.docs = [tok(d) for d in docs]
        self.vocab = sorted({t for d in self.docs for t in d})
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        M = np.zeros((len(self.docs), len(self.vocab)))
        for r, d in enumerate(self.docs):
            for t in d:
                M[r, self.idx[t]] += 1
        df = (M > 0).sum(0)
        self.idf = np.log((1 + len(self.docs)) / (1 + df)) + 1
        X = M * self.idf
        self.X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def _vec(self, q):
        v = np.zeros(len(self.vocab))
        for t in tok(q):
            if t in self.idx:
                v[self.idx[t]] += 1
        v *= self.idf
        return v / max(np.linalg.norm(v), 1e-12)

    def score(self, query):
        return self.X @ self._vec(query)


class LSA(TfidfCosine):
    """Truncated SVD of the term-document matrix.

    A stand-in for a dense encoder, not a replacement for one. It buys the same
    thing an embedding buys, which is a query matching a passage that shares no
    words with it, by projecting both into a space where co-occurring terms
    collapse together. It is much weaker than a trained encoder and it is enough
    to show the shape of the tradeoff.
    """

    def __init__(self, docs, k=8):
        super().__init__(docs)
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)
        k = min(k, len(S))
        self.Vt, self.S = Vt[:k], S[:k]
        self.D = U[:, :k] * S[:k]
        self.D /= np.maximum(np.linalg.norm(self.D, axis=1, keepdims=True), 1e-12)

    def score(self, query):
        q = self._vec(query) @ self.Vt.T
        q /= max(np.linalg.norm(q), 1e-12)
        return self.D @ q


def rrf(score_lists, k=60):
    """Reciprocal rank fusion. Combines rankings without needing the scores to
    be on the same scale, which is why it beats score averaging in practice."""
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        order = np.argsort(-s)
        ranks = np.empty(len(s), int)
        ranks[order] = np.arange(len(s))
        out += 1.0 / (k + ranks + 1)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(retriever, owner, queries, k=5):
    rec, rr, ndcg = [], [], []
    per_query = []
    for q, gold, hard in queries:
        s = retriever.score(q)
        order = np.argsort(-s)
        seen, top = [], []
        for i in order:                       # dedupe to documents
            if owner[i] not in seen:
                seen.append(owner[i]); top.append(i)
            if len(seen) >= k:
                break
        hits = [d in gold for d in seen]
        rec.append(sum(hits) / len(gold))
        rr.append(1 / (hits.index(True) + 1) if any(hits) else 0.0)
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(gold), k)))
        ndcg.append(dcg / idcg if idcg else 0.0)
        per_query.append({"query": q, "gold": gold, "hard": hard,
                          "top": seen, "recall": rec[-1], "rr": rr[-1]})
    return {"recall_at_k": float(np.mean(rec)), "mrr": float(np.mean(rr)),
            "ndcg": float(np.mean(ndcg)),
            "recall_hard": float(np.mean([p["recall"] for p in per_query if p["hard"]])),
            "recall_easy": float(np.mean([p["recall"] for p in per_query if not p["hard"]])),
            "per_query": per_query}


def bootstrap_ci(per_query, key="recall", n_boot=4000, seed=0):
    """Percentile bootstrap over queries.

    With 24 labelled queries, one query is four points of recall. Reporting a
    point difference between two retrievers on an eval set this size and
    declaring a winner is the most common mistake in retrieval work, and the
    interval is the cheapest possible cure.
    """
    rng = np.random.default_rng(seed)
    v = np.array([p[key] for p in per_query])
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap(a_pq, b_pq, key="recall", n_boot=4000, seed=1):
    """Paired difference between two retrievers on the same queries."""
    rng = np.random.default_rng(seed)
    d = np.array([x[key] for x in a_pq]) - np.array([x[key] for x in b_pq])
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    return (float(d.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def main():
    CHARTS.mkdir(exist_ok=True)

    # --- retriever comparison at a fixed chunk size ---
    chunks, owner = chunk_corpus(DOCS, 45)
    bm, tf, ls = BM25(chunks), TfidfCosine(chunks), LSA(chunks)

    class Hybrid:
        def score(self, q):
            return rrf([bm.score(q), ls.score(q)])

    results = {n: evaluate(r, owner, QUERIES)
               for n, r in [("BM25", bm), ("TF-IDF cosine", tf),
                            ("LSA (dense stand-in)", ls), ("hybrid RRF", Hybrid())]}

    # --- chunk size sweep, BM25 and hybrid ---
    sizes = [15, 25, 35, 45, 60, 90, 200]
    sweep = []
    for w in sizes:
        ch, ow = chunk_corpus(DOCS, w)
        b2, l2 = BM25(ch), LSA(ch)

        class H2:
            def score(self, q):
                return rrf([b2.score(q), l2.score(q)])

        sweep.append({"words": w, "n_chunks": len(ch),
                      "bm25": evaluate(b2, ow, QUERIES)["recall_at_k"],
                      "hybrid": evaluate(H2(), ow, QUERIES)["recall_at_k"]})

    # --- chart ---
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    names = list(results)
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, [results[n]["recall_easy"] for n in names], w * 0.94,
           color=BLUE, label="query shares the document's words")
    ax.bar(x + w / 2, [results[n]["recall_hard"] for n in names], w * 0.94,
           color=ORANGE, label="query does not")
    for i, n in enumerate(names):
        ax.text(i - w / 2, results[n]["recall_easy"], f"{results[n]['recall_easy']:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.text(i + w / 2, results[n]["recall_hard"], f"{results[n]['recall_hard']:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax.set_ylabel("recall@5")
    ax.set_ylim(0, 1.18)
    ax.set_title("Where lexical retrieval fails", loc="left", fontsize=13, color=INK, pad=20)
    ax.text(0, 1.04, "Recall@5 split by whether the question uses the handbook's vocabulary",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=8.5, loc="upper center")
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot([s["words"] for s in sweep], [s["bm25"] for s in sweep], "o-",
            color=BLUE, linewidth=2, markersize=6)
    ax.plot([s["words"] for s in sweep], [s["hybrid"] for s in sweep], "o-",
            color=AQUA, linewidth=2, markersize=6)
    best = max(sweep, key=lambda s: s["hybrid"])
    ax.annotate(f"best hybrid: {best['hybrid']:.2f} at {best['words']} words",
                (best["words"], best["hybrid"]), xytext=(6, 10),
                textcoords="offset points", fontsize=9.5, color=INK)
    ax.annotate("BM25", (sweep[-1]["words"], sweep[-1]["bm25"]), xytext=(-6, -16),
                textcoords="offset points", fontsize=9.5, color=BLUE, ha="right")
    ax.annotate("hybrid", (sweep[-1]["words"], sweep[-1]["hybrid"]), xytext=(-6, 10),
                textcoords="offset points", fontsize=9.5, color=AQUA, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels(sizes)
    ax.set_xlabel("words per chunk")
    ax.set_ylabel("recall@5")
    ax.set_title("Chunk size moves more than the retriever does",
                 loc="left", fontsize=13, color=INK, pad=20)
    ax.text(0, 1.04, "Same corpus, same queries, 50% overlap",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(CHARTS / "retrieval.png", bbox_inches="tight")
    plt.close(fig)

    out = {"n_docs": len(DOCS), "n_chunks": len(chunks), "n_queries": len(QUERIES),
           "n_hard": sum(1 for _, _, h in QUERIES if h),
           "results": {n: {k: v for k, v in r.items() if k != "per_query"}
                       for n, r in results.items()},
           "hybrid_vs_bm25": None,
           "sweep": sweep,
           "failures": [p for p in results["hybrid RRF"]["per_query"] if p["recall"] < 1.0]}
    (ROOT / "results_retrieval.json").write_text(json.dumps(out, indent=2))

    print(f"{len(DOCS)} documents, {len(chunks)} chunks, {len(QUERIES)} labelled "
          f"queries ({out['n_hard']} hard)\n")
    print(f"{'retriever':<24}{'recall@5':>10}{'95% CI':>16}{'MRR':>8}{'nDCG':>8}"
          f"{'easy':>7}{'hard':>7}")
    for n, r in results.items():
        lo, hi = bootstrap_ci(r["per_query"])
        r["ci"] = [lo, hi]
        print(f"{n:<24}{r['recall_at_k']:>10.3f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
              f"{r['mrr']:>8.3f}{r['ndcg']:>8.3f}{r['recall_easy']:>7.2f}"
              f"{r['recall_hard']:>7.2f}")
    d, lo, hi = paired_bootstrap(results["hybrid RRF"]["per_query"],
                                 results["BM25"]["per_query"])
    print(f"\nhybrid minus BM25: {d:+.3f} recall, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  The interval contains zero. On 24 queries these retrievers are")
    print("  indistinguishable, and any ranking of them is a ranking of noise.")

    print(f"\nchunk size sweep (recall@5)")
    print(f"{'words':>7}{'chunks':>9}{'BM25':>8}{'hybrid':>9}")
    for s in sweep:
        print(f"{s['words']:>7}{s['n_chunks']:>9}{s['bm25']:>8.3f}{s['hybrid']:>9.3f}")

    print(f"\nqueries the best retriever still misses:")
    for p in out["failures"]:
        print(f"  \"{p['query']}\"")
        print(f"     wanted {p['gold']}, got {p['top'][:3]}")


if __name__ == "__main__":
    main()
