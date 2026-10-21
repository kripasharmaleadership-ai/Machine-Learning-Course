"""
LAB 02 - Traditional Programming vs Machine Learning vs Generative AI
=====================================================================

THE THREE PARADIGMS IN ONE PICTURE

    Traditional programming :  RULES + DATA        ->  ANSWERS
    Machine learning        :  DATA  + ANSWERS     ->  RULES (a model)
    Generative AI           :  LOTS OF DATA        ->  a model that CREATES NEW DATA
                               (text, images, code) that looks like the training data

    Traditional / classic ML answer : "Is this email spam?"          -> yes / no
    Generative AI answers           : "Write me a reply to this email" -> new text

LEVELS IN THIS LAB
    EASY   : Spam filter written by hand (if/else) vs a spam filter that LEARNS.
    MEDIUM : Spammers change their words (concept drift). Who adapts faster?
             We also open the ML model and read the "rules" it learned.
    HARD   : Build a tiny Generative AI (a word-level language model) from scratch,
             sample text with temperature, then an engineer's decision guide:
             when to use rules vs ML vs GenAI in a real company.
"""
# %%
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from ml_lab_utils import RANDOM_STATE, banner, note, show, step

# %% ---------------------------------------------------------------------------
# A small, hand-labelled email dataset (1 = spam, 0 = normal "ham")
# In real life you'd have 100k+ emails. Small here so you can READ every row.
# ------------------------------------------------------------------------------
TRAIN_EMAILS = [
    ("WIN a FREE iPhone now, click here", 1),
    ("Congratulations you won a lottery prize claim now", 1),
    ("Limited offer: cheap meds, buy now", 1),
    ("You have been selected for a cash reward, click the link", 1),
    ("FREE gift card waiting for you, claim today", 1),
    ("Earn money fast from home, no experience needed", 1),
    ("Exclusive deal just for you, act now before it expires", 1),
    ("Claim your prize, urgent response required", 1),
    ("Lowest price guaranteed, buy now and save 90 percent", 1),
    ("Your account won a bonus, click to claim cash", 1),
    ("Hot singles in your area, click here", 1),
    ("Get rich quick with this one simple trick", 1),
    ("Meeting moved to 3pm, see agenda attached", 0),
    ("Can you review the quarterly report before Friday", 0),
    ("Lunch tomorrow with the data team?", 0),
    ("Here are the notes from today's standup", 0),
    ("Please approve the purchase order for new laptops", 0),
    ("The deployment finished successfully last night", 0),
    ("Reminder: performance review forms due next week", 0),
    ("Thanks for the update, I will check the dashboard", 0),
    ("Can we reschedule our one on one to Thursday", 0),
    ("Attached is the invoice for September services", 0),
    ("Welcome to the team, your onboarding starts Monday", 0),
    ("The client loved the demo, next steps below", 0),
]
TEST_EMAILS = [
    ("Click now to claim your free reward", 1),
    ("Congratulations, cash prize waiting, act fast", 1),
    ("Buy cheap watches, huge discount today", 1),
    ("You are a winner! Claim your gift", 1),
    ("Please review the attached contract draft", 0),
    ("Team lunch is moved to Friday", 0),
    ("The free coffee machine on floor 2 is fixed", 0),   # 'free' but NOT spam
    ("Click the link to join the 10am meeting", 0),       # 'click' but NOT spam
]
train_df = pd.DataFrame(TRAIN_EMAILS, columns=["text", "is_spam"])
test_df = pd.DataFrame(TEST_EMAILS, columns=["text", "is_spam"])

# %% =========================================================================
# LEVEL 1 (EASY): Rules written by a human vs rules learned from data
# ============================================================================
banner("Spam filter: hand-written rules vs learned model", "EASY")

step("Approach A - Traditional programming (a developer writes IF/ELSE)")
SPAM_KEYWORDS = ["free", "click", "win", "prize", "cash", "buy now"]


def rule_based_filter(text: str) -> int:
    """Classic programming: the developer IS the intelligence."""
    t = text.lower()
    return int(any(word in t for word in SPAM_KEYWORDS))


rule_preds = test_df["text"].apply(rule_based_filter)

step("Approach B - Machine learning (give examples, model finds the rules)")
ml_model = make_pipeline(CountVectorizer(lowercase=True), MultinomialNB())
ml_model.fit(train_df["text"], train_df["is_spam"])
ml_preds = ml_model.predict(test_df["text"])

results = test_df.assign(rules=rule_preds, ml=ml_preds)
print(results.to_string(index=False))
show("Rule-based accuracy", accuracy_score(test_df["is_spam"], rule_preds))
show("ML accuracy", accuracy_score(test_df["is_spam"], ml_preds))
note("""The keyword rules flag 'free coffee machine' and 'click the link to join
the meeting' as spam - a human can't list every exception. The ML model weighs
ALL words together ('meeting', 'floor', 'join' push towards normal), so context
wins over a single keyword.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): The world changes - concept drift
# ============================================================================
banner("Spammers change tactics: who adapts faster?", "MEDIUM")

# A few labelled examples of the new wave (what a moderation team would label)
wave_train = pd.DataFrame([
    ("Verify your crypto wallet to unlock bitcoin bonus", 1),
    ("Crypto airdrop ready, connect wallet for free tokens", 1),
    ("Double your bitcoin, guaranteed crypto returns", 1),
    ("Finance asked about crypto regulations for the audit", 0),
], columns=["text", "is_spam"])
# Fresh emails arriving next week
wave_test = pd.DataFrame([
    ("Connect wallet to receive bitcoin airdrop tokens", 1),
    ("Wallet locked, verify bitcoin identity", 1),
    ("Mint crypto tokens, connect wallet", 1),
    ("Please review the crypto audit numbers", 0),
], columns=["text", "is_spam"])

step("Before anyone reacts: both filters face the new spam wave")
show("Rules - spam caught (recall)", recall_score(wave_test["is_spam"], wave_test["text"].apply(rule_based_filter)))
show("Old ML - spam caught (recall)", recall_score(wave_test["is_spam"], ml_model.predict(wave_test["text"])))
note("Both fail: neither has ever seen 'crypto', 'wallet', 'bitcoin'.")

step("Reaction: Rules need a developer + code release. ML needs new labelled data.")
retrained = make_pipeline(CountVectorizer(), MultinomialNB())
retrained.fit(pd.concat([train_df, wave_train])["text"], pd.concat([train_df, wave_train])["is_spam"])
show("Retrained ML - spam caught (recall)", recall_score(wave_test["is_spam"], retrained.predict(wave_test["text"])))
show("Retrained ML - accuracy on wave", accuracy_score(wave_test["is_spam"], retrained.predict(wave_test["text"])))
note("""With just 4 new labelled emails, the retrained model catches the new wave
and still lets the finance email through. In production this is a scheduled
RETRAINING pipeline - no code change, just new data. Rules would need someone
to rewrite and redeploy code for every new spam trend.""")

step("Peek inside: what 'rules' did the ML model learn?")
vec = retrained.named_steps["countvectorizer"]
nb = retrained.named_steps["multinomialnb"]
words = np.array(vec.get_feature_names_out())
# log P(word | spam) - log P(word | ham): positive = spammy word
spamminess = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
order = np.argsort(spamminess)
print("  Most SPAM-like words :", ", ".join(words[order[-10:]][::-1]))
print("  Most NORMAL words    :", ", ".join(words[order[:10]]))
note("""The model's 'rules' are weights per word - learned, not typed. This is the
key difference: in ML the logic lives in the DATA, in traditional software
it lives in the CODE.""")

# %% =========================================================================
# LEVEL 3 (HARD): Build a tiny Generative AI from scratch
# ============================================================================
banner("Generative AI from scratch: a word-level language model", "HARD")
note("""Classifiers model P(label | input): 'given this email, is it spam?'
Generative models model P(data) itself: 'what does a plausible email look like?'
An LLM like GPT is, at its core, a VERY big version of what we build below:
a model that predicts the NEXT WORD given previous words, used in a loop.""")

SUPPORT_REPLIES = """
thank you for contacting support . we are sorry for the delay with your order .
thank you for your patience . your refund has been processed today .
we are sorry for the inconvenience . your order has been shipped today .
thank you for reaching out . your ticket has been escalated to our team .
we are happy to help . your refund will arrive in five business days .
sorry for the trouble . our team will contact you within one business day .
thank you for your feedback . we are happy to hear the issue is resolved .
your order has been delayed . we are sorry and will update you soon .
""".split()

step("Step 1: 'Pre-training' - count which word follows which word")
# Bigram model: P(next_word | previous word). GPT does the same with a context
# of thousands of previous tokens instead of just one word.
counts: dict[str, Counter] = defaultdict(Counter)
for prev, nxt in zip(SUPPORT_REPLIES, SUPPORT_REPLIES[1:]):
    counts[prev][nxt] += 1
show("Training tokens (words)", len(SUPPORT_REPLIES))
show("Vocabulary size", len(set(SUPPORT_REPLIES)))
print("  After 'your' the model saw ->", dict(counts["your"]))


def next_word_distribution(prev: str, temperature: float):
    """Turn counts into probabilities. Temperature reshapes them:
    low T  -> sharpen: almost always pick the most likely word (safe, repetitive)
    high T -> flatten: rare words get a real chance (creative, riskier)."""
    options = counts[prev]
    words_ = list(options.keys())
    logits = np.log(np.array(list(options.values()), dtype=float)) / temperature
    probs = np.exp(logits - logits.max())
    return words_, probs / probs.sum()


def generate(start: str, temperature: float, seed: int, max_words: int = 25) -> str:
    """The generation loop of every LLM: predict -> sample -> append -> repeat."""
    gen_rng = np.random.default_rng(seed)
    out = start.split()
    for _ in range(max_words):
        words_, probs = next_word_distribution(out[-1], temperature)
        out.append(str(gen_rng.choice(words_, p=probs)))
        if out[-1] == ".":
            break
    return " ".join(out)


step("Step 2: What temperature does to the next-word probabilities after 'your'")
for T in [0.2, 1.0, 3.0]:
    words_, probs = next_word_distribution("your", T)
    top = sorted(zip(words_, probs), key=lambda wp: -wp[1])
    print(f"  T={T:<4}", "  ".join(f"{w}:{p:.2f}" for w, p in top))

step("Step 3: Generate NEW sentences (5 samples per temperature)")
for T in [0.2, 1.0, 3.0]:
    print(f"\n  temperature = {T}")
    for s in range(5):
        print("   ", generate("thank you", T, seed=RANDOM_STATE + s))
note("""Low temperature -> nearly the same safe sentence every time. Higher
temperature -> more variety and some odd combinations. That is EXACTLY the
'temperature' setting in the OpenAI / Anthropic / Gemini APIs. Our model is
a count table; a modern LLM is a transformer with billions of learned
weights, but the generation loop is the same.""")

step("Step 4: Hallucination in miniature")
original = {" ".join(SUPPORT_REPLIES[i:i + 6]) for i in range(len(SUPPORT_REPLIES))}
for s in range(40):
    sentence = generate("your", 1.5, seed=s)
    words_s = sentence.split()
    novel = all(" ".join(words_s[i:i + 6]) not in original for i in range(max(1, len(words_s) - 5)))
    if novel and len(words_s) >= 6:
        print("   ", sentence)
        break
note("""Every word-pair above was seen in training, yet the full sentence never
was - it confidently states things that make no sense or never happened
(a support ticket 'shipped today'). A generative model optimizes for 'sounds plausible', not 'is
true'. That is why enterprise GenAI needs grounding (RAG), guardrails and
human review.""")

# %% ---------------------------------------------------------------------------
# Principal engineer's decision guide
# ------------------------------------------------------------------------------
step("Step 4: Decision guide - rules vs ML vs GenAI")
guide = pd.DataFrame([
    ["Logic is known & stable (tax rules, VAT)", "Traditional code", "Cheap, exact, auditable"],
    ["Pattern exists in data, too complex to hand-write", "Classic ML", "Learns from labelled history"],
    ["Output is a number/label/score (fraud, churn, price)", "Classic ML", "Fast, cheap, measurable"],
    ["Output is free text/images/code (draft reply, summary)", "Generative AI", "Creates new content"],
    ["Input is unstructured language, few/no labels", "GenAI (zero/few-shot)", "Pre-trained knowledge"],
    ["Must be 100% deterministic & explainable", "Code or simple ML", "GenAI is probabilistic"],
    ["Millions of predictions/sec at < 5 ms", "Classic ML", "LLMs are slow & costly per call"],
], columns=["Situation", "Use", "Why"])
print(guide.to_string(index=False))

banner("KEY TAKEAWAYS")
print("""  1. Traditional: humans write rules. ML: data writes the rules. GenAI: model writes NEW content.
  2. ML adapts to change by RETRAINING on new data instead of rewriting code.
  3. GenAI = predict the next token, repeatedly. Temperature controls creativity.
  4. GenAI optimizes for plausibility, not truth -> grounding & guardrails are mandatory.
  5. Most enterprise systems combine all three: code for rules, ML for scores, GenAI for language.""")
