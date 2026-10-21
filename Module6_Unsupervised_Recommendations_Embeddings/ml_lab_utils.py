"""
ml_lab_utils.py - Shared helpers for every lab in Module 6.

What lives here:
    1. Pretty console output helpers  -> banner(), step(), note(), show()
    2. Plot helper                    -> save_fig()
    3. Synthetic CUSTOMERS            -> make_customers()        (5 hidden segments we can check clusters against)
    4. Synthetic EXPENSE CLAIMS       -> make_expense_claims()   (planted anomalies of 4 kinds)
    5. Synthetic RATINGS              -> make_ratings()          (users with hidden tastes; movies with genres)
    6. Synthetic PRODUCT CATALOGUE    -> make_product_catalogue() (titles + descriptions with SYNONYMS)
    7. Synthetic KNOWLEDGE BASES      -> make_knowledge_base("hr" | "it") (articles + paraphrased questions
                                         with the gold article and the answer fact: for RAG evaluation)

Why synthetic data?
    Unsupervised learning has no labels, so normally nobody can say whether a clustering, an anomaly
    score or a retrieval result is RIGHT. Here every dataset carries a hidden ground truth, so each
    method can be checked. Everything runs offline.

You do NOT need to read this file before running the labs.
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

IN_NOTEBOOK = "ipykernel" in sys.modules
try:
    import matplotlib

    if not IN_NOTEBOOK:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False


def optional_import(name: str):
    """Return the module if installed (e.g. sentence_transformers), else None. Labs never require it."""
    try:
        return __import__(name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 1-2. Console and plot helpers
# ---------------------------------------------------------------------------
def banner(title: str, level: str | None = None) -> None:
    tag = f"[{level}] " if level else ""
    line = "=" * 78
    print(f"\n{line}\n  {tag}{title}\n{line}")


def step(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


def note(text: str, keep_lines: bool = False) -> None:
    text = textwrap.dedent(text).strip()
    body = text if keep_lines else textwrap.fill(text, width=74)
    print(textwrap.indent(body, "  >> ", predicate=lambda _: True))


def show(label: str, value) -> None:
    if isinstance(value, float):
        value = f"{value:,.4f}"
    print(f"  {label:<46}: {value}")


def save_fig(fig, name: str) -> None:
    if fig is None:
        return
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [plot saved] outputs/{path.name}")
    if not IN_NOTEBOOK:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Customers with hidden segments
# ---------------------------------------------------------------------------
SEGMENTS = {   # persona: (age mean, income k mean, monthly spend per category, visits / month, online share)
    "young digital": (26, 38, {"groceries": 180, "electronics": 140, "fashion": 160, "travel": 40, "home": 30}, 6, 0.85),
    "family shopper": (41, 62, {"groceries": 620, "electronics": 60, "fashion": 120, "travel": 60, "home": 180}, 14, 0.35),
    "affluent traveller": (52, 140, {"groceries": 300, "electronics": 120, "fashion": 260, "travel": 520, "home": 120}, 5, 0.55),
    "bargain senior": (69, 28, {"groceries": 260, "electronics": 15, "fashion": 30, "travel": 25, "home": 50}, 10, 0.10),
    "home improver": (45, 75, {"groceries": 280, "electronics": 90, "fashion": 50, "travel": 40, "home": 480}, 7, 0.45),
}
SPEND_COLS = ["groceries", "electronics", "fashion", "travel", "home"]


def make_customers(n: int = 3000, seed: int = RANDOM_STATE, outliers: int = 0) -> pd.DataFrame:
    """Retail loyalty-card customers. Each belongs to one of 5 HIDDEN personas (column `_segment`,
    never used for training: only to check what clustering finds). Spend is right-skewed.
    outliers > 0 adds a few extreme customers (resellers, data errors)."""
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    seg = rng.choice(names, n, p=[.24, .26, .12, .18, .20])
    rows = []
    for s in seg:
        age, inc, spend, visits, online = SEGMENTS[s]
        row = {"age": float(np.clip(rng.normal(age, 9), 18, 92)),
               "income_k": float(np.clip(rng.lognormal(np.log(inc), 0.35), 8, 600))}
        for c in SPEND_COLS:
            row[c] = float(np.clip(rng.lognormal(np.log(spend[c]), 0.55), 0, 20000))
        row["visits_per_month"] = float(np.clip(rng.normal(visits, 3.5), 0.5, 40))
        row["online_share"] = float(np.clip(rng.normal(online, 0.15), 0, 1))
        row["_segment"] = s
        rows.append(row)
    df = pd.DataFrame(rows)
    if outliers:
        idx = rng.choice(n, outliers, replace=False)
        df.loc[idx, "electronics"] *= rng.uniform(15, 40, outliers)
        df.loc[idx, "visits_per_month"] = rng.uniform(25, 40, outliers)
        df.loc[idx, "_segment"] = "outlier (reseller)"
    df.insert(0, "customer_id", [f"C{100000 + i}" for i in range(n)])
    return df.round(2)


# ---------------------------------------------------------------------------
# 4. Expense claims with planted anomalies
# ---------------------------------------------------------------------------
EXPENSE_CATEGORIES = {"meals": 35, "taxi": 28, "hotel": 140, "flight": 320, "office_supplies": 45, "training": 400}


def make_expense_claims(n: int = 8000, seed: int = RANDOM_STATE, anomaly_share: float = 0.012) -> pd.DataFrame:
    """Employee expense claims. `_anomaly` (hidden, for checking only) marks planted problems of 4 kinds:
        inflated   : an amount far above normal for its category
        weekend    : a large claim submitted on a weekend at night
        split      : training courses just under the approval limit ($500): normal-looking amounts
        mismatch   : an expensive category claimed by a junior with no travel that month"""
    rng = np.random.default_rng(seed)
    cats = list(EXPENSE_CATEGORIES)
    cat = rng.choice(cats, n, p=[.34, .22, .16, .10, .12, .06])
    base = pd.Series(cat).map(EXPENSE_CATEGORIES).to_numpy()
    level = rng.choice([1, 2, 3, 4], n, p=[.35, .35, .2, .1])
    amount = np.clip(rng.lognormal(np.log(base * (1 + 0.15 * (level - 1))), 0.35), 2, None)
    weekday = rng.choice(7, n, p=[.19, .19, .19, .19, .18, .03, .03])
    hour = np.clip(rng.normal(14, 3, n), 6, 22).round()
    travel_days = np.where(np.isin(cat, ["hotel", "flight", "taxi"]), rng.integers(1, 6, n), rng.integers(0, 3, n))
    kind = np.array(["normal"] * n, dtype=object)
    k = int(n * anomaly_share)
    idx = rng.choice(n, 4 * k, replace=False).reshape(4, k)
    amount[idx[0]] *= rng.uniform(4, 9, k)
    kind[idx[0]] = "inflated"
    weekday[idx[1]], hour[idx[1]] = rng.choice([5, 6], k), rng.choice([0, 1, 2, 23], k)
    amount[idx[1]] *= rng.uniform(2, 4, k)
    kind[idx[1]] = "weekend"
    amount[idx[2]], cat[idx[2]] = rng.uniform(470, 499, k), "training"           # a normal amount for training
    kind[idx[2]] = "split"
    cat[idx[3]], level[idx[3]], travel_days[idx[3]] = "flight", 1, 0
    amount[idx[3]] = rng.uniform(600, 1500, k)
    kind[idx[3]] = "mismatch"
    return pd.DataFrame({"claim_id": [f"EX{700000 + i}" for i in range(n)], "category": cat, "amount": amount.round(2),
                         "employee_level": level, "weekday": weekday, "hour": hour.astype(int),
                         "travel_days_this_month": travel_days, "_anomaly": kind})


# ---------------------------------------------------------------------------
# 5. Ratings: users with hidden tastes
# ---------------------------------------------------------------------------
GENRES = ["action", "comedy", "drama", "romance", "sci-fi", "horror", "documentary", "animation"]
_GENRE_WORDS = {
    "action": ["explosive", "chase", "heist", "mercenary", "showdown"], "comedy": ["hilarious", "quirky", "misadventure", "prank", "witty"],
    "drama": ["moving", "family", "struggle", "redemption", "courtroom"], "romance": ["love", "wedding", "heartfelt", "summer", "letters"],
    "sci-fi": ["space", "robot", "future", "alien", "time-travel"], "horror": ["haunted", "terrifying", "curse", "night", "cabin"],
    "documentary": ["true", "nature", "investigation", "history", "portrait"], "animation": ["animated", "kids", "talking", "magical", "adventure"],
}


def make_ratings(n_users: int = 800, n_items: int = 400, seed: int = RANDOM_STATE, density: float = 0.04):
    """A movie-rating platform. Each movie has 1-2 genres; each user has a HIDDEN taste vector over the
    8 genres. Rating (1-5) = taste . genres + item quality + user leniency + noise. Popular movies are
    rated far more often (long tail). Returns (ratings long table, movies table, true expected ratings matrix)."""
    rng = np.random.default_rng(seed)
    g = len(GENRES)
    item_genres = np.zeros((n_items, g))
    for i in range(n_items):
        item_genres[i, rng.choice(g, rng.choice([1, 2], p=[.6, .4]), replace=False)] = 1
    item_genres /= item_genres.sum(axis=1, keepdims=True)
    tastes = rng.dirichlet(np.full(g, 0.4), n_users) * 3 - 0.6                      # strong, peaked preferences
    quality = rng.normal(0, 0.4, n_items)
    leniency = rng.normal(0, 0.3, n_users)
    true = 3.1 + 1.6 * (tastes @ item_genres.T) + quality[None, :] + leniency[:, None]
    popularity = rng.pareto(1.2, n_items) + 0.2
    p_item = popularity / popularity.sum()
    activity = rng.lognormal(0, 0.8, n_users)
    n_per_user = np.clip((activity / activity.mean() * density * n_items).astype(int), 5, n_items // 2)
    rows = []
    for u in range(n_users):
        seen = rng.choice(n_items, n_per_user[u], replace=False, p=p_item)
        r = np.clip(np.round(true[u, seen] + rng.normal(0, 0.6, len(seen))), 1, 5)
        rows += [(u, int(i), float(v)) for i, v in zip(seen, r)]
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating"])
    titles, descs = [], []
    for i in range(n_items):
        gs = [GENRES[j] for j in np.flatnonzero(item_genres[i])]
        words = [w for gg in gs for w in rng.choice(_GENRE_WORDS[gg], 2, replace=False)]
        titles.append(f"The {rng.choice(words).title()} {rng.choice(['Story', 'Files', 'Project', 'Road', 'Hour', 'Club'])} {i}")
        descs.append(" ".join(words + gs))
    movies = pd.DataFrame({"item": range(n_items), "title": titles, "genres": ["|".join(GENRES[j] for j in np.flatnonzero(item_genres[i]))
                                                                               for i in range(n_items)],
                           "description": descs, "popularity_weight": p_item})
    return ratings, movies, np.clip(true, 1, 5)


# ---------------------------------------------------------------------------
# 6. A product catalogue with synonyms (semantic search)
# ---------------------------------------------------------------------------
CATALOGUE = {   # subcategory: (category, synonym group for the product noun, feature phrases)
    "headphones": ("electronics", ["headphones", "earphones", "headset", "earbuds", "cans"],
                   ["noise cancelling", "wireless bluetooth", "over-ear comfort", "long battery", "studio sound"]),
    "laptop": ("electronics", ["laptop", "notebook", "ultrabook", "portable computer"],
               ["16gb memory", "fast processor", "lightweight aluminium", "long battery", "retina display"]),
    "smartwatch": ("electronics", ["smartwatch", "fitness watch", "activity tracker", "wearable"],
                   ["heart rate monitor", "gps tracking", "sleep tracking", "waterproof", "long battery"]),
    "running shoes": ("sports", ["running shoes", "trainers", "sneakers", "joggers"],
                      ["lightweight cushioning", "breathable mesh", "marathon", "trail grip", "arch support"]),
    "yoga mat": ("sports", ["yoga mat", "exercise mat", "fitness mat", "pilates mat"],
                 ["non-slip", "extra thick", "eco-friendly", "carry strap", "joint support"]),
    "tent": ("outdoor", ["tent", "shelter", "camping tent", "backpacking tent"],
             ["waterproof", "two person", "lightweight", "easy setup", "four season"]),
    "coffee maker": ("kitchen", ["coffee maker", "espresso machine", "coffee machine", "brewer"],
                     ["programmable timer", "milk frother", "barista style", "thermal carafe", "single serve"]),
    "blender": ("kitchen", ["blender", "smoothie maker", "food processor", "mixer"],
                ["high speed", "glass jug", "ice crushing", "easy clean", "portable"]),
    "sofa": ("home", ["sofa", "couch", "settee", "loveseat"],
             ["three seater", "velvet", "corner", "sofa bed", "stain resistant"]),
    "desk lamp": ("home", ["desk lamp", "reading light", "table lamp", "task light"],
                  ["led", "dimmable", "usb charging", "adjustable arm", "warm light"]),
}
BRANDS = ["Nordwave", "Altura", "Kestrel", "Brightline", "Vantor", "Solace", "Peakform", "Orbis", "Lumen", "Tavera"]


GENERIC_PHRASES = ["compact design", "premium build", "easy to use", "great value", "lightweight", "portable",
                   "modern look", "durable materials"]


def make_product_catalogue(per_sub: int = 60, seed: int = RANDOM_STATE, nouns_per_product: int = 1) -> pd.DataFrame:
    """An online shop's catalogue: 10 kinds (subcategories) x per_sub products. Each supplier calls its
    product by ONE of the kind's synonyms ('earbuds', 'headset', 'earphones'...) and adds generic phrases
    shared across kinds ('compact design', 'portable'): the vocabulary-mismatch problem semantic search must
    solve. nouns_per_product=2 mentions two synonyms (like reviews and old listings do): used for the
    unlabelled 'history' text that embeddings are learned from."""
    rng = np.random.default_rng(seed)
    rows = []
    for sub, (cat, nouns, feats) in CATALOGUE.items():
        for _ in range(per_sub):
            n1, n2 = rng.choice(nouns, 2, replace=False)
            f = rng.choice(feats, 2, replace=False)
            g = rng.choice(GENERIC_PHRASES, 2, replace=False)
            brand = rng.choice(BRANDS)
            title = f"{brand} {f[0]} {n1}"
            second = f" A great {n2} for everyday use." if nouns_per_product == 2 else " Great for everyday use."
            desc = (f"This {n1} offers {f[1]}, {g[0]} and {g[1]}.{second} From {brand}. "
                    f"{rng.choice(['Free delivery.', 'Two year warranty.', 'Best seller.', 'New arrival.', ''])}")
            price = float(np.round(rng.lognormal(np.log({"electronics": 180, "sports": 60, "outdoor": 150, "kitchen": 90,
                                                         "home": 220}[cat]), 0.45), 2))
            rows.append({"title": title, "description": desc.strip(), "category": cat, "subcategory": sub,
                         "brand": brand, "price": price})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "product_id", [f"P{10000 + i}" for i in range(len(df))])
    return df


# ---------------------------------------------------------------------------
# 7. Knowledge bases with gold questions (document retrieval / RAG)
# ---------------------------------------------------------------------------
_SYN = {
    "hr": {"leave": ["leave", "time off", "holiday", "vacation", "days off"], "manager": ["manager", "supervisor", "line manager", "team lead"],
           "request": ["request", "apply for", "ask for", "book"], "pay": ["pay", "salary", "wages", "compensation"],
           "employee": ["employee", "staff member", "colleague", "worker"], "claim": ["claim", "reimbursement", "refund"],
           "remote": ["remote work", "working from home", "home office", "hybrid work"], "doctor": ["doctor", "GP", "physician"]},
    "it": {"password": ["password", "passcode", "login credentials", "sign-in details"], "laptop": ["laptop", "computer", "notebook", "work machine"],
           "vpn": ["VPN", "secure tunnel", "remote access client"], "printer": ["printer", "print device", "copier"],
           "email": ["email", "mailbox", "Outlook", "inbox"], "ticket": ["ticket", "support request", "help desk case"],
           "install": ["install", "set up", "download", "add"], "wifi": ["wifi", "wireless network", "Wi-Fi", "internet connection"]},
}
_KB = {
    "hr": [
        ("Annual leave allowance", ["Every {employee} receives 25 days of paid {leave} per year.", "Unused {leave} up to 5 days carries over to the next year.",
                                    "{leave} accrues monthly from the first day of employment."],
         ["How many {leave} days do I get per year?", "What is my yearly {leave} entitlement?", "how much paid {leave} does an {employee} get"], "25 days"),
        ("Booking leave", ["To {request} {leave}, use the HR portal at least two weeks in advance.", "Your {manager} approves or declines within 3 working days.",
                           "Short notice {leave} needs a phone call to your {manager}."],
         ["How do I {request} {leave}?", "What is the process to {request} some {leave}?", "how far in advance must I {request} {leave}"], "two weeks"),
        ("Sick leave", ["If you are ill, inform your {manager} before 10am on the first day.", "From day 4 of illness a note from a {doctor} is required.",
                        "Sick days are paid in full for up to 20 days a year."],
         ["When do I need a note from a {doctor} when sick?", "Do I need a {doctor} certificate for illness?", "how many paid sick days are there"], "day 4"),
        ("Parental leave", ["Birth parents receive 26 weeks of parental {leave} at full {pay}.", "Partners receive 6 weeks of paid parental {leave}.",
                            "Parental {leave} can be split into up to three blocks."],
         ["How long is parental {leave}?", "How much {leave} do new parents get?", "is parental {leave} at full {pay}"], "26 weeks"),
        ("Expense claims", ["Submit every {claim} within 30 days with the original receipt.", "Meals are covered up to 40 euros per day when travelling.",
                            "Approved {claim} payments arrive with the next {pay} run."],
         ["How long do I have to submit an expense {claim}?", "What is the deadline for a travel {claim}?", "what is the daily meal limit when travelling"], "30 days"),
        ("Remote work policy", ["{remote} is allowed up to 3 days per week with {manager} agreement.", "The company provides a monitor and chair for {remote}.",
                                "Core hours of 10am to 3pm apply on {remote} days."],
         ["How many days a week can I do {remote}?", "Is {remote} allowed and how often?", "does the company pay for equipment for {remote}"], "3 days"),
        ("Overtime", ["Overtime is paid at 1.5 times the normal hourly {pay}.", "Overtime must be agreed with your {manager} in advance.",
                      "Instead of {pay}, overtime can be taken as extra {leave}."],
         ["How is overtime paid?", "What rate is overtime {pay}?", "can I take overtime as {leave} instead of money"], "1.5 times"),
        ("Pay day and payslips", ["{pay} is paid on the 25th of every month.", "Payslips are available in the HR portal two days before {pay} day.",
                                  "Report any error in your {pay} to payroll within one month."],
         ["When is {pay} day?", "On what date do I get my {pay}?", "where can I find my payslip"], "25th"),
        ("Training budget", ["Each {employee} has a training budget of 1,500 euros per year.", "Courses must be approved by your {manager} before booking.",
                             "Certification exams are also covered by the training budget."],
         ["How much is my training budget?", "What can I spend on courses each year?", "does the training budget cover exams"], "1,500 euros"),
        ("Notice period", ["The notice period is one month in the first year of employment.", "After one year the notice period becomes three months.",
                           "Notice must be given in writing to your {manager} and HR."],
         ["What is my notice period if I resign?", "How much notice must I give when leaving?", "how do I hand in my resignation"], "three months"),
        ("Pension scheme", ["The company adds 6 percent of your {pay} to the pension plan.", "Every {employee} is enrolled automatically after 3 months.",
                            "You can add voluntary contributions through payroll."],
         ["How much does the company pay into my pension?", "What is the employer pension contribution?", "when am I enrolled in the pension"], "6 percent"),
        ("Bereavement leave", ["Up to 5 days of paid bereavement {leave} are granted for a close family member.", "Tell your {manager} as soon as you can.",
                               "Additional unpaid {leave} can be agreed if needed."],
         ["How much {leave} do I get when a relative dies?", "What is the bereavement {leave} policy?", "time off after a death in the family"], "5 days"),
    ],
    "it": [
        ("Resetting your password", ["To reset your {password}, open the self-service portal and choose Forgot.", "A new {password} must have 14 characters.",
                                     "Your {password} expires every 90 days."],
         ["How do I reset my {password}?", "I forgot my {password}, what do I do?", "how long must a new {password} be"], "14 characters"),
        ("Connecting to the VPN", ["{install} the {vpn} client from the software centre.", "Sign in with your company account and approve the phone prompt.",
                                   "The {vpn} is required to reach internal sites from home."],
         ["How do I connect to the {vpn} from home?", "Where do I get the {vpn} client?", "cannot reach internal sites when working from home"], "software centre"),
        ("Printer problems", ["If the {printer} shows an error, switch it off for 30 seconds.", "Print jobs are released by tapping your badge on the {printer}.",
                              "Toner is replaced by the office team, not by IT."],
         ["My {printer} is not working, what should I try?", "How do I release my print job?", "who replaces toner in the {printer}"], "30 seconds"),
        ("Email storage limits", ["Each {email} has a limit of 50 GB.", "When the {email} is 90 percent full you get a warning.",
                                  "Archive old messages to free space in your {email}."],
         ["How big is my {email} quota?", "My {email} is full, what can I do?", "what is the storage limit of the {email}"], "50 GB"),
        ("Requesting new software", ["Open a {ticket} in the service portal to request software.", "Licensed software needs approval from your manager.",
                                     "Free tools from the software centre can be {install}ed directly."],
         ["How do I get new software on my {laptop}?", "Can I {install} a program myself?", "who approves paid software"], "service portal"),
        ("Laptop replacement", ["A {laptop} is replaced every 4 years.", "Damaged {laptop} repairs are booked through a {ticket}.",
                                "Return the old {laptop} within 2 weeks of receiving the new one."],
         ["How often is my {laptop} replaced?", "When do I get a new {laptop}?", "my {laptop} is broken, how do I get it fixed"], "4 years"),
        ("Wi-Fi access", ["Staff connect to the corporate {wifi} with their company account.", "Guests use the guest {wifi}; the code changes every week.",
                          "Personal phones must use the guest {wifi}."],
         ["How do I connect to the {wifi} in the office?", "What {wifi} should guests use?", "can my personal phone use the corporate {wifi}"], "guest"),
        ("Multi-factor authentication", ["Multi-factor authentication uses the authenticator app on your phone.", "If you change phone, open a {ticket} to move MFA.",
                                         "Backup codes can be printed from the security page."],
         ["How do I set up multi-factor authentication?", "I got a new phone, how do I move my MFA?", "where are my backup codes"], "authenticator app"),
        ("Reporting phishing", ["Forward suspicious {email} messages to the security team with the Report button.", "Never click links in unexpected {email} messages.",
                                "Security replies within one business day."],
         ["What do I do with a suspicious {email}?", "How do I report phishing?", "I clicked a strange link in an {email}"], "Report button"),
        ("Shared drives", ["Team folders live on the shared drive and are backed up every night.", "Access to a shared drive is requested through a {ticket}.",
                           "Files deleted in the last 30 days can be restored."],
         ["How do I get access to a shared drive?", "Can deleted files be restored?", "are shared folders backed up"], "30 days"),
        ("Video conferencing", ["Meetings use the company video platform, which supports 300 participants.", "Recordings are kept for 60 days.",
                                "Use a headset to reduce echo in meetings."],
         ["How many people can join a video meeting?", "How long are meeting recordings kept?", "how do I stop echo in calls"], "300 participants"),
        ("Mobile phones", ["Company phones are managed with the device management app.", "Lost phones must be reported within 24 hours through a {ticket}.",
                           "Personal phones can access {email} after installing the management app."],
         ["What do I do if I lose my company phone?", "Can I read work {email} on my own phone?", "how quickly must I report a lost phone"], "24 hours"),
    ],
}


_DISTRACTORS = {   # pages that share vocabulary with the real answers but answer other questions
    "hr": [("Office locations", "The company has offices in Dublin, Lisbon and Warsaw. Each office has a reception desk open from 8am."),
           ("Parking", "Staff parking is free at all offices. Electric car chargers are available on the first floor."),
           ("Code of conduct", "Every {employee} must follow the code of conduct. Report concerns to your {manager} or the ethics line."),
           ("Holiday party", "The annual holiday party takes place in December. Partners are welcome to join."),
           ("Company cars", "Company cars are available for sales staff. Fuel is paid with a company card."),
           ("Volunteering days", "Every {employee} may spend 2 days per year volunteering for a charity."),
           ("Performance reviews", "Reviews happen twice a year with your {manager}. Goals are set in the HR portal."),
           ("Promotions", "Promotions are decided once a year after the performance review cycle."),
           ("Diversity and inclusion", "Employee networks meet monthly. All {employee} groups are open to everyone."),
           ("Health insurance", "Private health insurance covers the {employee} and family members. Dental care is included."),
           ("Gym membership", "A gym membership discount of 30 percent is available at partner gyms."),
           ("Relocation support", "Relocation costs are covered for international moves, including temporary housing."),
           ("Working hours", "The standard week is 40 hours, Monday to Friday. Flexible start times are possible."),
           ("Mentoring programme", "New starters are paired with a mentor for their first six months."),
           ("Onboarding", "The first week includes HR induction, IT setup and a team lunch."),
           ("Uniform and dress code", "Business casual is the norm. Client meetings may require formal dress."),
           ("Business travel booking", "Book flights and hotels through the travel portal. Economy class is standard below 6 hours."),
           ("Whistleblowing", "Concerns can be reported anonymously through the ethics line, available 24 hours a day."),
           ("Jury duty", "Time spent on jury duty is paid. Inform your {manager} as soon as you receive the summons."),
           ("Retirement", "Employees may retire from age 60. Contact HR one year before your planned date.")],
    "it": [("Software centre", "The software centre lists approved applications. Most tools {install} in a few minutes."),
           ("Office printers map", "A map of every {printer} is on the intranet. Colour printing is available on floor 3."),
           ("Data classification", "Documents are classified as public, internal or confidential. Label every file."),
           ("Clean desk policy", "Lock your screen when you leave your desk. Store papers in locked drawers."),
           ("Accessibility tools", "Screen readers and magnifiers can be requested through a {ticket}."),
           ("Monitors and docking stations", "Each desk has a docking station and two monitors."),
           ("Collaboration tools", "Use the team chat for quick questions and {email} for formal messages."),
           ("IT service hours", "The help desk is open from 7am to 7pm on working days."),
           ("Security training", "Every user completes security awareness training once a year."),
           ("Backup policy", "Laptops are not backed up automatically. Store work files on the shared drive."),
           ("Encryption", "Every {laptop} is encrypted. Never disable encryption."),
           ("Personal devices policy", "Personal devices may not connect to the internal network."),
           ("Software licences", "Licences are assigned per user and reviewed every quarter."),
           ("Browser support", "Use the latest version of a supported browser for internal sites."),
           ("Keyboard and mouse", "Replacement keyboards and mice are available from the IT kiosk."),
           ("Conference room screens", "Every meeting room has a wireless screen-sharing device."),
           ("Developer tools", "Developers request admin rights through a {ticket} with manager approval."),
           ("Cloud storage", "Personal cloud storage services are blocked. Use the company file storage instead."),
           ("Out of office replies", "Set an automatic reply in your {email} when you are away."),
           ("Hardware disposal", "Old devices are wiped and recycled by IT. Never throw them away.")],
}


def make_text_corpus(domain: str = "hr", n_sentences: int = 6000, seed: int = 1) -> list[str]:
    """Unlabelled 'pre-training' text: years of intranet pages, chats and tickets written in the same
    domain language. Every template is filled many times with random synonyms, so synonyms ('vacation',
    'time off', 'leave') keep appearing in the SAME contexts. That shared context is exactly what
    embedding methods learn from (the distributional hypothesis). Contains no gold questions."""
    rng = np.random.default_rng(seed)
    syn = _SYN[domain]
    templates = [f for _, facts, qs, _ in _KB[domain] for f in facts + qs] + [t for _, t in _DISTRACTORS[domain]]
    return [_fill(str(rng.choice(templates)), syn, rng).lower() for _ in range(n_sentences)]


def _fill(template: str, syn: dict, rng) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(rng.choice(syn[m.group(1)])), template)


def make_knowledge_base(domain: str = "hr", seed: int = RANDOM_STATE, variants: int = 3, questions_per_article: int = 8):
    """A company knowledge base (`hr`: an HR policy handbook; `it`: an IT help centre).
    Returns (articles, questions):
        articles  : one row per article (several near-duplicate VERSIONS per topic are NOT used; instead each
                    article is written once with randomly chosen synonyms, plus filler sentences)
        questions : user questions, paraphrased with OTHER synonyms, each with `gold_article` and `answer`
                    (a fact that must appear in a correct answer). Built for retrieval and RAG evaluation."""
    rng = np.random.default_rng(seed)
    syn = _SYN[domain]
    filler = ["This policy applies to all locations.", "Contact the help desk if anything is unclear.",
              "This page was last reviewed this year.", "Local rules may add further details.",
              "See the related pages for more information."]
    arts, qs = [], []
    for a_id, (title, facts, questions, answer) in enumerate(_KB[domain]):
        sents = [_fill(f, syn, rng) for f in facts] + list(rng.choice(filler, 2, replace=False))
        body = " ".join(s[0].upper() + s[1:] for s in sents)
        arts.append({"article_id": f"{domain.upper()}-{a_id:02d}", "title": title, "text": body, "answer_fact": answer})
        for q in range(questions_per_article):
            text = _fill(questions[q % len(questions)], syn, rng)
            if rng.random() < 0.3:
                text = text.lower().rstrip("?")
            qs.append({"question": text, "gold_article": f"{domain.upper()}-{a_id:02d}", "answer": answer})
    for d_id, (title, text) in enumerate(_DISTRACTORS[domain]):
        body = _fill(text, syn, rng) + " " + " ".join(rng.choice(filler, 2, replace=False))
        arts.append({"article_id": f"{domain.upper()}-X{d_id:02d}", "title": title, "text": body, "answer_fact": None})
    return pd.DataFrame(arts), pd.DataFrame(qs).sample(frac=1, random_state=seed).reset_index(drop=True)
