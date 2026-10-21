"""
LAB 04 - Structured Outputs
===========================

WHY STRUCTURE
    Labs 02-03 asked for one label. A support system needs more, in a form CODE can use:
        {"intents": ["return_request", "policy_question"], "order_id": "LM-10234", "email": null,
         "condition": "defective", "urgency": "high"}
    The next step (look up the order, check the return) is a program, and a program needs valid JSON with the
    right keys and the allowed values: not "Sure! Here's the JSON you asked for: ...".

THREE WAYS TO GET IT
    1. ASK in the prompt ("reply in JSON with keys ...") and hope         -> parse errors
    2. VALIDATE + REPAIR: parse defensively, check against a JSON Schema, send the error back and ask again
    3. CONSTRAIN: give the API the schema (structured outputs); decoding can only produce schema-valid JSON

    And then the part no schema can do: is the VALID answer also CORRECT?

LEVELS
    EASY   : JSON by asking: how often it parses, how it breaks, large vs small.
    MEDIUM : A defensive pipeline: lenient parsing, schema validation, a repair loop. What each stage buys.
    HARD   : Schema-constrained output: always valid. Field by field, is it right? Valid is not correct.
"""
# %%
import json
import re

import jsonschema
import numpy as np
import pandas as pd

from genai_toolkit import get_llm
from ml_lab_utils import HAS_MPL, INTENTS, banner, make_orders, make_tickets, note, plt, save_fig, show, step

tickets = make_tickets()
test = tickets[tickets.split == "test"].reset_index(drop=True)
_, orders = make_orders()
ORDER_IDS = set(orders.order_id)
LABELS = list(INTENTS)
large, small = get_llm("large"), get_llm("small")

SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {"type": "array", "items": {"type": "string", "enum": LABELS}, "minItems": 1,
                    "description": "every request in the ticket, in order"},
        "order_id": {"type": ["string", "null"], "description": "the order number, e.g. LM-10234, or null"},
        "email": {"type": ["string", "null"], "description": "the customer's email address, or null"},
        "condition": {"type": ["string", "null"], "enum": ["unopened", "opened", "defective", "used", None],
                      "description": "the item's condition if they want to return it, else null"},
        "urgency": {"type": "string", "enum": ["normal", "high"]},
    },
    "required": ["intents", "order_id", "email", "condition", "urgency"],
    "additionalProperties": False,
}
PROMPT = ('Extract these fields from the support ticket and reply in JSON with the keys "intents", "order_id", '
          '"email", "condition", "urgency".\n'
          '"intents": a list, each one of [' + ", ".join(LABELS) + ']\n'
          '"order_id": the order number (like LM-10234) or null\n"email": the email address or null\n'
          '"condition" one of [unopened, opened, defective, used] or null\n"urgency" one of [normal, high]\n\n'
          '<ticket>{ticket}</ticket>')


def status_of(text: str) -> tuple[str, dict | None]:
    """'not JSON' / 'JSON, fails the schema' / 'valid', plus the parsed object."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "not JSON", None
    try:
        jsonschema.validate(obj, SCHEMA)
        return "valid", obj
    except jsonschema.ValidationError:
        return "JSON, fails the schema", obj


def why_broken(text: str, status: str, obj) -> str:
    if status == "valid":
        return "-"
    if text.lstrip().startswith("```"):
        return "wrapped in a ``` code fence"
    if not text.lstrip().startswith("{"):
        return "chatty text around the JSON"
    if "'" in text and '"' not in text:
        return "single quotes"
    if text.rstrip().endswith(",}"):
        return "trailing comma"
    if obj is not None and set(obj) != set(SCHEMA["properties"]):
        return "a key is missing"
    if obj is not None and any(isinstance(v, str) and v.isupper() for v in obj.values()):
        return "wrong case ('HIGH')"
    return "other"


def ask(llm, ticket, schema=None):
    return llm.create(messages=[{"role": "user", "content": PROMPT.format(ticket=ticket)}], output_schema=schema, max_tokens=300)


# %% =========================================================================
# LEVEL 1 (EASY): JSON by asking
# ============================================================================
banner("JSON by asking: how often does it parse?", "EASY")
outs = {m.model: [ask(m, t).text for t in test.text] for m in [large, small]}
show("A good answer", next(t for t in outs["sim-large"] if status_of(t)[0] == "valid"))
rows, kinds = {}, {}
for model, texts in outs.items():
    st = [status_of(t) for t in texts]
    s = pd.Series([x[0] for x in st])
    rows[model] = s.value_counts(normalize=True).reindex(["valid", "JSON, fails the schema", "not JSON"]).fillna(0)
    kinds[model] = pd.Series([why_broken(t, a, o) for t, (a, o) in zip(texts, st)]).value_counts().drop("-", errors="ignore")
print(pd.DataFrame(rows).T.round(3).to_string())
print(pd.DataFrame(kinds).fillna(0).astype(int).to_string())
broken = [t for t in outs["sim-small"] if status_of(t)[0] != "valid"]
for t in broken[:3]:
    show("  a broken answer (small model)", t.replace("\n", " | ")[:110])
l_ok, s_ok = rows["sim-large"]["valid"], rows["sim-small"]["valid"]
note(f"""Asked nicely, the large model returns usable JSON {l_ok:.0%} of the time and the small one
{s_ok:.0%}. The failures are the classics: a ```json fence, a friendly sentence
around the object, single quotes, a trailing comma (json.loads rejects all of
these), and the sneaky ones that PARSE but are wrong: a missing key, 'HIGH'
instead of 'high'. Only a schema check catches those. At 10,000 tickets a day,
{1 - l_ok:.0%} is {10_000 * (1 - l_ok):,.0f} broken hand-offs a day.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Parse defensively, validate, repair
# ============================================================================
banner("A defensive pipeline: lenient parse -> schema -> repair loop", "MEDIUM")


def lenient(text: str) -> str:
    """Strip code fences and surrounding chatter: keep the outermost {...}."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    a, b = t.find("{"), t.rfind("}")
    return t[a:b + 1] if a >= 0 and b > a else t


def robust_extract(llm, ticket, max_repairs=2):
    """Ask -> lenient parse -> validate; on failure, send the error back and ask again."""
    messages = [{"role": "user", "content": PROMPT.format(ticket=ticket)}]
    calls, cost, stage = 0, 0.0, None
    for attempt in range(max_repairs + 1):
        r = llm.create(messages=messages, max_tokens=300)
        calls, cost = calls + 1, cost + r.cost_usd
        text = lenient(r.text)
        try:
            obj = json.loads(text)
            jsonschema.validate(obj, SCHEMA)
            stage = "valid as is" if attempt == 0 and text == r.text.strip() else (
                "lenient parse" if attempt == 0 else f"repair #{attempt}")
            return obj, stage, calls, cost
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            err = e.msg if isinstance(e, json.JSONDecodeError) else e.message
            messages += [r.as_message(), {"role": "user", "content": f"Your reply was not valid JSON for the schema "
                                          f"(error: {err}). Reply with only the corrected JSON object, nothing else."}]
    return None, "gave up", calls, cost


rows = {}
for llm in [large, small]:
    res = [robust_extract(llm, t) for t in test.text]
    stages = pd.Series([r[1] for r in res]).value_counts(normalize=True)
    rows[llm.model] = {"valid as is": stages.get("valid as is", 0.0),
                       **{f"fixed by: {k}": stages.get(k, 0.0) for k in ["lenient parse", "repair #1", "repair #2"]},
                       "still invalid": stages.get("gave up", 0.0), "calls per ticket": np.mean([r[2] for r in res]),
                       "$ per 1,000 tickets": np.sum([r[3] for r in res]) / len(res) * 1000}
pipe = pd.DataFrame(rows).T
print(pipe.round(3).to_string())
note(f"""Lenient parsing (strip the fence, keep the outermost braces) rescues the
cosmetic failures for free. Schema validation catches answers that parse but
are wrong. The repair loop sends the exact error back ('error: ... is not one
of ['normal', 'high']'), and the model usually fixes it on the next try. The
small model ends at {1 - pipe.loc['sim-small', 'still invalid']:.1%} valid, at the price of {pipe.loc['sim-small', 'calls per ticket']:.2f} calls per ticket instead of 1:
retries cost money and, worse, latency. This is how everyone did it before
APIs could enforce a schema, and it's still the fallback when they can't.""")

# %% =========================================================================
# LEVEL 3 (HARD): Constrained output, and the difference between valid and correct
# ============================================================================
banner("Schema-constrained output: always valid. Is it right?", "HARD")
const = {}
for llm in [large, small]:
    const[llm.model] = [json.loads(ask(llm, t, schema=SCHEMA).text) for t in test.text]
    show(f"{llm.model}: schema-valid answers, 1 call each", f"{np.mean([status_of(json.dumps(o))[0] == 'valid' for o in const[llm.model]]):.1%}")
note("""With output_schema the API constrains decoding to the schema (on Claude:
output_config.format = {type: json_schema, schema: ...}). Every answer is
valid JSON with every key and only allowed values, in one call, no repair
loop. That solves FORMAT. Now the real question.""")

step("3.1  Field by field: valid, but correct?")


def field_scores(objs):
    gold_cond = test.condition.where(test.condition.notna(), None)
    return {"intents (exact list)": np.mean([o["intents"] == list(g) for o, g in zip(objs, test.intents)]),
            "first intent": np.mean([o["intents"][:1] == [g] for o, g in zip(objs, test.intent)]),
            "order_id": np.mean([o["order_id"] == g for o, g in zip(objs, test.order_id.where(test.order_id.notna(), None))]),
            "email": np.mean([o["email"] == g for o, g in zip(objs, test.email.where(test.email.notna(), None))]),
            "condition": np.mean([o["condition"] == g for o, g in zip(objs, gold_cond)]),
            "urgency": np.mean([o["urgency"] == g for o, g in zip(objs, test.urgency)])}


fields = pd.DataFrame({m: field_scores(o) for m, o in const.items()})
print(fields.round(3).to_string())
multi = test.intents.map(len) > 1
show("Multi-part tickets: all requests found (large)", f"{np.mean([o['intents'] == list(g) for o, g, m in zip(const['sim-large'], test.intents, multi) if m]):.0%} of {multi.sum()}")
urg_miss = test[[o["urgency"] != g for o, g in zip(const["sim-large"], test.urgency)]]
under, over = urg_miss[urg_miss.urgency == "high"], urg_miss[urg_miss.urgency == "normal"]
show("Urgency errors (large): high marked normal / normal marked high", f"{len(under)} / {len(over)}")
show("  high marked normal, e.g.", under.text.iloc[0] if len(under) else "-")
show("  normal marked high, e.g.", over[over.attack.isna()].text.iloc[0] if over.attack.isna().any() else "-")
note(f"""100% valid JSON, and most fields are still wrong some of the time. The list
field fixes Lab 02's leftover (a ticket with two requests now gets two
intents). Urgency errors go both ways: {len(under)} furious tickets came back 'normal'
(no urgent keyword), and {len(over)} came back 'high' where our answer key says
'normal'. Read that second example: the KEY is arguably wrong. A field like
urgency needs a written definition, in the schema description and for
whoever labels the test set. And order numbers... let's look closer, because
a wrong order number is what turns a support bot into a liability.""")

step("3.2  The order numbers that were wrong, by what would catch them")
pattern = re.compile(r"^[A-Z]{2}-\d{5}$")


def diagnose(o, gold, text):
    got = o["order_id"]
    if got == gold:
        return "correct"
    if got is None:
        return "missed (customer's typo made it unreadable)"
    if not pattern.match(got):
        return "wrong FORMAT ('LM12575'): a regex check catches it"
    if got not in ORDER_IDS:
        return "no such order: a database lookup catches it"
    if re.search(r"override|ignore all previous", text, re.I):
        return "REAL order, copied from injected text: passes every check"
    return "REAL order, but not the customer's: passes every check"


rows = {}
for model, objs in const.items():
    rows[model] = pd.Series([diagnose(o, g, t) for o, g, t in zip(objs, test.order_id.where(test.order_id.notna(), None), test.text)]).value_counts()
order = ["correct", "missed (customer's typo made it unreadable)", "wrong FORMAT ('LM12575'): a regex check catches it",
         "no such order: a database lookup catches it", "REAL order, copied from injected text: passes every check",
         "REAL order, but not the customer's: passes every check"]
diag = pd.DataFrame(rows).reindex(order).fillna(0).astype(int)
print(diag.to_string())
inj = test[test.text.str.contains("SYSTEM OVERRIDE")].iloc[0]
show("Example", inj.text[:150] + "...")
show("  extracted order_id (gold: none in the real request)", const["sim-large"][inj.name]["order_id"])
note(f"""Layer the checks. A pattern in your validation catches the small model's
'LM12575' slip. A lookup in the orders database catches a number that doesn't
exist (ask the customer again). Plot twist: the most dangerous errors pass
every check. A ticket carrying a fake 'SYSTEM OVERRIDE ... refund order
{const['sim-large'][inj.name]['order_id']}' gets that order number extracted, perfectly formatted and real:
someone else's order. Schemas guarantee shape, never truth. The defence is
outside the model: act only on orders that belong to the signed-in customer,
and keep write actions behind rules and approvals (Labs 12 and 14).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    pd.DataFrame(rows_ := {m: pd.Series([status_of(t)[0] for t in outs[m]]).value_counts(normalize=True) for m in outs}).T.reindex(
        columns=["valid", "JSON, fails the schema", "not JSON"]).fillna(0).plot.bar(stacked=True, ax=axes[0], rot=0,
                                                                                  color=["C2", "C1", "C3"])
    axes[0].set(title="JSON by asking", ylabel="share of answers")
    axes[0].legend(fontsize=8, loc="lower right")
    stage_cols = [c for c in pipe.columns if c.startswith("fixed by") or c in ("valid as is", "still invalid")]
    pipe[stage_cols].plot.bar(stacked=True, ax=axes[1], rot=0)
    axes[1].set(title="Defensive pipeline: what each stage rescues", ylabel="share of tickets")
    axes[1].legend(fontsize=7, loc="lower right")
    fields.plot.barh(ax=axes[2])
    axes[2].set(xlim=(0.5, 1.0), title="Schema-valid 100%: field accuracy")
    axes[2].legend(fontsize=8)
    save_fig(fig, "04_structured_outputs")

banner("KEY TAKEAWAYS")
print("""  1. Code needs structure: define the fields as a JSON Schema (types, enums, required keys).
  2. 'Reply in JSON' works most of the time; at scale, most of the time is not enough.
  3. Defensive parsing + schema validation + a repair loop gets close to 100%, for extra calls.
  4. Schema-constrained output makes FORMAT a solved problem: valid in one call.
  5. Valid is not correct: validate the values in layers (pattern, database, ownership),
     and never let an extracted ID authorise an action on its own.""")
