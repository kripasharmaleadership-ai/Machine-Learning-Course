# Contributing

Thanks for helping make this course better! Fixes for bugs, clearer explanations, new exercises and better "plot twists" are all welcome.

## Reporting a problem

Open an [issue](https://github.com/kripasharmaleadership-ai/Machine-Learning-Course/issues/new/choose) with:
- the module and lab (e.g. `Module4 / 11_hands_on_lab_customer_churn.py`),
- what you ran (script or cell mode), your OS, and `python --version`,
- the full error, or the number you expected vs the number you got.

## Lab conventions

Every lab follows the same shape. Please keep it:

1. **One file per lab**, `NN_short_topic.py`, with `# %%` cell markers so it runs both as a script and cell by cell.
2. **A docstring** with the idea, the setup and the three levels.
3. **EASY → MEDIUM → HARD.** EASY shows the core idea with every number visible, MEDIUM the realistic comparison, HARD the production problem, ending in a **plot twist**.
4. **Talking points (`note(...)` → `>>` lines) are computed from the run.** Write them as f-strings that read the numbers from the results. Never type a number that the code produces. If you change a lab, read every note again and rewrite any sentence the new numbers contradict.
5. **Honest results.** If the fancy method loses, the lab says so.
6. **Seeded randomness and offline runs.** Use the module's `RANDOM_STATE`; real datasets are downloaded once and cached in `data/`.
7. **Shared code goes in the module's toolkit** (`*_toolkit.py`, `preprocessing.py`), checked against a standard library where one exists.

## Testing a change

```bash
cd ModuleN_...
python run_all.py            # every lab in the module, pass/fail + runtime
python run_all.py 07 11      # only the labs whose file name starts with 07 or 11
```

Install with `pip install -r requirements.txt -c constraints.txt` so your numbers match the READMEs. If you change a module's toolkit or `ml_lab_utils.py`, run the **whole** module before and after your change and compare the outputs: a shared helper can move numbers in labs you didn't touch.

## Pull requests

- Keep a PR to one topic (one lab, one fix, one new exercise).
- Update the module's `README.md` if a key number in its episode plan changes.
- Describe what you changed and paste the `run_all.py` summary.
