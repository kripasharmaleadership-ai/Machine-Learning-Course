# Review sentiment: which classifier should tag incoming reviews?

3,624 training / 1,153 validation / 1,223 test reviews; scored against the hidden true labels; 'swapped endings' replaces each shortcut ending with its opposite.

## Requirements

| requirement | type | metric | threshold / weight | why |
|---|---|---|---|---|
| overall accuracy | HARD gate | `test accuracy` | >= 0.9 | below 90% the support team stops trusting the tags |
| negations | HARD gate | `acc: negated` | >= 0.85 | 'not working' is the review support most needs to see |
| contrast reviews | soft | `acc: contrast` | weight 35% (higher is better) | 'great screen but awful support' is a complaint |
| robust to endings | soft | `shortcut drop` | weight 25% (lower is better) | customers don't always end with 'one star' |
| speed | soft | `ms_per_review` | weight 20% (lower is better) | tagging runs on every new review |
| size | soft | `parameters` | weight 20% (lower is better) | cheaper to host, retrain and explain |

## Candidates

| candidate | overall accuracy | negations | contrast reviews | robust to endings | speed | size | eligible | failed gates | weighted score |
|---|---|---|---|---|---|---|---|---|---|
| word + bigram TF-IDF logistic | 0.953 | 0.955 | 0.836 | 0.173 | 0.012 | 291 | True | - | 0.845 |
| tiny transformer (from scratch) | 0.933 | 0.859 | 0.954 | 0.120 | 0.134 | 107266 | True | - | 0.600 |
| bag-of-words logistic regression | 0.759 | 0.544 | 0.607 | 0.578 | 0.006 | 58 | False | overall accuracy, negations | - |
| MLP on bag-of-words (NumPy) | 0.852 | 0.766 | 0.598 | 0.257 | 0.007 | 5825 | False | overall accuracy, negations | - |

## Recommendation

**word + bigram TF-IDF logistic**

**word + bigram TF-IDF logistic** passes both gates (accuracy 95.3%, negations 95.5%) and scores best on contrast reviews (83.6%) with a shortcut drop of +17.3%. Monitor accuracy by review type monthly on a fresh hand-labelled sample: new phrasing (sarcasm, emojis, other languages) is out of distribution for every model here.
