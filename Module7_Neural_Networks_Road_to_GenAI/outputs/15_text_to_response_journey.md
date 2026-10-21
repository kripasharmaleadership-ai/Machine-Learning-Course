# From raw text to an LLM-generated response

Traced through the instruction-tuned TinyGPT of Module 7 (852,736 parameters, 4 blocks, vocabulary of 400 tokens). Question: **What is the capital of Piri?** (true answer: Eomezor).

![journey](15_text_to_response_journey.png)

## The stages

| stage | what comes out here | why |
|---|---|---|
| 1 raw text | `What is the capital of Piri?` | what the user typed |
| 2 chat template | `Q: What is the capital of Piri?\nA:` | the format instruction tuning taught |
| 3 pre-tokenize | `12 chunks` | words keep their leading space; digits split |
| 4 BPE -> ids | `15 token ids` | subwords from a learned 400-token vocabulary |
| 5 embeddings | `15 x 128 matrix` | token vector + position vector |
| 6 transformer x4 | `15 x 128 matrix` | attention moves information; MLP transforms it |
| 7 logits | `400 scores` | last position vs every token embedding |
| 8 softmax + decode | `' E' p = 0.77` | greedy here; temperature / top-p optional |
| 9 loop | `6 passes` | append the token, run again, stop at newline |
| 10 detokenize | `' Eomezor.'` | ids -> text |

## Inside the transformer (stage 6)

| block | output shape | size of last position's vector | last position attends most to |
|---|---|---|---|
| 1 | (15, 128) | 23.935 | ' Piri' 0.33, 'A' 0.18, 'at' 0.13 |
| 2 | (15, 128) | 26.779 | '\n' 0.16, ' Piri' 0.14, '?' 0.13 |
| 3 | (15, 128) | 28.984 | ' the' 0.15, '?' 0.13, ' is' 0.13 |
| 4 | (15, 128) | 32.541 | ':' 0.19, ' of' 0.09, 'A' 0.09 |

## The next-token distribution (stages 7-8)

| token | logit | probability |
|---|---|---|
|  E | 12.350 | 0.774 |
| E | 10.740 | 0.155 |
|  its | 8.780 | 0.022 |
|  Seren | 8.270 | 0.013 |
|  Dor | 7.780 | 0.008 |
| are | 7.680 | 0.007 |

## The generation loop (stage 9)

| step | context length | chosen token | probability | runner-up | runner-up prob |
|---|---|---|---|---|---|
| 1 | 15 |  E | 0.774 | E | 0.155 |
| 2 | 16 | om | 1.000 | of | 0.000 |
| 3 | 17 | e | 1.000 | es | 0.000 |
| 4 | 18 | zor | 0.999 | cor | 0.001 |
| 5 | 19 | . | 1.000 |  is | 0.000 |
| 6 | 20 | 
 | 1.000 |  arrive | 0.000 |

**Response:** ' Eomezor.'

## Where it goes wrong: a question it cannot answer

The same pipeline for **Venmir**, a country with no facts in the training data, answers ' Dorcor.': a hallucination. 8 samples at T = 1:  Dorlencor.,  Dorbel.,  Dor at school.,  DorFezorka.,  Doros.,  Dorithcor.,  Dorithcor.,  Dorithcor..

| index | first-token probability | whole-answer probability | samples agreeing with the most common |
|---|---|---|---|
| known: Piri | 0.774 | 0.772 | 0.875 |
| unknown: Venmir | 0.822 | 0.295 | 0.375 |

The first token is not a reliable warning; the whole-answer probability and the agreement between samples are.

## Explanation

An LLM turns text into integers (tokenization), integers into vectors (embeddings), lets the vectors exchange information through stacked self-attention and MLP blocks, and turns the last position's vector into a probability for every possible next token. Decoding picks one token, which is appended to the input, and the whole computation repeats until a stop token. Knowledge comes from pretraining (next-token prediction on raw text); the question-answer behaviour comes from instruction tuning; the randomness and the risk of hallucination come from sampling likely, not true, text. Production systems wrap this loop with chat templates, retrieval (RAG), safety filters and evaluation.
