# Chapter 4: Tokenization: Exercises

Read the chapter online: https://jackluu.io/book/section-1-foundations/ch04-tokenization/

## Try It

See how different inputs are converted into numbers. We wrote a short script that imports our tokenizer and encodes a few business phrases.

Open the terminal and run the example script. Notice how every letter, space, and punctuation mark is assigned a specific number from our vocabulary.

```console
$ python src/examples/ch04_tokenizer_demo.py
Demo: Encoding short business phrases
'invoice'       -> [47, 52, 60, 53, 47, 41, 43]
'URGENT!'       -> [33, 30, 19, 17, 26, 32, 2]
'Hello.'        -> [20, 43, 50, 50, 53, 8]

```

## Check Your Understanding

1. Why must we convert text to numbers before feeding it to a language model?
2. In our character-level tokenizer, what happens if we try to encode a character that wasn't in the training data?
3. What is the purpose of the validation set?
4. How many unique characters are in our vocabulary?
