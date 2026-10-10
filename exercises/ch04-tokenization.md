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

## Assignment

**Assignment 4: Same word, new numbers**

**About 15 minutes. You change one word and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch04_tokenizer_demo.py` as it is. Copy the three rows it prints, each a phrase and its list of numbers.
2. **Change one word.** Find the line `"URGENT!",`. It is one of the three phrases the script turns into numbers. Change `URGENT` to `urgent`, in small letters, and keep everything else on the line, so it reads `"urgent!",`. Run the script again.
3. **Compare.** Which numbers in the second row changed? Which one stayed the same, and which character does it stand for? Two numbers in your new row also appear in the `invoice` row. Which two letters do they stand for?
4. **Explain it.** In two or three sentences a manager could follow, say why the model sees `URGENT!` and `urgent!` as different text when a person reads the same word.

**Hand in:** both sets of three rows, your answers to step 3, and your sentences.
