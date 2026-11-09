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

1. How come we need to turn our text into numbers to use with a language model?
2. If we attempt to encode a character that wasn't in the training data using our character-level tokenizer, what will happen?
3. Why do we need a validation set?
4. How many unique characters are in our vocabulary?

## Assignment

**Assignment 4: Same word, new numbers**

**About 15 minutes. You change one word and write no new code.** Work in this chapter's Colab notebook or on your own laptop.

1. **Run it.** Run `src/examples/ch04_tokenizer_demo.py` unaltered. Copy the three rows it outputs, which will be phrases with their lists of numbers.
2. **Change one word.** Locate the line `"URGENT!",`. It's one of the three phrases we'll convert to numbers with this script. Replace the word `URGENT` with `urgent`, making it all lowercase letters, but leave the rest of the line intact, so it says `"urgent!",`. Run the script again.
3. **Compare.** Which numbers changed in the second row? Which did not, and what letter does it represent? Two numbers in your new row are also in the `invoice` row. What two letters do they represent?
4. **Explain it.** In two or three sentences someone who manages you would understand, explain why the model sees `URGENT!` and `urgent!` are two different pieces of text when people reading them understand it's the same word.

**Hand in:** both sets of three rows, your answers to step 3, and your sentences.
