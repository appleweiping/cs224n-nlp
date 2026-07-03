# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import os
import utils

# Locate birth_dev.tsv whether run from the repo root or from src/.
eval_corpus_path = "birth_dev.tsv"
if not os.path.exists(eval_corpus_path):
    eval_corpus_path = os.path.join(os.path.dirname(__file__), "..", "birth_dev.tsv")
# Count the dev examples so we can predict "London" for each one.
with open(eval_corpus_path, encoding='utf-8') as f:
    num_examples = len([line for line in f if line.strip()])

predictions = ["London"] * num_examples
total, correct = utils.evaluate_places(eval_corpus_path, predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
else:
    print('No gold birth places provided; returning (0,0)')
