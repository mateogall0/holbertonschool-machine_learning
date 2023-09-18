#!/usr/bin/env python3
"""
Answer Questions
"""


import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer
exit_cases = ["exit", "quit", "goodbye", "bye"]
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
  while 1:
      user_in = input("Q: ").strip().lower()
      print(end='A: ')
      if user_in in exit_cases:
          print("Goodbye")
          break
      answer = question_answer(user_in, reference)
      if answer is None:
          answer = 'Sorry, I do not understand your question.'
      print(answer)
