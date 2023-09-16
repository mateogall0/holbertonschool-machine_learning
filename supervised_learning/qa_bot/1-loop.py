#!/usr/bin/env python3
"""
Create the loop
"""


exit_cases = ["exit", "quit", "goodbye", "bye"]


while 1:
    user_in = input("Q: ").strip().lower()
    print(end='A: ')
    if user_in in exit_cases:
        print("Goodbye")
        break
    print()