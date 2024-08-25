#!/usr/bin/python

import sys
import re
from submodules.rules import rules

def main():
	with open(sys.argv[1], "r") as fp:
		lines = fp.readlines()

		for idx, line in enumerate(lines):

			if line.strip() == "# ------------------------ >8 ------------------------":
				break

			if line[0] == "#":
				continue

			if not line_valid(idx, line):
				print(f"line# {idx} failed")
				show_rules()
				sys.exit(1)

	sys.exit(0)

def line_valid(idx, line):
	if idx == 0:
		#return re.match("^[A-Z].{,48}[0-9A-z \t]$", line)
		return re.match("^\[((?!\s*$).{0,15})\][ \t].*?[A-Z].{0,48}[0-9A-z \t]$", line)
	else:
		return len(line.strip()) <= 72

def show_rules():
	print(rules)

if __name__ == "__main__":
	main()