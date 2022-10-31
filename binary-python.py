#!/usr/bin/env python3

import numpy as np


def sign(a):
	return -(a >> 32)
	
	
def hd(a, b):
	count = 0
	diff = a ^ b
	while diff:
		count += 1
		diff &= diff - 1
	return count
	
	
def swap(a, b):
	a ^= b
	b ^= a
	a ^= b
	return a, b
	
	
def hw(a):
	count = 0
	while a:
		count += 1
		a &= a - 1
	return count


def hw_slower(a):
	count = 0
	while a:
		count += a & 1
		a >>= 1
	return count
	

def myadd(a, b):
	while a:
		c = b & a
		b ^= a
		c <<= 1
		a = c
	return b


def myabs(a):
	high_bit_mask = a >> 31
	return (a ^ high_bit_mask) - high_bit_mask


def iseven(a):
	return a & 1 == 0


def to_binary(a, len):
	b = ""
	while len:
		b = ("1" if (a & 1) else "0") + b
		a >>= 1
		len -= 1
	return b

def my_hash(s, l):
	hash = 0
	for i in range(len(s)):
		hash = hash * 31 + s[i]
	return hash % l
		



print(f"sign(2): {sign(2)}")
print(f"sign(-2): {sign(-2)}")
print(f"sign(3): {sign(3)}")
print(f"sign(-3): {sign(-3)}")
print(f"hd(1, 2): {hd(1, 2)}")
print(f"hd(1, 3): {hd(1, 3)}")
print(f"swap(1, 2): {swap(1, 2)}")
print(f"hw(1): {hw(1)}")
print(f"hw(3): {hw(3)}")
print(f"hw(123123): {hw(123123)}")
print(f"myadd(1, 2): {myadd(1, 2)}")
print(f"myabs(5): {myabs(5)}")
print(f"myabs(-5): {myabs(-5)}")
print(f"iseven(1): {iseven(1)}")
print(f"iseven(2): {iseven(2)}")
print(f"iseven(3): {iseven(3)}")