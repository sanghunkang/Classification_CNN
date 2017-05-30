#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author        : ...
# Contact       : ...
# Started on    : 20170310(yyyymmdd)
# Last modified : 20170417(yyyymmdd)
# Project       : ...
#############################################################################
import sys, os

if __name__ == "__main__":
	isFinish = False
	dirpath = sys.path[0]

	while isFinish == False:
		cmd_input = input("Input command: ")

		if cmd_input == "-e":
			import extract_model
		elif cmd_input in ["run gpu"]:
			import run_gpu
		elif cmd_input == "flush-logs":
			try:
				os.chdir(dirpath + ".\\logs\\train\\")
				for f in os.listdir("."):
					os.remove(f)

				os.chdir(dirpath + ".\\logs\\test\\")
				for f in os.listdir("."):
					os.remove(f)
			except BaseException as e:
				print(e)
		elif cmd_input == "flush-model":
			try:
				os.chdir(dirpath + ".\\model\\")
				for f in os.listdir("."):
					os.remove(f)
			except BaseException as e:
				print(e)
		elif cmd_input in ["q","Q"]:
			isFinish = True
		else:
			print("Invalid Command")