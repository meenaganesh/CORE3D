import argparse
import json

from geon_infer.semseg.run import run_tasks as semseg_run_tasks

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('file_path', nargs='?')
  parser.add_argument('tasks', nargs='*')
  return parser.parse_args()

def run_tasks():
  args = parse_args()
  with open(args.file_path) as options_file:
    options_dict = json.load(options_file)
    semseg_run_tasks(options_dict, args.tasks)
           
if __name__ == '__main__':
    run_tasks()
