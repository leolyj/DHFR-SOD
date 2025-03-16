from __future__ import print_function, division
import os
import argparse
import shutil


backup_files_list = ['bashscripts', 'configs', 'datasets', 'lib', 'misc', 'pyscripts', 'spml']

def main(dst_dir):
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

  for src_dir in backup_files_list:
    if not os.path.exists(src_dir):
      print('error: source dir %s is not exist!' %(src_dir))
      return -1

    base_name = os.path.basename(src_dir)
    dst_path = os.path.join(dst_dir, base_name)
    if os.path.exists(dst_path):
      shutil.rmtree(dst_path)
    shutil.copytree(src_dir, dst_path)

  print('backup files successed!')
  
  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dst_dir', required=True, type=str,
                      help='/path/to/snapshot/backup/dir.')
  args = parser.parse_args()

  main(args.dst_dir)

