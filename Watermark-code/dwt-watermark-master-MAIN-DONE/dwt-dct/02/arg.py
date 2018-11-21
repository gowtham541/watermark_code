import argparse
parser = argparse.ArgumentParser(description= 'nhap')
parser.add_argument('-f', '--my-foo', default='foobar')
parser.add_argument('-b', '--bar-value', default=3.14)
args = parser.parse_args()
print(args.my_foo)