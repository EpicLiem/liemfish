"""Portable timing experiments, including the trained reference and schema-v2 data."""
import sys

sys.dont_write_bytecode = True
import bundle

bundle.DIRECTORIES += ('configs', 'browser', 'data/timing-v2')
bundle.FILES += ('models/personalized/heads-01.pt',)

if __name__ == '__main__':
    bundle.main()
