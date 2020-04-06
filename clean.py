'''
Looks for .txt files in <indir> and runs them through the strip_headers()
function, writing the result with the same filename to <outdir>.
replace this with clean.py in https://github.com/mbforbes/Gutenberg.git
'''

import glob
import sys

from tqdm import tqdm

import gutenberg.cleanup.strip_headers as strip_headers
from gutenberg._util.os import reopen_encoded
from gutenberg import Error
import pdb
from nltk.tokenize import sent_tokenize



indir = sys.argv[1]
outdir = sys.argv[2]

try:
    linesentence = sys.argv[3]
except IndexError:
    linesentence = False

files = glob.glob(indir + '/*.TXT')
for f in tqdm(files):
    try:
        with reopen_encoded(open(f, 'r'), 'r', 'utf8') as infile:
            cleaned = strip_headers(infile.read())
            if linesentence: 
                cleaned = sent_tokenize(' '.join(cleaned.replace('\n',' ').split()))
        short = f.split('/')[-1]
        with open(outdir + '/' + short, 'w', encoding='utf8') as outfile:
            if linesentence:
                for sentence in cleaned:
                    outfile.write(f'{sentence}\n')
            else:
                outfile.write(cleaned)
    except:
        print('Error processing', f, '; skipping...')
