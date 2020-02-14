#!/usr/bin/python3

''' Changelog:
Feb 12th 2020: Start cleaning up code + genericizing.
Made for generic number of columns
'''

import json
import glob
import os
import re
import argparse
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from benedict import benedict

# starts_w_num = re.compile(r'^\d\d\d')


def df2txt(xf, columns=2, show_col=True):
    ''' This is really the key method here. Take the frame, divide into columns, group lines, get text
    '''
    tx = ''
    nf = xf.copy()
    nf.sort_values('y', inplace=True)
    # TODO: figure this out in a more generic way
    nf = nf.loc[(nf.y > 0.1)]  # dropping the top gutter
    nf = nf.loc[((nf.ymax - nf.ymin) < 0.03)]  # drop anything massive [for now]
    # split to cols, numbering will be 0 --> columns from LTR, which we will adjust for
    nf['col_bin'] = pd.cut(nf.x, bins=columns, labels=False)
    nf['visited'] = False
    # rhc = nf[(nf.col_bin == 1)]
    # lhc = nf[(nf.col_bin == 0)]
    for col_index in range(columns - 1, -1, -1):  # iterate from R to L
        if show_col:
            tx += f'<COLUMN {columns - col_index}>\n'
        col = nf.loc[nf.col_bin == col_index].copy()
        # col = x.copy()
        for qq in col.itertuples():
            if qq.visited:
                continue
            collinear = col.loc[(col.ymin < qq.y) & (col.ymax > qq.y)].copy()
            collinear.sort_values('x', ascending=False, inplace=True)
            if collinear.visited.any():
                continue
            col.loc[col.index.isin(collinear.index),'visited'] = True
            tx += ' '.join(collinear.t) + '\n'
    return tx


def makeframe(a):
    x = []
    px = []
    y = []
    py = []
    bx, by = [], []
    allc = []
    centroids_x, centroids_y = [], []
    texts = []
    breaks = []
    for pg in a['pages']:
        for blk in pg['blocks']:
            for p in blk['paragraphs']:
                for w in p['words']:
                    c = w['boundingBox']['normalizedVertices']
                    coords = []
                    t = ''.join([s['text'] for s in w['symbols']])
                    for v in c:
                        coords.append((v['x'], v['y']))
                        x.append(v['x'])
                        y.append(v['y'])
                    centroids_x.append(np.average(x[-4:]))
                    centroids_y.append(np.average(y[-4:]))
                    texts.append(t)
                    allc.append(coords)
                    # Look for the end type? For now, not useful
                    # bk = benedict(w['symbols'][-1])
                    # bktype = bk.get('property.detectedBreak.type')
                    # if bktype:
                    #     breaks.append(bktype)
                    # else:
                    #     breaks.append('')
    df = pd.DataFrame()
    df['x'] = centroids_x
    df['y'] = centroids_y
    df['t'] = texts
    df['c'] = allc

    # Unused for now: the following block handles the corners. For now, not needed in the DataFrame, may be useful later
    '''
    tl, tr, br, bl = [], [], [], []
    for itl, itr, ibr, ibl in allc:
        tl.append(itl)
        tr.append(itr)
        br.append(ibr)
        bl.append(ibl)

    df['tl'] = tl
    df['tr'] = tr
    df['br'] = br
    df['bl'] = bl
    '''

    ymin, ymax = [], []
    for cset in allc:
        ys = [iy for ix, iy in cset]
        ymin.append(min(ys))
        ymax.append(max(ys))

    df['ymin'] = ymin
    df['ymax'] = ymax

    return df


makaf = '־'


def cleanup(txt):
    ''' Handle at least some of the extra spaces we generated in the process
    '''
    txt = txt.replace(' " ', '"')
    txt = txt.replace(' - ', '-')
    txt = txt.replace(' ' + makaf, makaf)
    txt = txt.replace(makaf + '\n', '')
    txt = txt.replace(makaf + ' ', makaf)
    txt = txt.replace(' + ', '+')
    txt = txt.replace(' ,', ',')
    txt = txt.replace(' )', ')')
    txt = txt.replace(' .', '.')
    txt = txt.replace('( ', '(')
    txt = txt.replace(" '", "'")
    txt = txt.replace(' *', '*')
    txt = txt.replace(' :', ':')
    return txt


# TODO: Make the filename hooks here generic
def files_to_text(files, output_directory, columns=2):
    for file in tqdm(files):
        d = benedict.from_json(file)
        for r in tqdm(d['responses'], leave=False):
            df = makeframe(r['fullTextAnnotation'])
            text = df2txt(df, columns=columns)
            text = cleanup(text)
            page_num = r['context']['pageNumber']

            page_handle = f'page_{page_num:04d}.txt'
            output_handle = os.path.join(output_directory, page_handle)

            with open(output_handle, 'w') as f:
                f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse columnar RTL text from Google Cloud OCR')
    parser.add_argument('columns', type=int)
    parser.add_argument('files', nargs='+', help="List of output files to parse")
    parser.add_argument('output_directory', help="Directory to write to")
    
    # Print help if empty
    parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    args = parser.parse_args()
    
    files_to_text(args.files, args.output_directory, args.columns)

