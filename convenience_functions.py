'''Some convenience functions'''
import sys

from numba import njit

@njit
def print_progress_njit(i, total):
    '''Print progress while in no-python mode'''
    frac = i/total
    print('Progress: ', frac*100, '%', i, '/', total, '\r')

def print_progress(i, total):
    '''Print pretty progress bar without needing tqdm'''
    frac = i/total
    sys.stdout.flush()
    print('\r['+'@'*round(frac*40)+'-'*round((1-frac)*40)+
          '] {}/{}, {:.1%}'.format(i, total, frac), end='\r')
