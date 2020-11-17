import os

"""
A collection of functions to help with file stuff.
"""

def maybe_mkdir(fp):
	if not os.path.exists(fp):
		os.mkdir(fp)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
