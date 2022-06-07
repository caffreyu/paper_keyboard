from PIL import Image
import sys 

fname = sys.argv[1]
fname_ls = fname.split('.')
fname_ls[-1] = 'png'
fname_updated = '.'.join(fname_ls)

img = Image.open(fname)
img.save(fname_updated)
