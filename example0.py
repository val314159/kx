# -*- coding: utf-8 -*-

import os

os.system('rm -fr data')
os.system('mkdir -p preview')
os.system('mkdir -p data/train/cats')
os.system('mkdir -p data/train/dogs')
os.system('mkdir -p data/validation/cats')
os.system('mkdir -p data/validation/dogs')

def loop(X,Y,L):
    for n in range(X,X+Y):
        os.system('cp train/cat.%s.jpg data/%s/cats/cat.%s.jpg' % (n,L,n-X))
        os.system('cp train/dog.%s.jpg data/%s/dogs/dog.%s.jpg' % (n,L,n-X))

loop(X=0, Y=1000, L='train')
loop(X=1000, Y=400, L='validation')
