#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:10:11 2019

@author: hm1234
"""

import numpy as np
import matplotlib.pyplot as plt
from boututils.datafile import DataFile

#class LineDrawer(object):
#    lines = []
#    def draw_line(self):
#        ax = plt.gca()
#        xy = plt.ginput(2)
#
#        x = [p[0] for p in xy]
#        y = [p[1] for p in xy]
#        line = plt.plot(x,y)
#        ax.figure.canvas.draw()
#
#        self.lines.append(line)
#        
#plt.plot([1,2,3,4,5])
#ld = LineDrawer()
#ld.draw_line() # here you click on the plot

#def draw_boundary(gridFile):
#    grid_dat = DataFile(gridFile)
#    R = grid_dat['Rxy']
#    Z = grid_dat['Zxy']
#    nx = grid_dat['nx']
#    ny = grid_dat['ny']
#    
#    fig, ax = plt.subplots()
#    for i in range(nx):
#        ax.plot(R[:, i], Z[:, i], linewidth=1, color='k', alpha=0.5)
#        
#    for i in range(ny):
#        ax.plot(R[i, :], Z[i, :], linewidth=1, color='k', alpha=0.1)
#    
#    print('left click to select points')
#    print('right click to deselect last point')
#    print('middle click to finish input')
#    plt.axis('scaled')
#    plt.tight_layout()
#        
#    # global points
#    points = plt.ginput(n=-1)
#    points = [item for t in points for item in t]
#    
#    plt.close()
#    
#    return points


#def extract_gpoints(gfile):
#    f = open(gfile)
#    lines = f.read()
#    
#    core = lines.split('\n\n')[0]
#    core = core.split('\n')[1:]
#    
#    dec_idx = []
#    for i in range(len(core)):
#        dec_idx.append([pos for pos, char in enumerate(core[0]) if char == '.'])
#    
#    test = []
#    for i in range(len(core)):
#        for j in dec_idx[i]:
#            test.append(eval(core[i][(j-2):(j+15)]))
#        
#    return test


def draw_boundary(gfile):
    f = open(gfile)
    lines = f.read()
    
    bndry = lines.split('\n\n')[1]
    bndry_head = bndry.split('\n')[0]
    
    num_sep_points = eval(bndry_head.split('\t')[0])
    num_bndry_points = eval(bndry_head.split('\t')[1].split('\n')[0])
    
    nsp = num_sep_points
    nbp = num_bndry_points
    
    bndry = bndry.split('\n')[1:]
    bndry = bndry[0:-1]
    
    dec_idx = []
    for i in range(len(bndry)):
        dec_idx.append([pos for pos, char in enumerate(bndry[i]) if char == '.'])
    
    bndry_coords = []
    for i in range(len(bndry)):
        for j in dec_idx[i]:
            bndry_coords.append(eval(bndry[i][(j-2):(j+15)]))
            
    print('left click to select points')
    print('right click to deselect last point')
    print('middle click to finish input')
            
    fig, ax = plt.subplots(figsize=(10,15))
    # plt.figure(figsize=(10,15))
    ax.scatter(bndry_coords[0:][::2][:nsp], bndry_coords[1:][::2][:nsp])
    ax.plot(bndry_coords[0:][::2][nsp:], bndry_coords[1:][::2][nsp:])
    # plt.gca().set_aspect('equal', adjustable='box')
    
    plt.axis('scaled')
    plt.tight_layout()
        
    # global points
    points = plt.ginput(n=-1)
    points = [item for t in points for item in t]
    
    plt.close()
        
    return points
    

def modify_boundary(gfile, newName='test.g', bndry_coords=[]):
    f = open(gfile)
    lines = f.read()
    f.close()

    core = lines.split('\n\n')[0]
    bndry = lines.split('\n\n')[1]
    bndry_head = bndry.split('\n')[0] + '\n'
    bndry = bndry.split('\n')[1:]
    bndry = bndry[0:-1]
    
    if len(bndry_coords) == 0:
        num_bndry_points = eval(bndry_head.split('\t')[1].split('\n')[0])
        num_sep_points = eval(bndry_head.split('\t')[0])
        nsp = num_sep_points
        dec_idx = []
        for i in range(len(bndry)):
            dec_idx.append([pos for pos, char in enumerate(bndry[i]) if char == '.'])
        old_bndry_coords = []
        for i in range(len(bndry)):
            for j in dec_idx[i]:
                old_bndry_coords.append(eval(bndry[i][(j-2):(j+15)]))
                
        old_bndry_coords = old_bndry_coords[2*nsp:]
    else:
        num_bndry_points = eval(int(len(bndry_coords)/2))

    nbp = num_bndry_points
    
    new_bndry_head = '0\t{}\n'.format(nbp)

    new_core = core.split('\n')[0] + '\n'
    
    for j in core.split('\n')[1:]:
        new_core += j + '\n'
    
    new_bndry = new_bndry_head
    
    if len(bndry_coords) == 0:
        bndry_coords = old_bndry_coords
    
    for i, j in enumerate(bndry_coords):
        i += 1 
        if str(j)[0] != '-':
            new_bndry += ' '
        if i % 5 != 0:
            new_bndry += '{:.10E}'.format(j)
        else:
            new_bndry += '{:.10E}\n'.format(j)
    
    new_gfile = new_core + '\n' + new_bndry

    f = open(newName, 'w+')
    f.write(new_gfile)
    f.close()

    return new_gfile


a = [ 6.3400000000E-01, 7.0400000000E-01, 6.7600000000E-01, 7.5000000000E-01,
     9.6500000000E-01, 7.5000000000E-01, 1.1260000000E+00, 5.5000000000E-01,
     1.1260000000E+00, -5.5000000000E-01, 9.6500000000E-01, -7.5000000000E-01,
     6.7600000000E-01, -7.5000000000E-01, 6.3400000000E-01, -7.0400000000E-01,
     6.3400000000E-01, 7.0400000000E-01]

gfile = '63127_1400ms.g'

new_gfile = modify_boundary(gfile, newName='test_63127.g')

#points = draw_boundary(gfile)
#new_gfile = modify_boundary(gfile, bndry_coords=points)




