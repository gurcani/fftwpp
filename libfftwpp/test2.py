#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 03:38:11 2020

@author: ogurcan
"""

import numpy as np
import os,sys
from mpi4py import MPI
from numpy.ctypeslib import ndpointer
from ctypes import CDLL,CFUNCTYPE,POINTER,c_double,c_int,c_uint,c_void_p,c_bool,Structure,byref
base = os.path.dirname(os.path.abspath(__file__))
clib = CDLL(os.path.join(base, 'libfftwpp.so'))
sys.path.insert(1, os.path.realpath('../wrappers/'))
#clib = CDLL(os.path.join(base, 'libfftwpp.so'))
import fftwpp

class split(Structure):
  _fields_=[("X",c_uint),("Y",c_uint)
            ,("x",c_uint),("y",c_uint)
            ,("x0",c_uint),("y0",c_uint)
            ,("n",c_uint)
            ]

class group(Structure):
  _fields_=[("rank",c_int),("size",c_int)
            ]

clib.fftwpp_mpi_set_options.argtypes = [c_int,c_int,c_uint,c_uint]

clib.fftwpp_plan_rcfft2dmpi.argtypes = [c_void_p, c_void_p, ndpointer(dtype = float),ndpointer(dtype = np.complex128)]
clib.fftwpp_plan_rcfft2dmpi.restype = c_void_p

clib.fftwpp_rcfft2dmpi_forward0.argtypes = [c_void_p,ndpointer(dtype = float),ndpointer(dtype = np.complex128)]
clib.fftwpp_rcfft2dmpi_backward0.argtypes = [c_void_p,ndpointer(dtype = np.complex128),ndpointer(dtype = float)]

clib.fftwpp_mpi_group.restype = c_void_p
clib.fftwpp_mpi_group.argtypes = [c_uint, c_void_p]

clib.fftwpp_mpi_split.restype = c_void_p
clib.fftwpp_mpi_split.argtypes = [c_uint, c_uint, c_void_p]

clib.fftwpp_create_hconv2d_mpi.restype = c_void_p
clib.fftwpp_create_hconv2d_mpi.argtypes=[c_void_p,c_void_p,c_bool, c_bool,ndpointer(dtype = np.complex128)]

clib.fftwpp_hconv2d_mpi_convolve.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

clib.fftwpp_doublealign.argtypes=[c_uint]
clib.fftwpp_doublealign.restype=POINTER(c_double)

clib.fftwpp_complexalign.argtypes=[c_uint]
clib.fftwpp_complexalign.restype=POINTER(c_double)

comm=MPI.COMM_WORLD
commp=MPI._addressof(comm)

xcomp,ycomp=True,True

mx,my=4,4
nx=2*mx-int(xcomp)
ny=2*my-int(ycomp)
nyp=int(ny/2+1)

grpptr=clib.fftwpp_mpi_group(nyp,commp)
grp=group.from_address(grpptr)

print("grp.size=",grp.size,"grp.rank=",grp.rank)

if (grp.rank<grp.size):
    dgptr=clib.fftwpp_mpi_split(nx,nyp,grpptr)
    duptr=clib.fftwpp_mpi_split(mx+int(xcomp),nyp,grpptr)
    du=split.from_address(duptr)
    dg=split.from_address(dgptr)
    
    fc=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.n),shape=(2*dg.n,)).view(dtype=complex)
    f=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.X*dg.y),shape=(2*dg.X*dg.y,)).view(dtype=complex).reshape((dg.X,dg.y))
    g=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.X*dg.y),shape=(2*dg.X*dg.y,)).view(dtype=complex).reshape((dg.X,dg.y))

    cptr=clib.fftwpp_create_hconv2d_mpi(dgptr,duptr,xcomp,ycomp,f)
    f.fill(0)
    g.fill(0)
    f[int(not xcomp):,:]=np.array([[l+1j*(m+dg.y0) for m in range(dg.y)] for l in range(dg.X-int(not xcomp)) ])
    g[int(not xcomp):,:]=np.array([[2*l+1j*(m+dg.y0+1) for m in range(dg.y)] for l in range(dg.X-int(not xcomp))])
    if (dg.y0+dg.y==dg.Y and not ycomp):
        f[:,-1]=0
        g[:,-1]=0
    print("f=",f)
    print("g=",g)
    clib.fftwpp_hconv2d_mpi_convolve(cptr,f,g)
    h=f
    print("h=",h)