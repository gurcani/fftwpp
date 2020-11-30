#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 03:38:11 2020

@author: ogurcan
"""

import numpy as np
import os
from mpi4py import MPI
from numpy.ctypeslib import ndpointer
from ctypes import CDLL,CFUNCTYPE,POINTER,c_double,c_int,c_uint,c_void_p,c_bool,Structure,byref
base = os.path.dirname(os.path.abspath(__file__))
clib = CDLL(os.path.join(base, 'libfftwpp.so'))

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


nx,ny=8,8
nyp=int(ny/2)+1
mx=int((nx+1)/2);
my=int((ny+1)/2);

grpptr=clib.fftwpp_mpi_group(nyp,commp)
grp=group.from_address(grpptr)

dfptr=clib.fftwpp_mpi_split(nx,ny,grpptr)
print("grp.size=",grp.size,"grp.rank=",grp.rank)
#print("comm.size=",comm.size)
dgptr=clib.fftwpp_mpi_split(nx,nyp,grpptr)
duptr=clib.fftwpp_mpi_split(mx,nyp,grpptr)
df=split.from_address(dfptr)
dg=split.from_address(dgptr)

if (grp.rank<grp.size):
    print("rank=",comm.rank,",dg.X=",dg.X,"dg.x=",dg.x,"dg.Y=",dg.Y,"dg.y=",dg.y,"dg.n=",dg.n)
    print("rank=",comm.rank,",df.X=",df.X,"df.x=",df.x,"df.Y=",df.Y,"df.y=",df.y,"df.n=",df.n)
    f0=np.ctypeslib.as_array(clib.fftwpp_doublealign(df.n),shape=(df.x,df.Y))
    f1=np.ctypeslib.as_array(clib.fftwpp_doublealign(df.n),shape=(df.x,df.Y))
    g0=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.X*dg.y),shape=(2*dg.X*dg.y,)).view(dtype=complex).reshape((dg.X,dg.y))
    g1=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.X*dg.y),shape=(2*dg.X*dg.y,)).view(dtype=complex).reshape((dg.X,dg.y))
    plan=clib.fftwpp_plan_rcfft2dmpi(dfptr,dgptr,f0,g0)
    cptr=clib.fftwpp_create_hconv2d_mpi(dgptr,duptr,False,False,g0)
    f0[:]=np.array([[j+i+df.x0 for j in range(df.Y)] for i in range(df.x)],dtype=float)
    f1[:]=np.array([[j+i+df.x0 for j in range(df.Y)] for i in range(df.x)],dtype=float)
    f0c=f0.copy()
    clib.fftwpp_rcfft2dmpi_forward0(plan,f0,g0)
    clib.fftwpp_rcfft2dmpi_forward0(plan,f1,g1)
    clib.fftwpp_hconv2d_mpi_convolve(cptr,g0,g1)
    h=g0
    print("h=",h)
    clib.fftwpp_rcfft2dmpi_backward0(plan,g0,f0)
    print("f0=",f0/df.X/df.Y)
