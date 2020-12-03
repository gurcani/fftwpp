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
from ctypes import CDLL,POINTER,c_double,c_int,c_uint,c_void_p,c_bool,Structure
base = os.path.dirname(os.path.abspath(__file__))
clib = CDLL(os.path.join(base, 'libfftwpp.so'))
sys.path.insert(1, os.path.realpath('../wrappers/'))

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

clib.fftwpp_create_hconv2d_mpi_adv2.restype = c_void_p
clib.fftwpp_create_hconv2d_mpi_adv2.argtypes =[c_void_p,c_void_p,c_bool, c_bool,ndpointer(dtype = np.complex128)]

clib.fftwpp_hconv2d_mpi_convolve_adv2.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128) ]

clib.fftwpp_create_hconv2d_mpi_adv34.restype = c_void_p
clib.fftwpp_create_hconv2d_mpi_adv34.argtypes =[c_void_p,c_void_p,
                                                c_bool, c_bool,
                                                ndpointer(dtype = np.complex128)]

clib.fftwpp_hconv2d_mpi_convolve_adv34.argtypes = [ c_void_p,
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128),
                                          ndpointer(dtype = np.complex128)]

clib.fftwpp_create_hconv2d_mpiAB.restype = c_void_p
clib.fftwpp_create_hconv2d_mpiAB.argtypes =[c_void_p,c_void_p,
                                                c_bool, c_bool,
                                                ndpointer(dtype = np.complex128),c_uint,c_uint]

clib.fftwpp_hconv2d_mpi_convolve_adv.argtypes = [ c_void_p,
                                          POINTER(POINTER(c_double)),
                                          c_void_p]


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

Lx=2*np.pi
Ly=2*np.pi

dkx=2*np.pi/Lx
dky=2*np.pi/Ly

#ky=np.arange(0,nyp)*dkx
kx,ky=np.meshgrid(np.arange(-mx+int(xcomp),mx)*dkx,np.arange(0,nyp)*dkx,indexing='ij')
ksqr=(kx**2,ky**2)
grpptr=clib.fftwpp_mpi_group(nyp,commp)
grp=group.from_address(grpptr)

print("grp.size=",grp.size,"grp.rank=",grp.rank)

if (grp.rank<grp.size):
    dgptr=clib.fftwpp_mpi_split(nx,nyp,grpptr)
    duptr=clib.fftwpp_mpi_split(mx+int(xcomp),nyp,grpptr)
    du=split.from_address(duptr)
    dg=split.from_address(dgptr)
    
    phik=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.X*dg.y),shape=(2*dg.X*dg.y,)).view(dtype=complex).reshape((dg.X,dg.y))
    nk=np.ctypeslib.as_array(clib.fftwpp_complexalign(2*dg.X*dg.y),shape=(2*dg.X*dg.y,)).view(dtype=complex).reshape((dg.X,dg.y))

    cptr=clib.fftwpp_create_hconv2d_mpiAB(dgptr,duptr,xcomp,ycomp,phik,6,2)
    phik.fill(0)
    nk.fill(0)
    
    phik[int(not xcomp):,:]=np.array([[l+1j*(m+dg.y0) for m in range(dg.y)] for l in range(dg.X-int(not xcomp)) ])
    nk[int(not xcomp):,:]=np.array([[2*l+1j*(m+dg.y0+1) for m in range(dg.y)] for l in range(dg.X-int(not xcomp))])
    if (dg.y0+dg.y==dg.Y and not ycomp):
        phik[:,-1]=0
        nk[:,-1]=0
    G=(1j*kx*phik,1j*ky*phik,-1j*kx*ksqr*phik,-1j*ky*ksqr*phik,1j*kx*nk,1j*ky*nk)
    Gp=(POINTER(c_double)*len(G))(*[l.ctypes.data_as(POINTER(c_double)) for l in G])
    clib.fftwpp_hconv2d_mpi_convolve_adv(cptr,Gp,clib.multadvection62)
    G=[np.rint(l.real)+1j*np.rint(l.imag) for l in G]
    print("G[0]=",G[0])
    print("G[1]=",G[1])
    
    # Verification:
    f=phik.copy()
    cptr2=clib.fftwpp_create_hconv2d_mpi(dgptr,duptr,xcomp,ycomp,f,6,2)
    f=1j*kx*phik
    g=-1j*ky*ksqr*phik
    clib.fftwpp_hconv2d_mpi_convolve(cptr2,f,g)
    h0=f.copy()
    f=1j*ky*phik
    g=-1j*kx*ksqr*phik
    clib.fftwpp_hconv2d_mpi_convolve(cptr2,f,g)
    h0-=f
    h0=np.rint(h0.real)+1j*np.rint(h0.imag)
    print("h0=",h0)
    
    f=1j*kx*phik
    g=1j*ky*nk
    clib.fftwpp_hconv2d_mpi_convolve(cptr2,f,g)
    h1=f.copy()
    f=1j*ky*phik
    g=1j*kx*nk
    clib.fftwpp_hconv2d_mpi_convolve(cptr2,f,g)
    h1-=f
    h1=np.rint(h1.real)+1j*np.rint(h1.imag)
    print("h1=",h1)   