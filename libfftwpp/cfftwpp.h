#ifndef _CFFTWPP_H_
#define _CFFTWPP_H_

#include "Complex.h"
#include <mpi.h>
#include "mpiconvolution.h"

#ifdef  __cplusplus
extern "C" {
#endif
  using namespace utils;
  using namespace fftwpp;
  
  mpiOptions dfoptions;

  realmultiplier multadvection34;
  
  double __complex__ *fftwpp_complexalign(unsigned int n){
    return (double __complex__ *) ComplexAlign(n);
  }
  double *fftwpp_doublealign(unsigned int n){
    return doubleAlign(n);
  }
  void fftwpp_mpi_set_options(int a,int alltoall, unsigned int threads, unsigned int verbose){
    dfoptions=mpiOptions(a,alltoall,threads,verbose);
  }
  rcfft2dMPI* fftwpp_plan_rcfft2dmpi(split &df,split &dg,double *f,double __complex__ *g){
    return new rcfft2dMPI(df,dg,f,(Complex *)g,dfoptions);
  }
  void fftwpp_rcfft2dmpi_forward0(rcfft2dMPI *plan, double *f, double __complex__ *g){
    plan->Forward0(f,(Complex *)g);
  }
  void fftwpp_rcfft2dmpi_backward0(rcfft2dMPI *plan, double __complex__ *g, double *f){
    plan->Backward0((Complex *)g,f);
  }
  MPIgroup * fftwpp_mpi_group(unsigned int nyp,MPI_Comm &comm){
    MPIgroup *grp = new MPIgroup(comm,nyp);
    return grp;
  }

  split *fftwpp_mpi_split(unsigned int nx,unsigned int nyp, MPIgroup &grp){
    return new split(nx,nyp,grp.active);
  }
  
  ImplicitHConvolution2MPI* fftwpp_create_hconv2d_mpi(split &dg,split &du,
						      bool xcomp, bool ycomp, double __complex__ *g){
    unsigned int nx=dg.X,ny=(dg.Y-1)*2+xcomp;
    unsigned int nyp=ny/2+1;
    unsigned int mx=(nx+1)/2;
    unsigned int my=(ny+1)/2;
    return new ImplicitHConvolution2MPI(mx,my,xcomp,ycomp,dg,du,(Complex *)g,dfoptions);
  }
  void fftwpp_hconv2d_mpi_convolve(ImplicitHConvolution2MPI* hconv, double __complex__ *f, double __complex__ *g) {
    Complex *G[]={(Complex *)f,(Complex *)g};
    hconv->convolve(G,multbinary);
  }

  ImplicitHConvolution2MPI* fftwpp_create_hconv2d_mpi_adv2(split &dg,split &du,
						      bool xcomp, bool ycomp, double __complex__ *g){
    unsigned int nx=dg.X,ny=(dg.Y-1)*2+xcomp;
    unsigned int nyp=ny/2+1;
    unsigned int mx=(nx+1)/2;
    unsigned int my=(ny+1)/2;
    return new ImplicitHConvolution2MPI(mx,my,xcomp,ycomp,dg,du,(Complex *)g,dfoptions,2,2);
  }
  
  void fftwpp_hconv2d_mpi_convolve_adv2(ImplicitHConvolution2MPI* hconv, double __complex__ *f, double __complex__ *g) {
    Complex *G[]={(Complex *)f,(Complex *)g};
    hconv->convolve(G,multadvection2);
  }
  
  ImplicitHConvolution2MPI* fftwpp_create_hconv2d_mpi_adv34(split &dg,split &du,
							   bool xcomp, bool ycomp, double __complex__ *g){
    unsigned int nx=dg.X,ny=(dg.Y-1)*2+xcomp;
    unsigned int nyp=ny/2+1;
    unsigned int mx=(nx+1)/2;
    unsigned int my=(ny+1)/2;
    return new ImplicitHConvolution2MPI(mx,my,xcomp,ycomp,dg,du,(Complex *)g,dfoptions,3,4);
  }
  
  void fftwpp_hconv2d_mpi_convolve_adv34(ImplicitHConvolution2MPI* hconv, double __complex__ *f, double __complex__ *g, double __complex__ *h, double __complex__ *r) {
    Complex *G[]={(Complex *)f,(Complex *)g, (Complex *)h, (Complex *)r};
    hconv->convolve(G,multadvection34);
  }

  ImplicitHConvolution2MPI* fftwpp_create_hconv2d_mpiAB(split &dg,split &du,
							bool xcomp, bool ycomp, double __complex__ *g, unsigned int A, unsigned int B){
    unsigned int nx=dg.X,ny=(dg.Y-1)*2+xcomp;
    unsigned int nyp=ny/2+1;
    unsigned int mx=(nx+1)/2;
    unsigned int my=(ny+1)/2;
    return new ImplicitHConvolution2MPI(mx,my,xcomp,ycomp,dg,du,(Complex *)g,dfoptions,A,B);
  }
  
  void fftwpp_hconv2d_mpi_convolve_adv(ImplicitHConvolution2MPI* hconv, double __complex__ **Gp, realmultiplier mul) {
    Complex **G = (Complex **) Gp;
    hconv->convolve((Complex**)G,mul);
  }
  
#ifdef  __cplusplus
}
#endif

#endif
