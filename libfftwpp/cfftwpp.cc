#include "Complex.h"
#include "cfftwpp.h"
#include <iostream>

extern "C" {
  using namespace utils;
  using namespace fftwpp;

  
  
  // This 2D version of the scheme of Basdevant, J. Comp. Phys, 50, 1983
  // applied to passive scalar problem. Requires 7 FFTs per stage, with 3 inputs and 4 outputs.
  // one could also do 4 inputs and 3 outputs.
  
  void multadvection34(double **F, unsigned int m,
		      const unsigned int indexsize,
		      const unsigned int *index,
		      unsigned int r, unsigned int threads)
  {
    double* F0=F[0];
    double* F1=F[1];
    double* F2=F[2];
    double* F3=F[3];
#ifdef __SSE2__
    unsigned int m1=m-1;
    PARALLEL(
	     for(unsigned int j=0; j < m1; j += 2) {
	       double *F0j=F0+j;
	       double *F1j=F1+j;
	       double *F2j=F2+j;
	       double *F3j=F3+j;
	       Vec u=LOAD(F0j);
	       Vec v=LOAD(F1j);
	       Vec n=LOAD(F2j);
	       STORE(F0j,v*v-u*u);
	       STORE(F1j,u*v);
	       STORE(F2j,n*u);
	       STORE(F3j,n*v);
	     }
	     );
    if(m % 2) {
      double u=F0[m1];
      double v=F1[m1];
      double n=F2[m1];
      F0[m1]=v*v-u*u;
      F1[m1]=u*v;
      F2[m1]=n*u;
      F3[m1]=n*v;
    }
#else
    for(unsigned int j=0; j < m; ++j) {
      double u=F0[j];
      double v=F1[j];
      double n=F2[j];
      F0[j]=v*v-u*u;
      F1[j]=u*v;
      F2[j]=n*u;
      F3[j]=n*v;
    }
#endif
  }

  void multadvection62(double **F, unsigned int m,
		      const unsigned int indexsize,
		      const unsigned int *index,
		      unsigned int r, unsigned int threads)
  {
    double* F0=F[0];
    double* F1=F[1];
    double* F2=F[2];
    double* F3=F[3];
    double* F4=F[4];
    double* F5=F[5];
#ifdef __SSE2__
    unsigned int m1=m-1;
    PARALLEL(
	     for(unsigned int j=0; j < m1; j += 2) {
	       double *F0j=F0+j;
	       double *F1j=F1+j;
	       double *F2j=F2+j;
	       double *F3j=F3+j;
	       double *F4j=F4+j;
	       double *F5j=F5+j;
	       Vec dxphi=LOAD(F0j);
	       Vec dyphi=LOAD(F1j);
	       Vec dxom=LOAD(F2j);
	       Vec dyom=LOAD(F3j);
	       Vec dxn=LOAD(F4j);
	       Vec dyn=LOAD(F5j);
	       STORE(F0j,dxphi*dyom-dyphi*dxom);
	       STORE(F1j,dxphi*dyn-dyphi*dxn);
	     }
	     );
    if(m % 2) {
      double dxphi=F0[m1];
      double dyphi=F1[m1];
      double dxom=F2[m1];
      double dyom=F3[m1];
      double dxn=F4[m1];
      double dyn=F5[m1];
      F0[m1]=dxphi*dyom-dyphi*dxom;
      F1[m1]=dxphi*dyn-dyphi*dxn;
    }
#else
    for(unsigned int j=0; j < m; ++j) {
      double dxphi=F0[j];
      double dyphi=F1[j];
      double dxom=F2[j];
      double dyom=F3[j];
      double dxn=F4[j];
      double dyn=F5[j];
      F0[j]=dxphi*dyom-dyphi*dxom;
      F1[j]=dxphi*dyn-dyphi*dxn;
    }
#endif
  }  
}
