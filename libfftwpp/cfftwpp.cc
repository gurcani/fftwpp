#include "Complex.h"
#include "cfftwpp.h"
#include <iostream>

extern "C" {
  using namespace utils;
  using namespace fftwpp;

  
  
  // This 2D version of the scheme of Basdevant, J. Comp. Phys, 50, 1983
  // applied to passive scalar problem. Requires 7 FFTs per stage, with 3 inputs and 4 outputs.
  // one could also do 4 inputs and 3 outputs.
  void multadvection4(double **F, unsigned int m,
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
      double n=F2[m1];
      F0[j]=v*v-u*u;
      F1[j]=u*v;
      F2[j]=n*u;
      F3[j]=n*v;
    }
#endif
  }
}
