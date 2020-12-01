#include "Complex.h"
#include "mpiconvolution.h"
#include "Array.h"

// Compile with:
// g++ -I .. -fopenmp exampleconv2.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

// size of problem
unsigned int m=8;

unsigned int mx=4;
unsigned int my=4;


inline void init(Complex *f, Complex *g, split dg){
  Complex I=-1;
  I=sqrt(I);
  unsigned int c=0;
  for(unsigned int lx=0;lx<dg.X;lx++){
    for(unsigned int ly=0;ly<dg.y;ly++){
      f[c]=lx+(ly+dg.y0)*I;
      g[c++]=2*lx+(ly+dg.y0+1)*I;
    }
  }
}

int main(int argc, char* argv[])
{
  //  fftw::maxthreads=get_max_threads();

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif
    
  int divisor=0;    // Test for best divisor
  int alltoall=-1;  // Test for best communication routine
  mpiOptions options(divisor,alltoall);
  
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
  MPIgroup group(MPI_COMM_WORLD,my);
  
  if(group.size > 1 && provided < MPI_THREAD_FUNNELED)
    fftw::maxthreads=1;
  
  defaultmpithreads=fftw::maxthreads;
  
  if(group.rank == 0) {
    cout << "Configuration: " 
	 << group.size << " nodes X " << fftw::maxthreads
	 << " threads/node" << endl;
  }
  if(group.rank < group.size) {
    bool main=group.rank == 0;
    if(main) cout << "2D centered Hermitian-symmetric convolution:" << endl;
    // Set up per-process dimensions
    split dg(2*mx-1,my,group.active);
    split du(mx+1,my,group.active);

    size_t align=sizeof(Complex);
    Complex *f=ComplexAlign(2*dg.n);
    Complex *g=ComplexAlign(2*dg.n);
    //    array2<Complex> f(2*mx-1,my,align);
    //    array2<Complex> g(2*mx-1,my,align);
    bool xcompact=true,ycompact=true;
    ImplicitHConvolution2MPI C(mx,my,xcompact,ycompact,dg,du,g,options);

    init(f,g,dg);
    if(main) cout << "\ninput:" << endl;
    if(main) cout << "f:" << endl;
    show(f,dg.x,dg.Y,group.active);
    if(main) cout << "g:" << endl;
    show(g,dg.x,dg.Y,group.active);
    
    /*
      cout << "input after symmetrization (done automatically):" << endl;
      HermitianSymmetrizeX(mx,my,mx-1,f);
      HermitianSymmetrizeX(mx,my,mx-1,g);
      cout << "f:" << endl << f << endl;
      cout << "g:" << endl << g << endl;
    */
    
    bool symmetrize=true;
    
    //    ImplicitHConvolution2 C(mx,my);
    C.convolve(f,g,symmetrize);
    if(main) cout << "\noutput:" << endl;
    if(main) cout << "h:" << endl;
    show(f,dg.X,dg.y,group.active);
  }
  MPI_Finalize();
  return 0;
}
