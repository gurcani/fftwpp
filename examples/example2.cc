#include "Array.h"
#include "fftw++.h"
#include <cstdlib>
#include <time.h>       /* time */

// Compile with
// g++ -I .. -fopenmp example2.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=get_max_threads();

  //srand(time(NULL));
  
  unsigned int nx=4, ny=4;
  size_t align=sizeof(Complex);
  
  array2<Complex> f(nx,ny,align);
  
  fft2d Forward2(-1,f);
  fft2d Backward2(1,f);
  
  for(unsigned int i=0; i < nx; i++) 
    for(unsigned int j=0; j < ny; j++) 
      f(i,j)=Complex(i,j);
      //f(i,j)=rand()%9+1;
  //      f(i,j)=i+j;

  cout << f << endl;
  
  Forward2.fft(f);
  
  cout << f << endl;
  
  Backward2.fftNormalized(f);

  cout << f << endl;
}
