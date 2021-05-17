#include <iostream>
//#include "maxent.h"
#include <vector>
#include <stdlib.h>

#include <selforg/matrix.h>

using namespace std; 




// this is to transform a 3d matrix to two 2d one !!
int main(){
  /*Matrix a;
  a.set(10,5);
  a.toId();
  
  cout<< a<< endl<< endl<<endl;
  Matrix aa;
  aa.set(2,25);
  aa = a.reshape(2,25);

  Matrix cc;
  cc.set(5,5);
  cc = aa.row(0).reshape(5,5);

  cout<< a<< endl<< endl<<endl;
  cout<< aa <<endl<< endl<<endl;
  cout<< cc<< endl<< endl<<endl;
  */

  matrix::Matrix a;
  a.set(2,3);
  a.val(1,0) = 299;
  a.val(1,1) = 9;
  a.val(1,2) = 2;
  cout << a << endl<< endl;
  // cout <<( a ^ matrix::T )<< endl;
  // cout << a<< endl<< endl;
  
  // matrix::Matrix b;
  // b.set(3,2);
  // b = a.reshape(3,2);
  // matrix::Matrix c = a.row(1).reshape(3,1);
  // // b = a.row(0).reshape(3,1); 
  // cout << b << endl<< endl;
  // cout << a << endl<< endl;
  
  // cout <<"lala"<< c << endl<< endl;
  matrix::Matrix b =a;
  b.reshape(3,2);
  cout << b << endl<< endl;
  
  return 0;
}
