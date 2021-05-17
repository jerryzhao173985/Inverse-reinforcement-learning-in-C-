#include <iostream>
#include "maxent.h"
#include <vector>
#include <stdlib.h>
using namespace std; 
#include <selforg/matrix.h>


void testMaxent(){
  
  // int size=10;
  // //  Maxent q(0.2,0.9,0.05, 5,false,false);
  // Maxent* maxent = new Maxent(0.2,0.9,0.05, 5,false,true);;
  // //maxent = new Maxent(0.2,0.9,0.05, 5,false,true);
  // maxent ->init(size,2);
  // cout << "Q Learning 10x2\n";  
  
  // matrix::Matrix mac;
  // mac.set(2,2);      //real (1,2) .. (2,2)
  // mac.val(0,0) = 0;   //state
  // mac.val(0,1) = 0;   //state
  // mac.val(1,0) = 0;    //action
  // mac.val(1,1) = 0;    //action
  // matrix::Matrix freq= maxent->find_svf(5, mac);
  // std::cout << freq << std::endl;
  
  int n_trajectories = 5;
  int trajectory_length  = 4;
  int epochs = 2000;
  double discount = 0.01;
  double learning_rate = 0.01;

  int n_actions = 4;
  int grid_size = 2;
  int n_states = (int) (grid_size* grid_size);


  matrix::Matrix feature_matrix;  //(number of states, dimensionality of states== default is 4??)
  feature_matrix.set(4,4);
  feature_matrix.toId();
  //std::cout  << feature_matrix.getM() <<feature_matrix.getN()  <<std::endl;

  matrix::Matrix ground_r;
  ground_r.set(4,1);
  ground_r.val(3,0) = 1;

  // (5,4,2) => (2* 5, 4) =>(2,5*4)
  double data[40] = {0.,2.,0.,2.,
                     0.,1.,1.,3.,
                     0.,1.,3.,3.,
                     0.,1.,3.,3.,
                     0.,2.,2.,3.,
                     1.,3.,1.,1.,
                     0.,3.,1.,3.,
                     0.,1.,0.,1.,
                     0.,1.,0.,0.,
                     1.,1.,0.,0. };

  //(2*T , L) => (0:2 , T*L) then (1, T*L) -> (T, L) based on specific state,
  //                              (2, T*L) -> (T, L) based on specific action.
  //if add another dimension reward then ((3, T*L) -> (T, L) based on corresponding reward)
  matrix::Matrix trajectories;
  trajectories.set(10, 4, data);
  //std::cout  <<  trajectories  <<std::endl;
  
  //transition_probability is fixed for both before training and after training!
  //trans_prob for every row should add up to 1 meaning (state, action) -> next_state
  double trans[64] = {0.15 , 0.775, 0.075, 0.   ,
                      0.075, 0.85 , 0.   , 0.075,
                      0.075, 0.   , 0.15 , 0.775,
                      0.   , 0.075, 0.075, 0.85 ,

                      0.15 , 0.075, 0.775, 0.   ,
                      0.075, 0.15 , 0.   , 0.775,
                      0.075, 0.   , 0.85 , 0.075,
                      0.   , 0.075, 0.075, 0.85 ,
  
                      0.85 , 0.075, 0.075, 0.   ,
                      0.775, 0.15 , 0.   , 0.075,
                      0.075, 0.   , 0.85 , 0.075,
                      0.   , 0.075, 0.775, 0.15 ,
  
                      0.85 , 0.075, 0.075, 0.   ,
                      0.075, 0.85 , 0.   , 0.075,
                      0.775, 0.   , 0.15 , 0.075,
                      0.   , 0.775, 0.075, 0.15 };
  //(A*N , N) => (A , N*N) then (a, N*N) -> (N, N) based on specific a (action) 
  matrix::Matrix transition_probabilities;  //(state, action, state) (N, A, N) => (A*N , N) => (A , N*N)
  transition_probabilities.set(n_states*n_actions, n_states, trans);
  //std::cout  <<  transition_probabilities  <<std::endl;
  //std::cout  << transition_probabilities.getM() <<transition_probabilities.getN()  <<std::endl;

  // Maxent* maxent = new Maxent(trajectories, feature_matrix, ground_r, 5, 4, 2000, 0.01, 0.01);  
  //false initialize, I think empty initialize is okay, no need for intermidiate results
  Maxent* maxent = new Maxent();
  
  int nn = 4; int dd = 4; int aa = 4; int tt = 5; int ll = 4; 
  maxent->init(nn, dd, aa, tt, ll);

  //---------Cautious when using reshape,: change the original matrix!!
  //matrix::Matrix state_sequence = trajectories;
	//state_sequence.set(tt, ll);
	//state_sequence = ((state_sequence.reshape(2, tt*ll)).row(0)).reshape(tt, ll);
	//matrix::Matrix action_sequence = trajectories;
	//action_sequence.set(tt, ll);
  //action_sequence = ((action_sequence.reshape(2, tt*ll)).row(1)).reshape(tt, ll);

  //std::cout << "all: "<< std::endl<< trajectories  <<std::endl;
	//std::cout << "Intermidiate debug(svf): "  <<state_sequence  <<std::endl<<std::endl<< action_sequence <<std::endl;


  
  matrix::Matrix reward;
  reward.set(n_states, 1);
  reward = maxent->irl(feature_matrix, n_actions, discount, transition_probabilities, trajectories, epochs, learning_rate);
  
  std::cout<< "irl reward is: " << std::endl<< reward<< std::endl;
  std::cout<< "ground truth reward is: " << std::endl<< ground_r << std::endl;




}




int main(){
  srand(time(0));
  
  cout << "******************** TEST Maxent\n";
  testMaxent();
 
  return 0;
}




// test expected result: (from python)

// ('----before----\ntrajectories0: ', array([[[0, 1, 0],
//         [2, 3, 0],
//         [0, 1, 0],
//         [2, 1, 0]],

//        [[0, 0, 0],
//         [1, 3, 0],
//         [1, 1, 1],
//         [3, 3, 0]],

//        [[0, 0, 0],
//         [1, 1, 1],
//         [3, 0, 1],
//         [3, 1, 1]],

//        [[0, 0, 0],
//         [1, 1, 1],
//         [3, 0, 1],
//         [3, 0, 1]],

//        [[0, 1, 0],
//         [2, 1, 0],
//         [2, 0, 1],
//         [3, 0, 1]]]))
// ('-----after---\ntrajectories0: ', array([[[0, 1],
//         [2, 3],
//         [0, 1],
//         [2, 1]],

//        [[0, 0],
//         [1, 3],
//         [1, 1],
//         [3, 3]],

//        [[0, 0],
//         [1, 1],
//         [3, 0],
//         [3, 1]],

//        [[0, 0],
//         [1, 1],
//         [3, 0],
//         [3, 0]],

//        [[0, 1],
//         [2, 1],
//         [2, 0],
//         [3, 0]]]))
// ('trajectories0: ', array([[0, 2, 0, 2],
//        [0, 1, 1, 3],
//        [0, 1, 3, 3],
//        [0, 1, 3, 3],
//        [0, 2, 2, 3]]))
// ('-----\ntrajectories1: ', array([[1, 3, 1, 1],
//        [0, 3, 1, 3],
//        [0, 1, 0, 1],
//        [0, 1, 0, 0],
//        [1, 1, 0, 0]]))
// ('feature_matrix: ', array([[1., 0., 0., 0.],
//        [0., 1., 0., 0.],
//        [0., 0., 1., 0.],
//        [0., 0., 0., 1.]]))
// ('Groung-Trueth: ', array([0, 0, 0, 1]))
