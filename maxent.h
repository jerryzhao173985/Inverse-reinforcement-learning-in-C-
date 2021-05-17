#ifndef __MAXENT_H
#define __MAXENT_H


#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <functional>   // std::plus, std::minus
#include <algorithm>    // std::transform
//Read/write a file
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>
#include <time.h>
#include <dirent.h> //mkdir
#include <sys/types.h>
#include <sys/stat.h>
#include <numeric>      // std::accumulate

//selforg library
#include <selforg/matrix.h>

#include <selforg/abstractcontroller.h>
#include <selforg/controller_misc.h>

#include <assert.h>
#include <cmath>

#include <selforg/matrix.h>
#include <selforg/teachable.h>
#include <selforg/configurable.h>
#include <selforg/parametrizable.h>
#include <selforg/storeable.h>
#include <selforg/randomgenerator.h>


using namespace std;

/// implements Maxent
class Maxent : public Configurable, public Storeable {
public:

    /// constructor
    Maxent(/*const DiamondConf& conf = getDefaultConf()*/);
    
    /**
    param eps learning rate (typically 0.1)
    param discount discount factor for Q-values (typically 0.9)
    param exploration exploration rate (typically 0.02)
    param eligibility number of steps to update backwards in time
    param random_initQ if true Q table is filled with small random numbers at the start (default: false)
    param useSARSA if true, use SARSA strategy otherwise qlearning (default: false)
    param tau number of time steps to average over reward for col_rew
    */
    Maxent(matrix::Matrix trajectories, matrix::Matrix feature_matrix, matrix::Matrix ground_r, 
				 int n_trajectories = 5, int trajectory_length  = 4, int epochs = 2000, double discount = 0.01, double learning_rate = 0.01);

    virtual ~Maxent();

    /** initialisation with the given number of action and states
      @param actionDim number of actions
      @param stateDim number of states
      @param unit_map if 0 the parametes are choosen randomly.
      Otherwise the model is initialised to represent a unit_map with the given response strength.
    */
    virtual void init(int nn, int dd, int aa, int tt, int ll, RandGen* randGen = 0);


    /**
    Find the reward function for the given trajectories.
    @param feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    @param n_actions Number of actions A. int.
    @param discount: Discount factor of the MDP. double.
    @param transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    @param trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    @param epochs: Number of gradient descent steps. int.
    @param learning_rate: Gradient descent learning rate. double.
    -> Reward vector with shape (N,).
    */
    virtual matrix::Matrix irl(matrix::Matrix feature_matrix, int n_actions, double discount, 
        matrix::Matrix transition_probability, matrix::Matrix trajectories, int epochs, double learning_rate);

    /**
        Find the state visitation frequency from trajectories.
    @param n_states: Number of states. int.
    @param trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    */
    virtual matrix::Matrix find_svf(int n_states, matrix::Matrix trajectories);

    /**
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.
    @param feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    @param trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    */
    virtual matrix::Matrix find_feature_expectations(matrix::Matrix feature_matrix, matrix::Matrix trajectories);


    //-------------------------------from value iteration.py---------------------------------------
    /**
    Find the value function associated with a policy.
    @param policy: List of action ints for each state.
    @param n_states: Number of states. int.
    @param transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    @param reward: Vector of rewards for each state.
    @param discount: MDP discount factor. float.
    @param threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    */
    virtual matrix::Matrix value_func(matrix::Matrix policy, int n_states, matrix::Matrix transition_probabilities, 
                    matrix::Matrix reward, double discount, double threshold=0.01);
    
    /**
    Find the optimal value function.
    @param n_states: Number of states. int.
    @param n_actions: Number of actions. int.
    @param transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    @param reward: Vector of rewards for each state.
    @param discount: MDP discount factor. float.
    @param threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    */
    virtual matrix::Matrix optimal_value(int n_states, int n_actions, matrix::Matrix transition_probabilities, 
                  matrix::Matrix reward, double discount, double threshold=0.01);
    
    /**
    Find the optimal policy.
    @param n_states: Number of states. int.
    @param n_actions: Number of actions. int.
    @param transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    @param reward: Vector of rewards for each state.
    @param discount: MDP discount factor. float.
    @param threshold: Convergence threshold, default 1e-2. float.
    @param v: Value function (if known). Default None.
    @param stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    */
    virtual matrix::Matrix find_policy(int n_states, int n_actions, matrix::Matrix transition_probabilities,
                matrix::Matrix reward, double discount, double threshold=0.01, bool stochastic=true);


    /**
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.
    @param n_states: Number of states N. int.
    @param alpha: Reward. NumPy array with shape (N,).
    @param n_actions: Number of actions A. int.
    @param discount: Discount factor of the MDP. float.
    @param transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    @param trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    */
    virtual matrix::Matrix find_expected_svf(int n_states, matrix::Matrix r, int n_actions, double discount,
        matrix::Matrix transition_probabilities, matrix::Matrix trajectories);
    
    /**
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.
    @param x1: float.
    @param x2: float.
    -> softmax(x1, x2)
    */
    virtual double softmax(double x1, double x2);


    //-------------using value iteration's find_policy instead of this --------------
    //virtual matrix::Matrix find_policy(int n_states, matrix::Matrix r, int n_actions, double discount,
    //    matrix::Matrix transition_probability);
    
    /**
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.
    @param n_states: Number of states. int.
    @param n_actions: Number of actions. int.
    @param transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    @param reward: Reward vector mapping state int to reward. Shape (N,).
    @param discount: Discount factor. float.
    @param p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    @param optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    @param true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    */
    virtual double expected_value_difference(int n_states, int n_actions, matrix::Matrix transition_probability,
        matrix::Matrix reward, double discount, matrix::Matrix p_start_state, matrix::Matrix optimal_value, 
        matrix::Matrix true_reward);


    /**
    must include this to overrider the father class Storable
    */
    virtual bool store(FILE* f) const;

    virtual bool restore(FILE* f);

        




protected:
//     double eps;
//     double discount;
//     double exploration;
//     double eligibility; // is used as integer (only for configration)
//     bool random_initQ;

// public:
//     bool useSARSA; ///< if true, use SARSA strategy otherwise qlearning


// protected:
//     int tau;       ///< time horizont for averaging the reward
//     matrix::Matrix Q; /// < Q table (mxn) == (states x actions)


//     int* actions;    // ring buffer for actions
//     int* states;     // ring buffer for states
//     double* rewards; // ring buffer for rewards
//     int ringbuffersize; // size of ring buffers, eligibility + 1
//     double* longrewards; // long ring buffer for rewards for collectedReward
//     int t; // time for ring buffers
//     bool initialised;
//     double collectedReward; // sum over collected reward

    RandGen* randGen;
    int t;

    int NN;
    int DD;  // where N is the number of states and D is the dimensionality of the state.
    int AA;
    int TT;
    int LL;

    matrix::Matrix trajectories;  //(2*T , L)
    matrix::Matrix feature_matrix; //  (NN, DD)          (trajectory_length, trajectory_length)
    matrix::Matrix ground_r;
    matrix::Matrix transition_probabilities;

    matrix::Matrix policy; //(N ,A)

    matrix::Matrix r;

    int n_trajectories = 5;
    int trajectory_length  = 4;
    int epochs = 2000;
    double discount = 0.01;
    double learning_rate = 0.01;



};
	

#endif
