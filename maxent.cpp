#include "maxent.h"
#include <selforg/matrix.h>
#include <selforg/controller_misc.h>

// max example
#include <iostream>     // std::cout
#include <algorithm>    // std::max

#include <cmath>
#include <cerrno>
#include <cstring>
#include <cfenv>

Maxent::Maxent(): Configurable("Maxent", "$Id$"){	
	t=0;
}


Maxent::Maxent( matrix::Matrix trajectories, matrix::Matrix feature_matrix, matrix::Matrix ground_r, 
				 int n_trajectories, int trajectory_length , int epochs , double discount, double learning_rate)
  : Configurable("Maxent", "$Id$"),
    trajectories(trajectories), feature_matrix(feature_matrix),  ground_r(ground_r),
    n_trajectories(n_trajectories), trajectory_length(trajectory_length), epochs(epochs), discount(discount), learning_rate(learning_rate){
	
	// if(eligibility<1) eligibility=1;
	// ringbuffersize = eligibility+1;
	// actions  = new int[ringbuffersize];
	// states   = new int[ringbuffersize];
	// rewards  = new double[ringbuffersize];
	// longrewards = new double[tau];
	// memset(actions,0,sizeof(int)*ringbuffersize);
	// memset(states,0,sizeof(int)*ringbuffersize);
	// memset(rewards,0,sizeof(double)*ringbuffersize);
	// memset(longrewards,0,sizeof(double)*tau);
	t=0;
	// collectedReward = 0;
	// initialised=false;

    //before assigning values to these integers
	NN = -1;
    DD = -1;  // where N is the number of states and D is the dimensionality of the state.
    AA = -1;
    TT = -1;
    LL = -1;

	//initialize all the matrices: 

	//addParameter("n_trajectories",&this->n_trajectories);
	//addParameter("trajectory_length",&this->trajectory_length);
	//addParameter("epochs",&this->epochs);
	//addParameter("discount",&this->discount);
	//addParameter("learning_rate",&this->learning_rate);

	// int NN = feature_matrix.getM();    //number of states
	// int DD = feature_matrix.getN();   //dimensionality of the state

	// //trajectory should be (T,L,2) but here weuse it as (T*2 ,L )
	// int two_tt = trajectories.getM();    //number of states
	// int ll = trajectories.getN();   //dimensionality of the state
	
	// int tt = (int) two_tt/2 ;
	// TT = tt;
	// LL = ll;
	// //T is the number of trajectories and L is the trajectory length.
	
	// int aa_nn = transition_probabilities.getM();    //actions * states
	// int nn = transition_probabilities.getN();       //states

	// //assert(NN == nn);
	// int aa = (int) (aa_nn / nn);
	// AA = aa;

	// std::cout<< "  NN " << NN <<"DD " <<DD <<"AA " <<AA <<"TT " << TT<<"LL "  <<LL <<std::endl;

	// //initialize all the matrix shape:
	// trajectories.set((int)(2*TT),LL);
	// feature_matrix.set(NN, DD);
	// ground_r.set(NN ,1);
	// r.set(NN ,1);
	// transition_probabilities.set((int)(AA*NN), NN);
	// policy.set(NN, AA);  


	// int policy_nn = policy.getM();       //states
	// int policy_aa = policy.getN();       //actions
	// assert(aa == policy_aa);
	// assert(nn == policy_nn);	
	

}

Maxent::~Maxent(){
	// if(actions) delete[] actions;
	// if(states) delete[] states;
	// if(rewards) delete[] rewards;
	// if(longrewards) delete[] longrewards;
};

void Maxent::init( int nn, int dd, int aa , int tt, int ll, RandGen* randGen){
	if(!randGen) randGen = new RandGen(); // this gives a small memory leak
	this->randGen=randGen;
    
	//trajectories = Trajectories;
	//feature_matrix = Feature_matrix;
	//ground_r = Ground_r;

	NN = nn;
	DD = dd;
	AA = aa;
	TT = tt;
	LL = ll;

	// initialize matrices :
	std::cout<< "Inialization--- NN = " << NN <<", DD = " <<DD <<", AA = " <<AA <<", TT = " << TT<<", LL = "  <<LL <<std::endl;

	//initialize all the matrix shape:
	trajectories.set((int)(2*TT),LL);
	feature_matrix.set(NN, DD);
	ground_r.set(NN ,1);
	r.set(NN ,1);
	transition_probabilities.set((int)(AA*NN), NN);
	policy.set(NN, AA);  
	
}



matrix::Matrix Maxent::irl(matrix::Matrix feature_matrix, int n_actions, double discount, 
        matrix::Matrix transition_probability, matrix::Matrix trajectories, int epochs, double learning_rate){
	
	//std::cout <<"Here, here : " <<transition_probability.getM() <<" "<< transition_probability.getN()<<std::endl;
	//std::cout<< "  NN " << NN <<"DD " <<DD <<"AA " <<AA <<"TT " << TT<<"LL "  <<LL <<std::endl;
	
	int n_states = feature_matrix.getM();    //number of states
	int d_states = feature_matrix.getN();   //dimensionality of the state
    
	NN = n_states;
	DD = d_states;

	// Initialise weights.
    //double r = randGen->rand(); //between [0,1)
    matrix::Matrix alpha;
	alpha.set(d_states,1);
	alpha= alpha.mapP(randGen, random_minusone_to_one);   //create random numbers from -1 to 1

	//Calculate the feature expectations \tilde{phi}.
    matrix::Matrix feature_expectations;

	feature_expectations.set(d_states,1);
	feature_expectations = find_feature_expectations(feature_matrix, trajectories);

	//Gradient descent on alpha.
    for(int i=0; i<epochs; i++){
		//
		matrix::Matrix r;
		r.set(n_states,1);
		r = feature_matrix * alpha;

		//this->r = r;  //save the reward function as only output!!

		//r = feature_matrix.dot(alpha)
		matrix::Matrix expected_svf;
		expected_svf.set(n_states,1);
		expected_svf = find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories);
		
		matrix::Matrix grad;
	    grad.set(d_states,1);
		grad = feature_expectations - (feature_matrix ^ matrix::T) * expected_svf;

		alpha += grad * learning_rate;

	}

	matrix::Matrix output;
	output.set(n_states,1);

	output = feature_matrix * alpha;

    return output.reshape(n_states,1);
	
}


matrix::Matrix Maxent::find_svf(int n_states, matrix::Matrix trajectories){

	NN = n_states;

	matrix::Matrix svf;
	svf.set(n_states,1);
	svf.toZero();


    //trajectory should be (T,L,2) but here weuse it as (T*2 ,L )
	int two_tt = trajectories.getM();    //number of states
	int ll = trajectories.getN();   //dimensionality of the state
	
	int tt = (int) two_tt/2 ;
	TT = tt;
	LL = ll;
	//T is the number of trajectories and L is the trajectory length.

	

	matrix::Matrix state_sequence = trajectories;
	//state_sequence.set(tt, ll);
	state_sequence = ((state_sequence.reshape(2, tt*ll)).row(0)).reshape(tt, ll);
	matrix::Matrix action_sequence = trajectories;
	//action_sequence.set(tt, ll);
    action_sequence = ((action_sequence.reshape(2, tt*ll)).row(1)).reshape(tt, ll);

	
	for(int i=0; i<tt; i++){
		for(int j=0; j<ll; j++){
			int current_state;
			assert(i<state_sequence.getM());
			assert(j<state_sequence.getN());
			current_state = (int)(state_sequence.val(i, j));
			assert(current_state<svf.getM());
			
			svf.val(current_state,0) = svf.val(current_state,0) + 1.0;
		}
	}

    svf = svf * (1.0 / (double)tt);  	// svf /= trajectories.shape[0]
	return svf;
}

matrix::Matrix Maxent::find_feature_expectations(matrix::Matrix feature_matrix, matrix::Matrix trajectories){
	int n_states = feature_matrix.getM();    //number of states
	int d_states = feature_matrix.getN();   //dimensionality of the state
    
	NN = n_states;
	DD = d_states;

	//trajectory should be (T,L,2) but here weuse it as (T*2 ,L )
	int two_tt = trajectories.getM();    //number of states
	int ll = trajectories.getN();   //dimensionality of the state
	
	int tt = (int) two_tt/2 ;
	TT = tt;
	LL = ll;
	//T is the number of trajectories and L is the trajectory length.

	// matrix::Matrix state_sequence;
	// state_sequence.set(tt, ll);
	// state_sequence = ((trajectories.reshape(2, tt*ll)).row(0)).reshape(tt, ll);
	// matrix::Matrix action_sequence;
	// action_sequence.set(tt, ll);
    // action_sequence = ((trajectories.reshape(2, tt*ll)).row(1)).reshape(tt, ll);
	matrix::Matrix state_sequence = trajectories;
	state_sequence = ((state_sequence.reshape(2, tt*ll)).row(0)).reshape(tt, ll);
	matrix::Matrix action_sequence = trajectories;
    action_sequence = ((action_sequence.reshape(2, tt*ll)).row(1)).reshape(tt, ll);

    //the feature expectations \tilde{phi}.
    matrix::Matrix feature_expectations;
	feature_expectations.set(d_states,1);
	
	for(int i=0; i<tt; i++){
		for(int j=0; j<ll; j++){
			int current_state;
			
			assert(i<state_sequence.getM());
			assert(j<state_sequence.getN());

			current_state = (int) (state_sequence.val(i, j));

			matrix::Matrix feature_matrix_copy = feature_matrix;
			feature_expectations = feature_expectations + (feature_matrix_copy.row(current_state)).reshape(d_states,1);
			
		}
	}

	feature_expectations = feature_expectations * (1.0 / (double) tt );
	return feature_expectations;

}



//------------value iterations--------------------------
//transition_probability: NumPy array mapping (state_i, action, state_k) to
//the probability of transitioning from state_i to state_k under action Shape (N, A, N).
// but here shape is (A * N, N) matrix;

//policy: -> NumPy array of states and the probability of taking each action in that
//state, with shape (N, A).

//reward: Vector of rewards for each state

matrix::Matrix Maxent::value_func(matrix::Matrix policy, int n_states, matrix::Matrix transition_probabilities, 
                    matrix::Matrix reward, double discount, double threshold){
	
	matrix::Matrix v;
	v.set(n_states,1);
	v.toZero();

	int aa_nn = transition_probabilities.getM();    //actions * states
	int nn = transition_probabilities.getN();       //states

	NN = nn;
	int aa = (int) (aa_nn / nn);
	AA = aa;

	matrix::Matrix tranprob;
	tranprob.set(aa, nn*nn);

	matrix::Matrix transition_probabilities_copy = transition_probabilities;
	tranprob = transition_probabilities_copy.reshape(aa, nn*nn);

	int policy_nn = policy.getM();       //states
	int policy_aa = policy.getN();       //actions
	assert(aa == policy_aa);
	assert(nn == policy_nn);

	float diff = (float) 1000000.0;
	while (diff > threshold) {
		diff = 0.0;
		for(int s=0; s<n_states; s++){
			assert(s< v.getM());
			double vs = v.val(s,0);
			//find the action based on the deterministic policy (max)
			//first turn the policy to deterministic policy
			matrix::Matrix as = policy.row(s);
			int a= argmax(as);  //deterministic action based on the argmax probability

			matrix::Matrix tranprob_a;
			tranprob_a.set(nn,nn);
			tranprob_a = (tranprob.row(a)).reshape(nn,nn);

			//the following sum: v[s] = sum(transition_probabilities[s, a, k] *(reward[k] + discount * v[k]) for k in range(n_states))
			double sum = 0.0;

			for(int k=0; k<n_states; k++){
				// now we know it is transition from state s to state k based on action a:
				assert(s<tranprob_a.getM());
				assert(k<tranprob_a.getN());
				sum = sum + tranprob_a.val(s,k) * (reward.val(k,0) + v.val(k,0) * discount ) ;
			}
            
			assert(s<v.getM());
			v.val(s,0) = sum;
			//end update for values (iterations).

			diff = std::max( (double)diff, (double) (std::abs((double)vs - (double)(v.val(s,0))) ) );
			
		}

	}

	return v;

}


matrix::Matrix Maxent::optimal_value(int n_states, int n_actions, matrix::Matrix transition_probabilities, 
                  matrix::Matrix reward, double discount, double threshold){
	
	matrix::Matrix v;
	v.set(n_states,1);
	v.toZero();

	int aa_nn = transition_probabilities.getM();    //actions * states
	int nn = transition_probabilities.getN();       //states

	NN = nn;
	int aa = (int) (aa_nn / nn);
	AA = aa;

	matrix::Matrix tranprob;
	tranprob.set(aa, nn*nn);
	tranprob = transition_probabilities;
	tranprob = tranprob.reshape(aa, nn*nn);

	float diff = (float) 1000000.0;
	while (diff > threshold) {
		diff = 0.0;
		for(int s=0; s<n_states; s++){
			double max_v = (float) -1000000.0;
			for(int a=0; a<aa; a++){
				matrix::Matrix tp;
				tp.set(nn, 1);
				
				matrix::Matrix tranprob_a;
			    tranprob_a.set(nn,nn);
			    tranprob_a = (tranprob.row(a)).reshape(nn,nn);
				
				tp =  (tranprob_a.row(s)).reshape(nn,1);

				matrix::Matrix output; 
				output.set(1,1);
				output = (tp ^ matrix::T) * (reward + v * discount);
				assert(output.getM()==1);
				assert(output.getN()==1);

				max_v = std::max(max_v, output.val(0,0) );
			}
            
			assert(s< v.getM());
			double new_diff = std::abs(v.val(s,0) - max_v);
			if(new_diff > diff){
				diff = new_diff;
			}
			v.val(s,0) = max_v;

		}

	}

	return v;
}




matrix::Matrix Maxent::find_policy(int n_states, int n_actions, matrix::Matrix transition_probabilities,
                matrix::Matrix reward, double discount, double threshold, bool stochastic){

	matrix::Matrix v;
	v.set(n_states,1);
    
	//std::cout<<"Debug:" << transition_probabilities.getM()<< "  " << transition_probabilities.getN()<<std::endl;
	int aa_nn = transition_probabilities.getM();    //actions * states
	int nn = transition_probabilities.getN();       //states

	NN = nn;
	int aa = (int) (aa_nn / nn);
	AA = aa;

	//std::cout<<"Debug:" << aa<<" "<< n_actions << nn<<" "<< n_states<<std::endl;
    

	assert(nn == n_states );
	assert(aa == n_actions );

	matrix::Matrix transition_probabilities_copy;
	transition_probabilities_copy.set(aa_nn, nn);
	transition_probabilities_copy = transition_probabilities;

	matrix::Matrix tranprob;
	tranprob.set(aa, nn*nn);
	tranprob = transition_probabilities_copy.reshape(aa, nn*nn);
	
	v = optimal_value(n_states, n_actions, transition_probabilities, reward, discount, threshold);
	
	//Get Q using equation 9.2 from Ziebart's thesis.
	matrix::Matrix Q;
	Q.set(n_states, n_actions);
	Q.toZero();
	
	if(stochastic==true){

		for(int i=0; i<n_states; i++){
			for(int j=0; j<n_actions; j++){
				matrix::Matrix p;
				p.set(n_states,1);
				
				matrix::Matrix tranprob_a;
			    tranprob_a.set(nn,nn);
			    tranprob_a = (tranprob.row(j)).reshape(nn,nn);
				
				p =  (tranprob_a.row(i)).reshape(nn,1);

				assert(i<Q.getM());
				assert(j<Q.getN());
				Q.val(i,j) = ( (p ^ matrix::T) * (reward + v * discount) ).val(0,0);

			}
		}

		//Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
		//[2 2 3], [1 1 2]] then max is [3 2] , subtract by [[3], [2]] ,broadcast then result [[-1 -1  0], [-1 -1  0]]

		for(int i=0; i<n_states; i++){
			matrix::Matrix qq;
			qq.set(1, n_actions);
			qq = Q.row(i);

			for(int j=0; j<n_actions; j++){
				assert(i<Q.getM());
				assert(j<Q.getN());
				Q.val(i,j) = Q.val(i,j) - max(qq);  // max function from the controller.misc.h (also argmax)
			}
		}

		matrix::Matrix Q_exp;
		Q_exp.set(n_states, n_actions);
		for(int i=0; i<n_states; i++){
			for(int j=0; j<n_actions; j++){
				assert(i<Q.getM());
				assert(j<Q.getN());
				assert(i<Q_exp.getM());
				assert(j<Q_exp.getN());
				Q_exp.val(i,j) = std::exp(Q.val(i,j));
			}
		}

		for(int i=0; i<n_states; i++){
			matrix::Matrix qqq;
			qqq.set(1, n_actions);
			qqq = Q_exp.row(i);

			double sum = 0.0;
			for(int j=0; j<n_actions; j++){
				
				assert(j<qqq.getN());
				sum = sum + qqq.val(0,j); 
			}

			for(int j=0; j<n_actions; j++){
				assert(i<Q.getM());
				assert(j<Q.getN());
				assert(i<Q_exp.getM());
				assert(j<Q_exp.getN());
				Q.val(i,j) = Q_exp.val(i,j) / sum ;  // max function from the controller.misc.h (also argmax)
			}
		}



	}

	return Q;


}




//policy: -> NumPy array of states and the probability of taking each action in that
//state, with shape (N, A).

matrix::Matrix Maxent::find_expected_svf(int n_states, matrix::Matrix r, int n_actions, double discount,
        matrix::Matrix transition_probabilities, matrix::Matrix trajectories){

	//std::cout<<"Debug: first level: "<<transition_probabilities.getM()<<transition_probabilities.getN()  <<std::endl;

	int aa_nn = transition_probabilities.getM();    //actions * states
	int nn = transition_probabilities.getN();       //states

	NN = nn;
	int aa = (int) (aa_nn / nn);
	AA = aa;

	assert(nn == n_states );
	assert(aa == n_actions );

    matrix::Matrix transition_probabilities_copy;
	transition_probabilities_copy.set(aa_nn, nn);
	transition_probabilities_copy = transition_probabilities;

	matrix::Matrix tranprob;
	tranprob.set(aa, nn*nn);
	tranprob = transition_probabilities_copy.reshape(aa, nn*nn);

	//trajectory should be (T,L,2) but here weuse it as (T*2 ,L )
	int two_tt = trajectories.getM();    //number of states
	int ll = trajectories.getN();   //dimensionality of the state
	
	int tt = (int) two_tt/2 ;
	TT = tt;
	LL = ll;
	//T is the number of trajectories and L is the trajectory length.


	// matrix::Matrix state_sequence;
	// state_sequence.set(tt, ll);
	// state_sequence = ((trajectories.reshape(2, tt*ll)).row(0)).reshape(tt, ll);
	// matrix::Matrix action_sequence;
	// action_sequence.set(tt, ll);
    // action_sequence = ((trajectories.reshape(2, tt*ll)).row(1)).reshape(tt, ll);
	matrix::Matrix state_sequence = trajectories;
	state_sequence = ((state_sequence.reshape(2, tt*ll)).row(0)).reshape(tt, ll);
	matrix::Matrix action_sequence = trajectories;
	action_sequence = ((action_sequence.reshape(2, tt*ll)).row(1)).reshape(tt, ll);

	// int current_state;
	// current_state = (int)state_sequence.val(i, j);
	
	//reshape trajectory
	// matrix::Matrix jj;
	// jj.set(2,tt*ll);
	// jj = trajectories.reshape(2,tt*ll);
	// matrix::Matrix traj;
	// traj.set(tt,ll);
	// //the third dimension is state! see below(0,0), ,eans initial state, initial sequence
	// traj = jj.row(0).reshape(tt,ll);
	
	//traj is the state_sequence!
    matrix::Matrix policy;
	policy.set(n_states, n_actions);
	policy = find_policy(n_states, n_actions, transition_probabilities, r, discount);

	matrix::Matrix start_state_count;
	start_state_count.set(n_states, 1);

	for(int i=0; i<tt; i++){
		//int sss = (int) traj.val(i,0);  //notice here the trajectory should all be integer??--------
		//start_state_count.val(sss,0) = start_state_count.val(sss,0) +1;
		
		int initial_state;
		assert(i <state_sequence.getM());
		initial_state = (int) (state_sequence.val(i, 0));
		assert(initial_state < start_state_count.getM());
		start_state_count.val(initial_state,0) = start_state_count.val(initial_state,0) +1;

	}

	//p_start_state = start_state_count/n_trajectories
	matrix::Matrix p_start_state;
	p_start_state.set(n_states, 1);
	
	for(int i=0; i<n_states; i++){
		assert(i< p_start_state.getM());
		assert(i< start_state_count.getM());
		p_start_state.val(i,0) = ( (double) (start_state_count.val(i,0))) / (double)tt;
	}

    //expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
	matrix::Matrix expected_svf_before;
	expected_svf_before.set(n_states, ll);            //enlarge another dimension (to 2D) by trajectory length
	for(int i=0; i<n_states; i++){
		for(int j=0; j<ll; j++){
			assert(i<expected_svf_before.getM());
			assert(j<expected_svf_before.getN());
			assert(i<p_start_state.getM());
			expected_svf_before.val(i, j) = p_start_state.val(i, 0);
		}
	}

	for(int t=1; t<ll; t++){
		for(int e=0; e<n_states; e++){
			assert(e< expected_svf_before.getM());
			assert(t<expected_svf_before.getN());
			expected_svf_before.val(e, t) = 0.0;
		}
		for(int i=0; i<n_states; i++){
			for(int j=0; j<n_actions; j++){

				matrix::Matrix tranprob_a;
				tranprob_a.set(nn,nn);
				tranprob_a = (tranprob.row(j)).reshape(nn,nn);

				for(int k=0; k<n_states; k++){
					assert(k< expected_svf_before.getM());
					assert(t<expected_svf_before.getN());
					assert(i< expected_svf_before.getM());
					assert(t-1<expected_svf_before.getN());
					assert(i<policy.getM());
					assert(j<policy.getN());
					assert(i< tranprob_a.getM());
					assert(k < tranprob_a.getN());

					expected_svf_before.val(k,t) = expected_svf_before.val(k,t) + ( expected_svf_before.val(i, t-1) * policy.val(i, j) * tranprob_a.val(i, k) );
				}
			}
		}

	}

	matrix::Matrix expected_svf;
	expected_svf.set(n_states,1);

	for(int i=0; i<n_states; i++){
		double new_sum = 0.0;
		for(int j=0; j<ll; j++){
			assert(i< expected_svf_before.getM());
			assert(j< expected_svf_before.getN());
			new_sum = new_sum + expected_svf_before.val(i,j);
		}
		assert(i<expected_svf.getM());
		expected_svf.val(i,0) = new_sum;		
	}

	return expected_svf;

}



double Maxent::softmax(double x1, double x2){

	double max_x = std::max(x1, x2);
	double min_x = std::min(x1, x2);

	return max_x + std::log(1.0 + std::exp(min_x - max_x));
}



//policy: -> NumPy array of states and the probability of taking each action in that
//state, with shape (N, A).
//optimal_value: Value vector for the ground reward with optimal policy.
//The ith component is the value of the ith state. Shape (N,).
//p_start_state: Probability vector with the ith component as the probability
//that the ith state is the start state. Shape (N,).

//reward: Reward vector mapping state int to reward. Shape (N,).
//true_reward: True reward vector. Shape (N,).

double Maxent::expected_value_difference(int n_states, int n_actions, matrix::Matrix transition_probability,
        matrix::Matrix reward, double discount, matrix::Matrix p_start_state, matrix::Matrix optimal_value, 
        matrix::Matrix true_reward){
	
	matrix::Matrix policy;
	policy.set(n_states, n_actions);
	policy = find_policy(n_states, n_actions,transition_probability, reward, discount);
	
	matrix::Matrix value;
	value.set(n_states,1);
	
	matrix::Matrix argmax_policy;
	argmax_policy.set(n_states,1);

    //policy.argmax(axis=1)
	for(int i=0; i<n_states; i++){
		matrix::Matrix ap;
		ap.set(1, n_actions);
		ap = policy.row(i);
		assert(i<argmax_policy.getM());
		argmax_policy.val(i,0) = argmax(ap);
	}


	value = value_func(argmax_policy, n_states, transition_probability, true_reward, discount);

	
	double evd = 0.0;
	
	evd = ( (optimal_value ^ matrix::T) * p_start_state - (value ^ matrix::T) * p_start_state ).val(0,0);

	return evd;



}














bool Maxent::store(FILE* f) const{
  r.store(f);
  Configurable::print(f,0);
  return true;
}

bool Maxent::restore(FILE* f) {
  r.restore(f);
  Configurable::parse(f);
  //t=0;
  return true;
}