########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
import pandas as pd

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # Matrix to store probability of transitioning from most likely
        # Previous hidden state to current hidden state then emmiting 
        # Observed state x[i]
        probs = [[0. for _ in range(self.L)] for _ in range(M)]
        # Matrix to store previous hidden state at time i-1 most likely 
        # To have transitioned to current hidden state at time i
        seqs = [['' for _ in range(self.L)] for _ in range(M)]
        
        # Initialize first step
        for s in range(self.L):
            # Probability of transitioning from start state to first observation
            probs[0][s] = self.A_start[s] * self.O[s][x[0]]
            # First observation in sequence
            seqs[0][s] = '0'
            
        # Run through each sequential observation
        for i in range(1, M):
            # Run over all states
            for s in range(self.L):
                # Probability of transitioning from any previous hidden state k 
                # To current hidden state s
                # Then emitting observation x[i]
                p_trans = [probs[i-1][k] * self.A[k][s] * self.O[s][x[i]] for k in range(self.L)]
                # Maximum probability
                max_p = np.max(p_trans)
                probs[i][s] = max_p
                # k-th state that gives the maximum probability
                seqs[i][s] = str(np.where(p_trans == max_p)[0][0])
        
        # Maximum final state
        opt = []
        max_p_end = np.max(probs[M-1])
        opt.append(str(np.where(probs[M-1] == max_p_end)[0][0]))
        
        # Follow maximum backtrack recursively to find path
        for i in range(1, M)[::-1]:
            opt.insert(0, seqs[i][int(opt[0])])
        
        # Maximum probability path string
        max_seq = ''.join(opt)

        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = np.array([[0. for _ in range(self.L)] for _ in range(M)])
        
        # Let O be an actual matrix for easier manipulation
        O_mat = np.array(self.O)
        
        # Initialization
        O_1 = np.diag(O_mat[:, x[0]])
        TO_1 = np.matmul(self.A, O_1)  # transform by transition matrix
        alphas[0, :] = np.matmul(self.A_start, O_1)
        
        # Normalize?
        if normalize:
            if np.sum(alphas[0, :]) > 0:
                alphas[0, :] = alphas[0, :] / np.sum(alphas[0, :])
        
        # Recursively propagate probabilities 
        for t in range(1, M):
            O_t = np.diag(O_mat[:, x[t]])
            TO_t = np.matmul(self.A, O_t)
            alphas[t, :] = np.matmul(alphas[t-1, :], TO_t)
            
            # Normalize?
            if normalize:
                if  np.sum(alphas[t, :]) > 0:
                    alphas[t, :] = alphas[t, :] / np.sum(alphas[t, :])

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = np.array([[0. for _ in range(self.L)] for _ in range(M)])
        
        # Let O be an actual matrix for easier manipulation
        O_mat = np.array(self.O)
        
        # Initialization
        betas[M-1, :] = np.ones(self.L).T
        
        # Normalize?
        if normalize:
            if np.sum(betas[M-1, :]) > 0:
                betas[M-1, :] = betas[M-1, :] / np.sum(betas[M-1, :])
        
        # Recursively propagate probabilities 
        for t in range(1, M)[::-1]:
            O_t = np.diag(O_mat[:, x[t]])
            TO_t = np.matmul(self.A, O_t)
            betas[t-1, :] = np.dot(TO_t, betas[t, :].T)
            
            # Normalize?
            if normalize:
                if np.sum(betas[t-1, :]) > 0:
                    betas[t-1, :] = betas[t-1, :] / np.sum(betas[t-1, :])
        

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas       
        for k in range(len(Y)):
            l = len(Y[k])
            for i in range(1, l):
                # If Y_(i-1) = a and Y_i = b
                # Then increase the count A_(ab) by 1
                self.A[Y[k][i-1]][Y[k][i]] += 1
        
        # Normalize the rows so that A is stochastic
        for i in range(self.L):
                self.A[i] = self.A[i] / np.sum(self.A[i])

        # Calculate each element of O using the M-step formulas
        for k in range(len(X)):
            l = len(X[k])
            for i in range(l):
                # If Y_i = w and X_i = z
                # Then increase the count O_(wz) by 1
                self.O[Y[k][i]][X[k][i]] += 1
        
        # Normalize the rows so that O is stochastic
        for i in range(self.L):
                self.O[i] = self.O[i] / np.sum(self.O[i])
                
        
        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        
        for i in range(N_iters):
            # Initialization
            A_num = np.zeros((self.L, self.L))
            O_num = np.zeros((self.L, self.D))
            
            # Iterate over input sequences
            for x in X:
                # Forward pass for the alpha vector
                alphas = self.forward(x, normalize=True)
                    
                # Backward pass for the beta vector
                betas = self.backward(x, normalize=True)
                
                # Iterate over sequence to update transition matrix        
                for t in range(len(x)-1):
                    A_den = np.zeros((self.L, self.L))
                    for a in range(self.L):
                        for b in range(self.L):
                            A_den[a][b] = alphas[t][a] * betas[t+1][b] * self.A[a][b] * self.O[b][x[t+1]]
                    St = np.sum(A_den.flatten())
                    for a in range(self.L):
                        for b in range(self.L):
                            A_num[a][b] += A_den[a][b] / St
                            
                
                # Iterate over sequence to update emission matrix        
                for t in range(len(x)):
                    # Sum product of alphas and betas for normalization
                    S = np.sum(alphas[t] * betas[t])
                    for z in range(self.L):
                        if S > 0:
                            O_num[z][x[t]] += alphas[t][z] * betas[t][z] / S
                               
            # Normalize and update transition and emission matrices           
            for i in range(self.L):
                if np.sum(A_num[i]) > 0:
                    self.A[i] = A_num[i] / np.sum(A_num[i])
                if np.sum(O_num[i]) > 0:
                    self.O[i] = O_num[i] / np.sum(O_num[i])

        pass


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        # Initial state
        s_0 = np.random.randint(low=0, high=self.L)
        states.append(s_0)
        
        # Initial observation
        p_0 = np.random.rand()
        
        k = 0
        p = 0
        while p <= p_0:
            p += self.O[s_0][k]
            k += 1
        x_0 = k-1 * (k >= 0)
        emission.append(x_0)
           
        
        # Generate sequence of states
        for t in range(1, M):
            # Sample hidden state
            h_t = np.random.rand()
            s_prev = states[t-1]
            
            kh = 0
            ph = 0
            while (ph <= h_t) and (kh < self.L):
                ph += self.O[s_prev][kh]
                kh += 1
            s = kh-1 * (kh >= 0)
            states.append(s)
            
            # Sample emission
            p_t = np.random.rand()
            
            ko = 0
            po = 0
            while (po <= p_t) and (ko < self.D):
                po += self.O[s][ko]
                ko += 1
            x = ko-1 * (ko >= 0)
            emission.append(x)
            

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1, :])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[0][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
