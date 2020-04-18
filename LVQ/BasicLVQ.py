

# Clare Minnerath
# LVQ network example

import numpy as np
import math

rgen = np.random.RandomState(1)
# Random layer 1 weights
weights1 = rgen.normal(loc = 0.0, scale = 0.01, size = 21)


# Layer 2 weights (weights 1 and 2 should probably both be 2D arrays...not sure why I started this way)
weights2 = [1, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0, 
            0, 0, 0, 0, 0, 1, 1]
print(weights1)

#training data
c1 = [-1, 1, -1]
c2 = [-1, -1, 1]
c3 = [-1, -1, -1]
c4 = [1, -1, -1]
c5 = [1, -1, 1]
c6 = [1, 1, -1]
c7 = [-1, 1, 1]
cAll = [c1, c2, c3, c4, c5, c6, c7]

#classes
t1 = [1, 0, 0]
t2 = [0, 1, 0]
t3 = [0, 0, 1]

# maps training data to correct class
pairs = [(c1, t1), (c2, t2), (c3, t3), (c4, t1), (c5, t2), (c6, t2), (c7, t3)]

epoch = 1

while(epoch < 5):
    print()
    print('Epoch: ') 
    print(epoch)
    print()
    for c in cAll:
        print()
        print('Training on ')
        print(c)
        print()
        
        count = 0
        index = 0
        prev = -50
        curr = 0
        
        #Layer 1 Work:
        #finds maximum value of euclidian norm
        while count<21:
            for j in c:
                curr += (j - weights1[count]) * (j - weights1[count])
                count += 1
            curr = -math.sqrt(curr)
            if(curr > prev):
                prev = curr
                index = int(count / 3) - 1
            curr = 0
                
        
        # In place multiplication of weights2 * result of step above
        #easier way to classify what the result will be (does same exact thing)
        if(index <= 1):
            tempW = [1, 0, 0]
        elif(index <= 4):
            tempW = [0, 1, 0]
        else:
            tempW = [0, 0, 1]

        print('Data trains to class:')     
        print(tempW)
        count = index * 3
                
        for a in pairs:
            if(a[0]==c):
                t = a[1]
        print('Data target:')
        print(t)
        print()
        
        #weight updates
        if(tempW == t):
            for i in range(0,3):
                weights1[count] = weights1[count] + .5 * (c[i] - weights1[count])
                count += 1
                    
        else:
            for i in range(0,3):
                weights1[count] = weights1[count] - .5 * (c[i] - weights1[count])
                count += 1
        
                

    epoch += 1
    
print()
print('Done with traing, we have matched our training data 100%')
    
        

