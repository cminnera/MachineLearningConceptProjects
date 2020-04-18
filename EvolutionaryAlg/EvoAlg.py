#Clare Minnerath

#Evolutionary Algorithm with closed form solution

import numpy as np
import matplotlib.pyplot as plt

class pop:
    
    def __init__(self, size, epoch):
        self.size = size
        self.populate = self.buildpop()
        self.count = 0
        self.mutrate = .078
        
        #Plotting fitnesses:
        
        plt.ylabel("Fitness Level")
        plt.xlabel("Epochs")
        
        while(self.count<epoch):
            self.dec = self.todec(self.populate)
            self.fittest, self.avgfit, self.maxfit = self.fit()
            if(self.avgfit >= .9999):
                print("Final fitness:")
                print("Average = ", self.avgfit)
                print("Max = ", self.maxfit)
                print()
                self.dec = self.todec(self.populate)
                break
            
            self.new = self.reproduce(self.fittest, self.populate)
            self.dec = self.todec(self.new)
            self.fittest, self.avgfit, self.maxfit = self.fit()
            self.greent, self.blues = plt.plot(self.count,self.maxfit, 'g^', self.count, self.avgfit, 'bs')
            if(self.avgfit >= .9999):
                print("Final fitness:")
                print("Average = ", self.avgfit)
                print("Max = ", self.maxfit)
                print()
                self.dec = self.todec(self.new)
                break
            self.final = self.mutate(self.new, self.mutrate)
            self.count += 1
            self.populate = self.final
            self.mutrate = max(self.mutrate - .00005, .001)
        
        if(self.count<=9999):
            print(self.count , "epochs to reach ideal population")
            plt.legend([self.greent, self.blues], ["Max fitness", "Average fitness"])
            plt.show()
        else:
            print("Epoch Max Reached")
            print()
        
    
    def buildpop(self):
        p = np.random.randint(2, size = (self.size, 8))
        return p
    
    
    def todec(self, pop):
        n = np.zeros(self.size)
        count = 0
        for i in pop:
            for j in i:
                n[count] = n[count] * 2 + j
            count += 1
            

        return n;
    
    
    def fit(self):
        d = self.dec
        d[:] = [np.sin(np.pi * x / 256.0) for x in d]
        avg = np.sum(d)/self.size
        Max = np.max(d)
        Sum = np.sum(d)
        d[:] = [x/Sum for x in d]
        n = np.zeros(self.size, dtype=int)
        
        p = np.random.uniform(0,1)
        
        total = 0
        count = 0
        
        for j in range(0,self.size):
            for i in d:
                total += i
                if(p < total):
                    n[j] = count
                    break;
                count+=1
            count = 0
            total = 0
            p = np.random.uniform(0,1)
            
        return n, avg, Max
    
    
    def reproduce(self, f, p):
        new = np.zeros((self.size,8), dtype = int)
        list=[]
        
        for l in range(0,4):
            first = np.random.randint(0,self.size)
            second = np.random.randint(0,self.size)
            
            temp = min(first,second)
            if(first!=temp):
                second = first
                first = temp
            
            p1 = np.random.randint(0,8)
            while p1 in list:
                p1 = np.random.randint(0,8)
            list.append(p1)
            
            p2 = np.random.randint(0,8)
            while p2 in list:
                p2 = np.random.randint(0,8)
            list.append(p2)
            
            binp1 = p[f[p1]]
            binp2 = p[f[p2]]
            
            
            
            
            for i in range(0,first):
                new[2 * l][i] = binp1[i]
                new[(2 * l)+1][i] = binp2[i]
            for j in range(first,second):
                new[2 * l][j] = binp2[j]
                new[(2 * l)+1][j] = binp1[j]
            for k in range(second,self.size):
                new[2 * l][k] = binp1[k]
                new[(2 * l)+1][k] = binp2[k]
                   
                
        return new;
        
    def mutate(self, children, mr):
        for i in range(0, 8):
            for j in range(0,8):
                p = np.random.uniform(0,1)
                if(p<=mr):
                    if(children[i][j]==0):
                        children[i][j]=1
                    else:
                        children[i][j]=0
                                        
        return children
                
            
        

def main():
    np.set_printoptions(threshold=np.nan)
    population = pop(size = 8, epoch = 10000)
    print()
    print("Final population:")
    print(population.dec)
    
    
if __name__=='__main__':
    main()