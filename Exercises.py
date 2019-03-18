import numpy as n



def Ex1():
    arr = n.zeros(10)
    ex1Vec = n.float64(arr)
    print(ex1Vec, "\n" )

def Ex2():
    arr = n.zeros(10)
    ex2Vec = n.float64(arr)
    ex2Vec[4]= 1
    print(ex2Vec,"\n" )

def Ex3():
    ex3Vec = n.arange(10, 49)
    print(ex3Vec,"\n")

def Ex4():
    ex4Mat = n.arange(1,10).reshape(3,3)
    print(ex4Mat,"\n")

def Ex5():
    arr = n.arange(1,10).reshape((3,3))
    ex5Mat = n.flip(arr, 0)
    print(ex5Mat,"\n")

def Ex6():
    arr = n.arange(1,10).reshape((3,3))
    ex6Mat = n.flip(arr, 1)
    print(ex6Mat,"\n")

def Ex7():
    ex7Mat = n.identity(3)
    print(ex7Mat,"\n")

def Ex8():
    ex8Mat = n.random.random((3,3))
    print(ex8Mat,"\n")

def Ex9():
    ex9Mat = n.random.randint(0, 100, 10)
    print(ex9Mat,"and the mean value is:",ex9Mat.mean(), "\n")

def Ex10():
    # ex10 = n.zeros(100).reshape(10,10)
    # ex10[:,0] = 1
    # ex10[0,:] = 1
    # ex10[9,:] = 1
    # ex10[:,9] = 1
# improved

    ex10 = n.ones((10,10))
    ex10[1:9,1:9] = 0

    print(ex10,"\n")

def Ex11():
    # ex11 = n.zeros(25).reshape(5,5)
    # ex11[0, :] = 1
    # ex11[1, :] = 2
    # ex11[2, :] = 3
    # ex11[3, :] = 4
    # ex11[4, :] = 5

    ex11 = n.zeros((5,5))
    ex11 += n.arange(1,6)

    print(ex11,"\n")

def Ex12():
    ex12 = n.float64(n.random.randint(0,10,9).reshape(3,3))
    print(ex12,"\n")

def Ex13():
    # ex13 = n.random.randint(0,10,25)
    # mean = ex13.mean()
    # for i in range(0,25):
    #     ex13[i] = ex13[i]-mean
    # ex13.reshape(5,5)
    #
    ex13 = n.random.random((5,5))
    mean = ex13.mean()
    ex13-mean
    print(ex13,"\n")

def Ex14():
    # ex14 = n.random.random((5,5))
    # avrg = ex14.mean(1)
    # ex14 = ex14 - avrg

    ex14 = n.random.random((5, 5))
    col = ex14.mean(axis=1, keepdims=True)
    ex14 = ex14 - col

    print(ex14,  "\n")

def Ex15():
    mat = n.random.random((5,5))
    ex15 = n.abs(mat-0.5).argmin()
    print("val: {}".format(mat.flat[ex15]))

def Ex16():
    ex16 = n.random.random((3,3))
    print(n.count_nonzero(ex16 > 5.0))

if __name__ == "__main__":
    Ex1()
    Ex2()
    Ex3()
    Ex4()
    Ex5()
    Ex6()
    Ex7()
    Ex8()
    Ex9()
    Ex10()
    Ex11()
    Ex12()
    Ex13()
    Ex14()
    Ex15()
    Ex16()