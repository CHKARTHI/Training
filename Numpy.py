# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''
y = 20
type(y)

z1 = "apple"
# z1

'''
z2 = True
type(z2)

x=10
y=20
x + y
x-y
x*y

x=24
y=10
x//y
x%y
x**2

x==y
print("check the condition is true / false:", x==y)

x!=y
x==y
x>y
x>y

x=10
y=20

x == 10 and y==20

age=10
gender=20

print("age and gende: ",age==11 and gender==10)

print("age and gende: ",age==10 or gender==10)

print("age and gende: ",not age==10 or not gender==20)

"apple"

x==10
l=[]
type(l)

l1=[10,20,20,40,50]

l1[0]
l1[0]=200

l1[0]
len(l1)
sum(l1)

l1.pop(1)
l=[4]
type[4]


l1.insert(1,50)

t1=(10,20,30)
type(t1)

# disctionary : keys, values
d1={"male":1,"female":2}
d1
d1.keys()


d2={"averagecallsattennded":"avg"}
d2.values()

# control structures

#if &  else

age=21
if age > 22:
    print("qualifie")
else:
    print("ok")
    
    #else if
    
age=21
gender="male"

if age>18 and gender == "male":
    print("ok")
elif (age>18 and gender =="female"):
    print("exception") 
else:
    print("not ok")


#nestedif

age=21
gender="male"

if age>21:
    if gender == "male":
        print("its fine for now")
    else:
        print("not qualified only")
else:
    if gender == "female":
        print("women NA")
    else:
        print("NA at any case")
        
        
age=22
gender="female"
native="indian"

if age==22:
    if gender == "female" and native=="indian":
        print("its fine for now")
    else:
        print("not qualified only")
else: 
    print("no")
    
#functions
    
def f1(x):
    z=x**2
    return z

#calling the function
f1(2)


def f1(x,y):
    z=(x+y)/2
    return z

f1(2,3)
     
#create a copy 


#created on sun age 7 14:15:15 2022

#pip install numpy
import numpy as np

#create a array 

x1 = np.array([101,102,103])
x1
x1=np.array([[101],[102]])
x1
x1.shape
x2=np.array([[101,20],[102,24]])
x2
x2.shape
x1=np.array([[101,20,1],[102,23,2]])
x1
x1.shape


#
x3 = np.random.randint(10,20,size=100)
x3

x3 = np.random.randint(10,30,size=(2,3,4,5))
x3.shape
x3.ndim
x3

x3 = np.random.randint(10,30,size=(2,3,4,5))
x3


x3[0,]
x3[1,]
x3[2,]

x3[1:3,]  #row and colummns,last number rows columns excluded
x3[0:2,2:4]


###############

x3 = np.random.randint(10,30,size=6)
x3
np.min(x3)
np.max(x3)
np.mean(x3)
np.var(x3)
np.std(x3)
np.var(x3)
np.percentile(x3,25)
np.percentile(x3,50)
np.sqrt(35)

d1={"ok":1}
d1.keys()

######## Panndas

import pandas as pd

#case 1
df=pd.read_csv("E:\\nyc_weather.csv")

#case 2
df=pd.read_csv("E:/nyc_weather.csv")

df=[1,2,3]
df























































