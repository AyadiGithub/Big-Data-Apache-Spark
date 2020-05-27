#Predicting Asperity (healthiness of a part) without Logistic Regression

dp = {'partnumber': 100, 'maxtemp': 35, 'mintemp': 35, 'maxvibration': 12, 'asperity': 0}
dp1 = {'partnumber': 101, 'maxtemp': 46, 'mintemp': 35, 'maxvibration': 21, 'asperity': 0}
dp2 = {'partnumber': 130, 'maxtemp': 56, 'mintemp': 46, 'maxvibration': 3412, 'asperity': 1}
dp3 = {'partnumber': 131, 'maxtemp': 58, 'mintemp': 48, 'maxvibration': 3542, 'asperity': 1}

def predict(dp):
    if dp['maxvibration']>100:
        print('broken')
        return 1
    else:
        print('not broken')
        return 0

predict(dp1)
predict(dp2)
predict(dp3)


#Predicting Asperity (healthiness of a part) with Logistic Regression
#But with manual parameter setting and a sigmoid computation step
#Importing math and defining sigmoid function
import math 


def sigmoid(x):
  return 1 /(1 + math.exp(-x))

#Manualing setting parameters
w1 = 0.30
w2 = 0
w3 = 0
w4 = 13/3412


def mlpredict(dp):
    if sigmoid(w1+w2*dp['maxtemp']+w3*dp['mintemp']+w4*dp['maxvibration']) > 0.7:
        print('Broken')
        return 1
    else:
        print('Not Broken')
        return 0
    

mlpredict(dp3)
