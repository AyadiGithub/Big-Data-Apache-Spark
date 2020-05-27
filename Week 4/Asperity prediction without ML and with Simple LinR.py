#Predicting Asperity (healthiness of a part) without Linear Regression

dp = {'partnumber': 100, 'maxtemp': 35, 'mintemp': 35, 'maxvibration': 12, 'asperity': 0.32}
dp1 = {'partnumber': 101, 'maxtemp': 46, 'mintemp': 35, 'maxvibration': 21, 'asperity': 0.34}
dp2 = {'partnumber': 130, 'maxtemp': 56, 'mintemp': 46, 'maxvibration': 3412, 'asperity': 12.42}
dp3 = {'partnumber': 131, 'maxtemp': 58, 'mintemp': 48, 'maxvibration': 3542, 'asperity': 13.43}

def predict(dp):
    if dp['maxvibration']>100:
        print('unhealthy')
        return 13
    else:
        print('healthy')
        return 0.33

predict(dp1)
predict(dp2)
predict(dp3)


#Predicting Asperity (healthiness of a part) with Linear Regression
#But with manual parameter setting
w1 = 0.30
w2 = 0
w3 = 0
w4 = 13/3412
def mlpredict(dp):
    return w1+w2*dp['maxtemp']+w3*dp['mintemp']+w4*dp['maxvibration']

mlpredict(dp3)
