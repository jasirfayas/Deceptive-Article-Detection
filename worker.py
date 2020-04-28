import joblib
import os

script_dir = os.path.dirname(__file__)

body_model_path1 = os.path.join(script_dir,r'static',r'body_nb.pkl')
body_model_path2 = os.path.join(script_dir,r'static',r'body_knn.pkl')
body_model_path3 = os.path.join(script_dir,r'static',r'body_rf.pkl')
body_model_path4 = os.path.join(script_dir,r'static',r'body_lgr.pkl')

headline_model_path1 = os.path.join(script_dir,r'static',r'head_nb.pkl')
headline_model_path2 = os.path.join(script_dir,r'static',r'head_knn.pkl')
headline_model_path3 = os.path.join(script_dir,r'static',r'head_rf.pkl')
headline_model_path4 = os.path.join(script_dir,r'static',r'head_lgr.pkl')

model_1 = joblib.load(body_model_path1)
model_2 = joblib.load(body_model_path2)
model_3 = joblib.load(body_model_path3)
model_4 = joblib.load(body_model_path4)

model_5 = joblib.load(headline_model_path1)
model_6 = joblib.load(headline_model_path2)
model_7 = joblib.load(headline_model_path3)
model_8 = joblib.load(headline_model_path4)

classifier = { 1: 'Naive Bayes', 2: 'K Neareast Neighbor', 3: 'Random Forest', 4: 'Logistic Regression' }
message = { 1: "3 or more models agreed on true/not-deceptive", 2: "3 or more models agreed on fake/deceptive", 3: "Models didn't agree on anything" } 

def hybrid(predcent1,predcent2,predcent3,predcent4):
    flag_true = 0
    flag_false = 0

    if(predcent1 < 40):
        flag_true = flag_true + 1
    elif(predcent1 > 60):
        flag_false = flag_false + 1

    if(predcent2 < 40):
        flag_true = flag_true + 1
    elif(predcent2 > 60):
        flag_false = flag_false + 1

    if(predcent3 < 40):
        flag_true = flag_true + 1
    elif(predcent3 > 60):
        flag_false = flag_false + 1

    if(predcent4 < 40):
        flag_true = flag_true + 1
    elif(predcent4 > 60):
        flag_false = flag_false + 1

    if(flag_true >= 3):
        msg_op = 1
    elif(flag_false >= 3):
        msg_op = 2
    else:
        msg_op = 3    
    return msg_op    

def calc_mean(decision,pred_score_nb,pred_score_knn,pred_score_rf,pred_score_lgr):
    if(decision == 1):
        if((pred_score_nb[1]*100)>=40):
            return ((pred_score_knn[1]+pred_score_rf[1]+pred_score_lgr[1])/3)
        elif((pred_score_knn[1]*100)>=40):
            return ((pred_score_nb[1]+pred_score_rf[1]+pred_score_lgr[1])/3)
        elif((pred_score_rf[1]*100)>=40):
            return ((pred_score_nb[1]+pred_score_knn[1]+pred_score_lgr[1])/3)
        elif((pred_score_lgr[1]*100)>=40):
            return ((pred_score_nb[1]+pred_score_knn[1]+pred_score_rf[1])/3)        
        else:
            return ((pred_score_nb[1]+pred_score_knn[1]+pred_score_rf[1]+pred_score_lgr[1])/4)
    elif(decision == 2):
        if((pred_score_nb[1]*100)<=60):
            return ((pred_score_knn[1]+pred_score_rf[1]+pred_score_lgr[1])/3)
        elif((pred_score_knn[1]*100)<=60):
            return ((pred_score_nb[1]+pred_score_rf[1]+pred_score_lgr[1])/3)
        elif((pred_score_rf[1]*100)<=60):
            return ((pred_score_nb[1]+pred_score_knn[1]+pred_score_lgr[1])/3)
        elif((pred_score_lgr[1]*100)<=60):
            return ((pred_score_nb[1]+pred_score_knn[1]+pred_score_rf[1])/3)        
        else:
            return ((pred_score_nb[1]+pred_score_knn[1]+pred_score_rf[1]+pred_score_lgr[1])/4)
    else:
            return 0.5       
        
def body_result(msg_op,mean):
    if ((mean*100 >= 40) & ((mean*100 <= 60))):
        tag = 'Not sure'
        color = 'is-warning'
    elif (mean*100 > 60):
        tag = 'This article body is fake/deceptive'
        color = 'is-danger'
    else:
        tag = 'This article body is true/not-deceptive'
        color = 'is-primary'
    prob = {'B-deceptive': round(mean,3), 'tag': tag, 'color':color, 'decision': message[msg_op] }
    return prob

def body_scoring(article):

    pred_score_nb = model_1.predict_proba([article])[0]
    pred_score_knn = model_2.predict_proba([article])[0]
    pred_score_rf = model_3.predict_proba([article])[0]
    pred_score_lgr = model_4.predict_proba([article])[0]   

    pred_cent_nb = pred_score_nb[1]*100
    pred_cent_knn = pred_score_knn[1]*100
    pred_cent_rf = pred_score_rf[1]*100
    pred_cent_lgr = pred_score_lgr[1]*100

    decision=hybrid(pred_cent_nb,pred_cent_knn,pred_cent_rf,pred_cent_lgr)
    mean = calc_mean(decision,pred_score_nb,pred_score_knn,pred_score_rf,pred_score_lgr)

    return body_result(decision,mean)

def headline_result(msg_op,mean):
    if ((mean*100 >= 40) & ((mean*100 <= 60))):
        tag = 'Not sure'
        color = 'is-warning'
    elif (mean*100 > 60):
        tag = 'This headline is fake/deceptive'
        color = 'is-danger'
    else:
        tag = 'This headline is true/not-deceptive'
        color = 'is-primary'
    prob = {'H-deceptive': round(mean,3), 'tag': tag, 'color':color, 'decision': message[msg_op] }
    return prob

def headline_scoring(headline):

    pred_score_nb = model_5.predict_proba([headline])[0]
    pred_score_knn = model_6.predict_proba([headline])[0]
    pred_score_rf = model_7.predict_proba([headline])[0]
    pred_score_lgr = model_8.predict_proba([headline])[0]  

    pred_cent_nb = pred_score_nb[1]*100
    pred_cent_knn = pred_score_knn[1]*100
    pred_cent_rf = pred_score_rf[1]*100
    pred_cent_lgr = pred_score_lgr[1]*100

    decision=hybrid(pred_cent_nb,pred_cent_knn,pred_cent_rf,pred_cent_lgr)
    mean = calc_mean(decision,pred_score_nb,pred_score_knn,pred_score_rf,pred_score_lgr)

    return headline_result(decision,mean)

if __name__ == "__main__":
    result = scoring('')
    result2 = scoring('')
    print(result,result2)