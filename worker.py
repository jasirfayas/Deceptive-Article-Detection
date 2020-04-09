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
        
def model(mean_article0,msg_op,mean_article):
    if ((mean_article*100 >= 40) & ((mean_article*100 <= 60))):
        tag = 'Not sure'
        color = 'is-warning'
    elif (mean_article*100 > 60):
        tag = 'This article body is fake/deceptive'
        color = 'is-danger'
    else:
        tag = 'This article body is true/not-deceptive'
        color = 'is-primary'
    prob = {'P-deceptive': round(mean_article,3), 'P-notDeceptive': round(mean_article0,3), 'tag': tag, 'color':color, 'decision': message[msg_op] }
    return prob

def scoring(article):

    pred_score1 = model_1.predict_proba([article])[0]
    pred_score2 = model_2.predict_proba([article])[0]
    pred_score3 = model_3.predict_proba([article])[0]
    pred_score4 = model_4.predict_proba([article])[0]   

    pred_cent1 = pred_score1[1]*100
    pred_cent2 = pred_score2[1]*100
    pred_cent3 = pred_score3[1]*100
    pred_cent4 = pred_score4[1]*100

    decision=hybrid(pred_cent1,pred_cent2,pred_cent3,pred_cent4)

    if(decision == 1):
        if(pred_cent1>=40):
            mean_article=((pred_score2[1]+pred_score3[1]+pred_score4[1])/3)
        elif(pred_cent2>=40):
            mean_article=((pred_score1[1]+pred_score3[1]+pred_score4[1])/3)
        elif(pred_cent3>=40):
            mean_article=((pred_score1[1]+pred_score2[1]+pred_score4[1])/3)
        elif(pred_cent4>=40):
            mean_article=((pred_score1[1]+pred_score2[1]+pred_score3[1])/3)        
        else:
            mean_article=((pred_score1[1]+pred_score2[1]+pred_score3[1]+pred_score4[1])/4)
    elif(decision == 2):
        if(pred_cent1<=60):
            mean_article=((pred_score2[1]+pred_score3[1]+pred_score4[1])/3)
        elif(pred_cent2<=60):
            mean_article=((pred_score1[1]+pred_score3[1]+pred_score4[1])/3)
        elif(pred_cent3<=60):
            mean_article=((pred_score1[1]+pred_score2[1]+pred_score4[1])/3)
        elif(pred_cent4<=60):
            mean_article=((pred_score1[1]+pred_score2[1]+pred_score3[1])/3)        
        else:
            mean_article=((pred_score1[1]+pred_score2[1]+pred_score3[1]+pred_score4[1])/4)
    else:
            mean_article=0.5       

    mean_article0=((pred_score1[0]+pred_score2[0]+pred_score3[0]+pred_score4[0])/4)

    return model(mean_article0,decision,mean_article)

def model2(mean_head0,msg_op,mean_head):
    if ((mean_head*100 >= 40) & ((mean_head*100 <= 60))):
        tag = 'Not sure'
        color = 'is-warning'
    elif (mean_head*100 > 60):
        tag = 'This headline is fake/deceptive'
        color = 'is-danger'
    else:
        tag = 'This headline is true/not-deceptive'
        color = 'is-primary'
    prob = {'C-deceptive': round(mean_head,3), 'C-notDeceptive': round(mean_head0,3), 'tag': tag, 'color':color, 'decision': message[msg_op] }
    return prob

def scoring2(headline):

    pred_score5 = model_5.predict_proba([headline])[0]
    pred_score6 = model_6.predict_proba([headline])[0]
    pred_score7 = model_7.predict_proba([headline])[0]
    pred_score8 = model_8.predict_proba([headline])[0]  

    pred_cent5 = pred_score5[1]*100
    pred_cent6 = pred_score6[1]*100
    pred_cent7 = pred_score7[1]*100
    pred_cent8 = pred_score8[1]*100

    decision2=hybrid(pred_cent5,pred_cent6,pred_cent7,pred_cent8)

    if(decision2 == 1):
        if(pred_cent5>=40):
            mean_head=((pred_score6[1]+pred_score7[1]+pred_score8[1])/3)
        elif(pred_cent6>=40):
            mean_head=((pred_score5[1]+pred_score7[1]+pred_score8[1])/3)
        elif(pred_cent7>=40):
            mean_head=((pred_score5[1]+pred_score6[1]+pred_score8[1])/3)
        elif(pred_cent8>=40):
            mean_head=((pred_score5[1]+pred_score6[1]+pred_score7[1])/3)        
        else:
            mean_head=((pred_score5[1]+pred_score6[1]+pred_score7[1]+pred_score8[1])/4)
    elif(decision2 == 2):
        if(pred_cent5<=60):
            mean_head=((pred_score6[1]+pred_score7[1]+pred_score8[1])/3)
        elif(pred_cent6<=60):
            mean_head=((pred_score5[1]+pred_score7[1]+pred_score8[1])/3)
        elif(pred_cent7<=60):
            mean_head=((pred_score5[1]+pred_score6[1]+pred_score8[1])/3)
        elif(pred_cent8<=60):
            mean_head=((pred_score5[1]+pred_score6[1]+pred_score7[1])/3)        
        else:
            mean_head=((pred_score5[1]+pred_score6[1]+pred_score7[1]+pred_score8[1])/4)
    else:
            mean_head=0.5   

    mean_head0=((pred_score5[0]+pred_score6[0]+pred_score7[0]+pred_score8[0])/4)

    return model2(mean_head0,decision2,mean_head)

if __name__ == "__main__":
    result = scoring('6 Possible Hurdles For The GOP Tax Plan')
    result2 = scoring('6 Possible Hurdles For The GOP Tax Plan')
    print(result,result2)