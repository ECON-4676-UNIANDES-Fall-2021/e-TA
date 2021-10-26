# %%
def fit_model_cross_approach(la):
    global sublists,train2 
    k=5
    x=sublists[la]
    dataframe=train2
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    la=int(dataframe.shape[0]/k)
    a=0
    cortes=[]
    for i in range(0,k):
        y=[]
        y.append(a)
        a+=la
        y.append(a)
        cortes.append(y)
    metrics=[]
    for i in cortes:
        test=dataframe[i[0]:i[1]]
        train=dataframe.drop(index=list(range(i[0],i[1])))
        clf = LogisticRegression(max_iter=20000).fit(train[x], train['Credit Default'])
        y_pred=clf.predict(test[x])
        metrics.append(accuracy_score(test['Credit Default'], y_pred))
    return sum(metrics)/k


# %%
