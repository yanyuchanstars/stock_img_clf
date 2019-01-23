import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.models import load_model
from keras import models
import vectorization as vr
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import pydotplus

# 将cnn的第一层dense层的特征提取，输入随机森林和决策树分类器

def data(shuff=False):
    x_train, Y_onhot = vr.ImgVectorization('train',shuff=shuff).vec_all()
    x_val, y_onehot = vr.ImgVectorization('val',shuff=shuff).vec_all()
    y_train=vr.ImgVectorization('train').lable_list()
    y_val=vr.ImgVectorization('val').lable_list()
    return x_train,y_train, x_val,y_val,Y_onhot,y_onehot


def get_layer_features(model,x_data,layer_name,channels):  # cnn中dense层特征提取
    layer_model = models.Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    i=0
    features = np.zeros(shape=(x_data.shape[0], channels))
    for x in x_data:
        x = np.expand_dims(x, axis=0)
        layer_output = layer_model.predict(x)
        features[i] = layer_output
        i += 1
    feature_col = []
    for r in range(channels):
        feature_col.append(str(r))
        r += 1
    df = pd.DataFrame(data=features, columns=feature_col)
    return df

def fit_random_forest(x_train,y_train,x_val,y_val):  # 随机森林分类器
    rf = RandomForestClassifier(max_depth= 11,min_samples_leaf= 40, min_samples_split= 6, n_estimators=30,max_features='sqrt')
    rf.fit(x_train,y_train)
    # plot(rf,x_train)
    print(rf)
    predictions = rf.predict(x_val)
    acc = accuracy_score(predictions, y_val)
    print(acc)

def plot(clf,x):  # 随机森林可视化及特征重要性
    Estimators = clf.estimators_
    for index, model in enumerate(Estimators):
        filename = 'tree_' + str(index) + '.pdf'
        dot_data = tree.export_graphviz(model, out_file=None,
                                        feature_names=x.columns,
                                        class_names=['A_shape','D_shape','R_shape','U_shape'],
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(this_dir+os.sep+'randomforest'+os.sep+filename)
    df=pd.DataFrame({'features':x.columns,'importances':clf.feature_importances_})
    df=df.sort_values(by='importances',ascending=False).head(10)
    plt.bar(df.features, df.importances)
    plt.xticks(np.arange(len(df.features)),df.features)
    plt.ylabel('Importances')
    plt.title('Features Importances')
    plt.show()


def find_param(x,y):   # 随机森林调参
    param_test2 =  {'max_depth':range(10,20),'min_samples_split':range(5,15), 'min_samples_leaf':range(10,60,10),'n_estimators': range(10, 71, 10)}
    gsearch2 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_test2,
                            scoring='roc_auc', cv=5)
    gsearch2.fit(x, y)
    print(gsearch2.best_params_, gsearch2.best_score_)

def fit_tree(x_train,y_train,x_val,y_val):  # 决策树分类器
    clf = DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    print(clf)
    predictions = clf.predict(x_val)
    acc = accuracy_score(predictions, y_val)
    print(acc)

if __name__ == '__main__':
    this_dir= os.getcwd()
    result_dir = this_dir + os.sep + 'result'
    model_path=result_dir+os.sep+'cnn_80_0.2_16_10.h5'
    model = load_model(model_path)
    model.summary()
    os.environ["PATH"] += os.pathsep + path
    x_train, y_train,x_val,y_val,Y_onehot,y_onhot=data(shuff=False)  # 调参时需要用打乱的数据; fit的时候lable是按次序读文件夹名的，此时shuffle要设为false
    train_features=get_layer_features(model,x_train,'dense_1',128)
    val_features=get_layer_features(model,x_val,'dense_1',128)
    fit_random_forest(train_features,y_train,val_features,y_val)
    # find_param(train_features,Y_onehot)
    # fit_tree(train_features,y_train,val_features,y_val)