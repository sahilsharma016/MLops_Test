from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

print(X.shape)
 

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =  train_test_split(X,y ,test_size=0.2,random_state=42)


ss= StandardScaler()
X_transform_train = ss.fit_transform(X_train)
X_transform_test = ss.transform(X_test)


tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_transform_train,y_train)

tree.score(X_transform_test,y_test)


from joblib import dump,load
dump(tree, "final_model.joblib")
dump(ss,"standard_scaler.joblib")
print("done")

