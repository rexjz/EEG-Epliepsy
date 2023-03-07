import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import pre_process
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.metrics import f1_score
from sklearn.model_selection import LearningCurveDisplay, learning_curve

dataset = pre_process()
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['targets'], test_size=0.3,
                                                    random_state=109)
clf = svm.SVC(kernel='poly', coef0=.2, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

plt.figure(0)
svc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
svc_disp.plot()
plt.show()

plt.figure(1)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=clf.classes_)
cm_disp.plot()
plt.show()

plt.figure(2)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()

f1 = f1_score(y_test, y_pred, average='binary')
print(f1)

plt.figure(3)
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train)
display = LearningCurveDisplay(train_sizes=train_sizes,
                               train_scores=train_scores, test_scores=test_scores, score_name="Score")
display.plot()
plt.show()
