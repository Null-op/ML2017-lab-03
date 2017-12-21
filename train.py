from PIL import Image
import pickle
import os
from feature import NPDFeature
import numpy as np
from ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# parameter
use_cache = True


def get_path(path):
    # 给路径添加前缀
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def conver_image(paths):
    ims = list()
    i = 0
    for path in paths:
        im = Image.open(path)
        im = im.convert(mode='L')
        im.thumbnail(size=(24, 24))
        im = np.asarray(im)
        im = NPDFeature(im)
        im = NPDFeature.extract(im)
        print(i)
        i +=1
        ims.append(im)
    return ims


if __name__ == "__main__":
    if os.stat("cache").st_size> 1024 and use_cache  :
        # 已有缓存
        with open('cache', mode='rb') as cache:
            dataset = pickle.load(cache)
        print("load cache done")
    else:
        # 没有缓存
        faces_path = get_path('./datasets/original/face')
        nonface_path = get_path('./datasets/original/nonface')
        dataset = dict()
        dataset['face'] = conver_image(faces_path)
        dataset['nonface'] = conver_image(nonface_path)
        with open('cache', mode='wb') as cache:
            pickle.dump(dataset, file=cache)
            print('save cache done')

    # conduct x_train, x_test, y_train, y_test
    y_true = [[1]] * len(dataset['face'])
    y_false = [[-1]] * len(dataset['nonface'])
    X = dataset['face'] + dataset['nonface']
    Y = y_true + y_false
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    print("conduct train set and testing set done")

    # fit
    classsfier = AdaBoostClassifier(tree.DecisionTreeClassifier, 5)
    classsfier.fit(x_train, y_train)

    # predict
    y_predict = classsfier.predict(x_test)


    # conduct report
    y_truth = [i[0] for i in y_test]
    y_pred = [i[0] for i in y_predict]
    report = classification_report(y_truth, y_pred)
    with open('report.txt', 'w') as report_file:
        report_file.write(report)
    print(report)
