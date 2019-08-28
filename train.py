import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import json
import sys
import argparse
import os
import pickle
from math import sqrt, log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, brier_score_loss, balanced_accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def printConfMat(cm, tunecm, labels, transpose=False, normalise=False):

    confMat = cm.T if transpose else cm
    tuneConfMat = tunecm.T if transpose else tunecm

    totalPerClass = np.sum(confMat, axis = 0, dtype = np.float32)
    if normalise:
        confMat /= totalPerClass

    tunePerClass = np.sum(tuneConfMat, axis = 0, dtype = np.float32)

    latex = '\\begin{tabular}{l'
    for _ in labels:
        latex += 'c'
    latex += '}\n\\toprule\n '

    for l in range(len(labels)):
        latex += '& %s ' % labels[l]
    latex += '\\\\\n\\midrule\n'

    for l in range(len(labels)):
        latex += labels[l]
        for i in confMat[l]:
            if normalise:
                latex += ' & %.2f' % (i * 100)
            else:
                latex += ' & %d' % i
        latex += '\\\\\n'

    latex += '\\midrule\n'

    for l in range(len(labels)):
        latex += labels[l]
        for i in tuneConfMat[l]:
            latex += ' & %d' % i
        latex += '\\\\\n'
    latex += '\\midrule\n'
    latex += 'accuracy (\\%)'
    for l in range(len(labels)):
        latex += ' & %.2f' % (100 * tuneConfMat[l,l] / tunePerClass[l])
    latex += '\\\\\n'

    latex += '\\bottomrule\n'
    latex += '\\end{tabular}'

    return latex

def cleanFilename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def loadReference(referenceFile, modelType):

    tunes = {}
    perID = {}

    binary_types = {
        '(hop) jig': 'compound',
        'jig': 'compound',
        'waltz': 'simple',
        'fling': 'simple',
        'slip jig': 'compound',
        'polka': 'simple',
        'barndance': 'simple',
        'slide': 'compound',
        'hornpipe': 'simple',
        'mazurka': 'simple',
        'highland': 'simple',
        'reel': 'simple'
    }

    multi_types = {
        '(hop) jig': 'slipjig',
        'jig': 'jig',
        'waltz': 'waltz',
        'fling': 'other44',
        'slip jig': 'slipjig',
        'polka': 'polka',
        'barndance': 'other44',
        'slide': 'slide',
        'hornpipe': 'hornpipe',
        'mazurka': 'waltz',
        'highland': 'other44',
        'reel': 'reel'
    }

    with open(referenceFile, 'r') as ref:
        reader = csv.DictReader(ref)
        for row in reader:
            tuneIndex = int(row['index'])
            tuneType = multi_types[row['type']] if modelType == 'multinomial' \
                       else binary_types[row['type']]
            if tuneType not in tunes:
                tunes[tuneType] = []
            tunes[tuneType].append(tuneIndex)
            perID[tuneIndex] = tuneType

    return tunes, perID

def prepareSets(path, tuneDict):

    x = [] # quantized peaks
    y = [] # label
    t = [] # tune index
    minNWindows = None

    dataFiles = glob.glob('%s/*.json' % path)
    for tune in dataFiles:
        tuneIndex = int(cleanFilename(tune)[:3])
        if tuneIndex not in tuneDict:
            continue
        target = tuneDict[tuneIndex]
        fp = open(tune, 'r')
        data = json.load(fp)
        ts = map(int, data.keys())
        ts.sort()
        if len(ts) < minNWindows or minNWindows is None:
            minNWindows = len(ts)
        for a in ts:
            x.append(data[str(a)])
            y.append(target)
            t.append(tuneIndex)

    return x, y, t, minNWindows

def makeSplits(instances, nFolds):
    splits = {i:[] for i in instances}
    for tuneType in instances:
        tuneIds = np.array(instances[tuneType])
        shuffle = np.random.permutation(len(tuneIds))
        splitSize = len(tuneIds) / float(nFolds)
        for i in range(nFolds - 1):
            splits[tuneType].append(
                list(tuneIds[ shuffle[int(np.round(i*splitSize)) :
                                      int(np.round((i+1)*splitSize))] ])
            )
        splits[tuneType].append(
            list(tuneIds[ shuffle[int(np.round((nFolds-1)*splitSize)):] ])
        )
    return splits

def getScores(fy, fp, ft, wLen, model):
    tindices = np.array(ft)
    indices = set(ft)
    globalRes = []
    for i in indices:
        loc = np.where(tindices == i)[0]
        res = []
        for w in range(len(loc) - wLen + 1):
            avgProbs = []
            for typeIndex in range(len(model.classes_)):
                avgProbs.append(np.average(fp[loc[w:w+wLen], typeIndex]))
            spanProb = model.classes_[np.argmax(avgProbs)]
            res.append(1 if spanProb == fy[loc[w]] else 0)
        globalRes.extend(res)
    return globalRes

def prepareTrainAndTest(path, referenceFile, modelType):

    print '\n---- TRAINING ON FULL DATASET ----\n'

    nFolds = 4 if modelType == 'multinomial' else 10

    labels = [
        'reel',
        'jig',
        'slide',
        'slipjig',
        'hornpipe',
        'polka',
        'other44',
        'waltz'
    ] if modelType == 'multinomial' else ['simple', 'compound']

    t, ref = loadReference(referenceFile, modelType)
    t2, ref2 = loadReference(referenceFile, 'multinomial')

    # unnecessarily rewritten every time, but no harm done
    with open('ref_%s.csv' % modelType, 'w') as refFile:
        for tune in range(1, 501):
            refFile.write('%d,%s\n' % (tune, ref[tune]))

    X, Y, names, minN = prepareSets(path, ref)
    Ynp = np.array(Y)
    print np.where(Ynp == 'simple')[0].shape, 'simple data'
    print np.where(Ynp == 'compound')[0].shape, 'compound data'

    splits = makeSplits(t, nFolds)

    folds = {}
    for a in splits:
        for b in range(nFolds):
            for c in splits[a][b]:
                folds[c] = b

    with open('folds_%s.csv' % modelType, 'w') as outfile:
        for i in range(1, 501):
            outfile.write('%03d, %d\n' % (i, folds[i]))    

    print 'Starting cross-validation'

    wLens = range(1, minN + 1)
    print minN, 'minimum number of windows'
    finalRes = {s: [] for s in wLens}

    confMat = np.zeros((len(labels),len(labels)), dtype = int)
    tuneConfMat = np.zeros((len(labels),len(labels)), dtype = int)
    tuneErrors = {}

    predPerType = {
        'reel': [],
        'jig': [],
        'slide': [],
        'slipjig': [],
        'hornpipe': [],
        'polka': [],
        'other44': [],
        'waltz': []
    }

    for i in range(nFolds):
        print 'Fold %d' % i
        testId = [ s[i] for s in splits.values() ]
        testId = [ x for s in testId for x in s ] # flatten list of lists

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        test_t = []
        for j in range(len(X)):
            if names[j] in testId:
                test_x.append(X[j])
                test_y.append(Y[j])
                test_t.append(names[j])
            else:
                train_x.append(X[j])
                train_y.append(Y[j])

        logreg = LogisticRegression(class_weight = 'balanced',
                                    solver='liblinear',
                                    multi_class = 'ovr')
        logreg.fit(train_x, train_y)

        score = logreg.score(test_x, test_y)
        print 'Accuracy on test set: %.3f' % score

        with open('models/%s_%d.pcl' % (modelType, i), 'w') as modFile:
            pickle.dump(logreg, modFile)

        prediction = logreg.predict(test_x)

        prediction_proba = logreg.predict_proba(test_x)

        cm = confusion_matrix(test_y, prediction,
                              labels = labels)
        confMat += cm

        if modelType == 'binomial':
            for ind in range(len(test_x)):
                predPerType[ref2[test_t[ind]]].append(prediction[ind])

        for r in wLens:
            finalRes[r].extend(getScores(test_y, prediction_proba, test_t, r, logreg))

        tuneRes = {}
        tindices = np.array(test_t)
        indices = set(test_t)
        for ind in indices:
            loc = np.where(tindices == ind)
            res = []
            avgProbs = []
            for typeIndex in range(len(logreg.classes_)):
                avgProbs.append(np.average(prediction_proba[loc, typeIndex]))
                tuneProb = logreg.classes_[np.argmax(avgProbs)]
            tuneRes[ind] = tuneProb

        for k in tuneRes:
            index_p = labels.index(tuneRes[k])
            index_ref = labels.index(ref[k])
            tuneConfMat[ index_ref, index_p ] += 1
            if index_p != index_ref:
                tuneErrors[k] = tuneRes[k]

    print 'Frame accuracy: %.2f%%' % (100. * np.trace(confMat) / np.sum(confMat))

    print 'Tune accuracy: %.2f%%' % (100. * np.trace(tuneConfMat) / np.sum(tuneConfMat))

    print printConfMat(confMat.astype(np.float32),
                       tuneConfMat,
                       labels, transpose = True, normalise = True)

    print '\nWrong predictions on tunes:'
    for te in tuneErrors:
        print '\t%s (%s), recognised as %s' % (te, ref[te], tuneErrors[te])

    if modelType == "binomial":
        for tuneType in ['jig', 'slide', 'slipjig']:
            c = 0
            for p in predPerType[tuneType]:
                if p=='compound':
                    c+=1
            print '>> %s: %.5f' % (tuneType, c / float(len(predPerType[tuneType])))

        for tuneType in ['reel', 'hornpipe', 'polka', 'other44', 'waltz']:
            c = 0
            for p in predPerType[tuneType]:
                if p=='simple':
                    c+=1
            print '>> %s: %.5f' % (tuneType, c / float(len(predPerType[tuneType])))

    plotValues = []
    for r in wLens:
        plotValues.append(100. * sum(finalRes[r]) / float(len(finalRes[r])))

    
    print "Highest span acc [%d]: %.2f" % (len(plotValues), plotValues[-1])

    plt.figure(figsize=(5, 4))
    plt.plot(wLens, plotValues)
    plt.xlabel("length of window span")
    plt.ylabel("prediction accuracy (%)")
    plt.savefig("figures/acc_span_%s.pdf" % modelType, bbox_inches='tight')

    return wLens, plotValues

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelType', required=True,
                        help='type of model to train (binomial/multinomial)')
    args = parser.parse_args()

    if args.modelType == 'binomial' or args.modelType == 'multinomial':
        np.random.seed(1564)
    else:
        print 'modelType should be \'binomial\' or \'multinomial\''
        sys.exit(1)

    prepareTrainAndTest('data', 'dataset.csv', args.modelType)
