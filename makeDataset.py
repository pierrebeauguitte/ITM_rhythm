import sys
import csv
import os
import glob
import json
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from madmom.audio import signal, spectrogram, stft, filters

def getBarkSpec(path, start, stop, nbins = 1024):
    sig = signal.Signal(path, start=start, stop=stop, num_channels=1)
    fsig = signal.FramedSignal(sig, frame_size = nbins*2)
    stspec = stft.ShortTimeFourierTransform(fsig)
    spec = spectrogram.Spectrogram(stspec)
    fft_bins = stft.fft_frequencies(nbins, 44100)
    bark_freq = filters.bark_double_frequencies()
    bark_bins = filters.frequencies2bins(bark_freq, fft_bins)
    center_bins = []
    bark_filters = []
    for i in range(0, len(bark_freq) - 2, 2):
        bark_filters.append(filters.TriangularFilter(bark_bins[i],
                                                     bark_bins[i+1],
                                                     bark_bins[i+2]))
        center_bins.append(bark_bins[i+1])
    fb = filters.Filterbank.from_filters(bark_filters, fft_bins)
    filtered = spectrogram.FilteredSpectrogram(spec, fb, norm_filters = True)
    return filtered

def detectOnsets(spec):
    def h(x):
        return (x + abs(x)) / 2.
    nFrames = spec.shape[0]
    diff = np.empty(nFrames)
    diff[0] = 0
    for t in range(nFrames - 1):
        diff[t+1] = np.sum( h(spec[t+1] - spec[t]) ** 2 )
    return diff

def serial_corr(sig, lag):
    n = len(sig)
    y1 = sig[lag:]
    y2 = sig[:n-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    return corr

def autocorr(sig):
    lags = range(len(sig)//2)
    corrs = [serial_corr(sig, lag) for lag in lags]
    return lags, corrs

def movingACF(f, step, window=500):
    nACF = ((len(f) - window) / step) + 1
    tmp = []
    for t in range(nACF):
        _, c = autocorr(f[t*step : t*step + window])
        tmp.append(c)
    return np.stack(tmp)

def peakPicking(odf):
    peaks = []
    values = []
    for t in range(1, len(odf) - 1):
        if odf[t-1] < odf[t] and odf[t] > odf[t+1]:
            peaks.append(t)
            values.append(odf[t])
    return peaks, values

def fuzzyHistogram(peaks):
    histogram = []
    for n in range(len(peaks)):
        previous = 0 if (n==0) else peaks[n-1]
        d = peaks[n] - previous
        for bin in histogram:
            bin_start = bin[0] * (2 / 3.)
            bin_end   = bin[0] * (4 / 3.)
            if d >= bin_start and d <= bin_end:
                # we found a fitting existing bin
                bin[0] = (bin[0] * bin[1] + d) / float(bin[1] + 1)
                bin[1] += 1
                break
        else:
            # if the loop didn't reach break, create a new bin
            histogram.append( [ d, 1 ] )
    histogram.sort(key = lambda x: x[1])
    return histogram[-1][0]

def quantizePeaks(peaks, vals):

    count = np.zeros(16)
    values = np.zeros(16)
    q = fuzzyHistogram(peaks)
    for p in range(len(peaks)):
        rounded = int(round(peaks[p]/q))
        if rounded < 1:
            continue
        if rounded > 16:
            break
        count[rounded-1] += 1
        values[rounded-1] += vals[p]

    for i in range(16):
        if count[i] > 1:
            values[i] /= count[i]

    return list(values)

def prepareSets(dataset, output, step = 50):

    with open(dataset, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if os.path.isfile('%s/%03d.json' % (output, int(row['index']))):
                print 'skipping %s - %s...' % (row['index'], row['file'])
                continue

            print 'preparing %s - %s...' % (row['index'], row['file'])

            s = getBarkSpec('../soundfiles/%s' % row['file'],
                            int(row['start']), int(row['end']))
            df = detectOnsets(s)
            macf = movingACF(df, step)
            hists = {}
            pos = 0
            for a in macf:
                fa = gaussian_filter1d(a,2)
                p, v = peakPicking(fa)
                qp = quantizePeaks(p, v)

                hists[pos] = qp
                pos += step
            with open('%s/%03d.json' % (output, int(row['index'])), 'w') as outfile:
                json.dump(hists, outfile)

def prepareChunks(dataset, output, step = 50):

    with open(dataset, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:

            for t in range(1, 5):
                
                start = int(row['t%d' % t])
                index = int(row['index'])
                filename = '%s/%03d.%d.json' % (output, index, start)
                
                if os.path.isfile(filename):
                    print 'skipping %s...' % (filename)
                    continue
                print 'preparing %s...' % (filename)
                s = getBarkSpec('../soundfiles/%s' % row['file'],
                                start, start + 12)
                df = detectOnsets(s)
                macf = movingACF(df, step)
                if len(macf) != 15:
                    print '\tSig length %d, nacf %d' % (len(df), len(macf))
                hists = {}
                pos = 0
                for a in macf:
                    fa = gaussian_filter1d(a,2)
                    p, v = peakPicking(fa)
                    qp = quantizePeaks(p, v)

                    hists[pos] = qp
                    pos += step
                with open(filename, 'w') as outfile:
                    json.dump(hists, outfile)

if __name__ == '__main__':

    if sys.argv[1] == 'tunes':
        prepareSets('dataset.csv', 'data')
    elif sys.argv[1] == 'chunks':
        prepareChunks('dataset.csv', 'data_chunks')
