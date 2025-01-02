#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:39:11 2018

@author: annabelle
"""

import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
import random
import pickle


def loading(cases, doping, maxShots, visualize=False):
    sample = []
    labels = []
    sms_fin = []
    ps = 1.0
    cnt = 0
    for case in cases:
        print(case)
        if case == 'hf':
            pickle_in = open("StringDataDopingGrouped/Doping exp/d" + str(0.0) + ".pkl", "rb")
        elif case.startswith('pi'):
            pickle_in = open("StringDataDopingGrouped/Doping " + case + "/big_d" + str(doping) + ".pkl", "rb")
        else:
            pickle_in = open("StringDataDopingGrouped/Doping " + case + "/d" + str(doping) + ".pkl", "rb")

        am = pickle.load(pickle_in, encoding='latin1')

        sample_tmp = []
        labels_tmp = []
        sms = []
        # shots=random.sample(range(len(snapshots)), k=sizes[cnt])
        k = 0
        for img in am:
            k += 1
            if k > 1:
                counter = 0
                for m in img:
                    m = np.array(m)
                    # print(m)
                    #
                    if visualize:
                        cmap = mlp.colors.ListedColormap(['white', 'lightseagreen', 'red'])
                        bounds = [-1.5, -0.5, 0.5, 1.5]
                        norm = mlp.colors.BoundaryNorm(bounds, cmap.N)
                        img = plt.imshow(m, interpolation='nearest', cmap=cmap, norm=norm)
                        # make a color bar
                        # plt.colorbar(img,cmap=cmap,norm=norm,boundaries=bounds,ticks=[-1,1])
                        plt.show()
                    #
                    # no mask necessary for full info snapshots
                    if case.startswith('pi') and not case.endswith("FI"):  # or case=='qmc':
                        # mask for aligned exp. data
                        mask1 = np.zeros([16, 16])
                        mask1[1][4:8] = [1, 1, 1, 1];
                        mask1[10] = mask1[1]
                        mask1[2][2:10] = [1, 1, 1, 1, 1, 1, 1, 1];
                        mask1[9] = mask1[2];
                        mask1[3] = mask1[2];
                        mask1[8] = mask1[2];
                        mask1[4][1:11] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
                        mask1[5] = mask1[4];
                        mask1[6] = mask1[4];
                        mask1[7] = mask1[4];
                        roiCutout = np.where(mask1 == 0)
                        m[roiCutout] = 0
                    elif case.endswith("FI"):
                        # cut out smaller square from 16x16 system
                        mask1 = np.zeros([16, 16])
                        mask1[1][1:11] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
                        mask1[2] = mask1[1];
                        mask1[3] = mask1[1];
                        mask1[4] = mask1[1];
                        mask1[5] = mask1[1];
                        mask1[6] = mask1[1];
                        mask1[7] = mask1[1];
                        mask1[8] = mask1[1];
                        mask1[9] = mask1[1];
                        mask1[10] = mask1[1]
                        roiCutout = np.where(mask1 == 0)
                        m[roiCutout] = 0

                    # cut out small square around ROI
                    roi = np.where(m != 0)
                    # find corners of big roi
                    xMin = max(0, min(np.unique(roi[0])) - 2)
                    xMax = min(max(np.unique(roi[0])) + 2, m.shape[0])
                    yMin = max(0, min(np.unique(roi[1])) - 2)
                    yMax = min(m.shape[1], max(np.unique(roi[1])) + 2)
                    # m=np.transpose(np.transpose(m[xMin:xMax+1])[yMin:yMax+1])
                    mX = m.shape[0]
                    mY = m.shape[1]
                    ### move window to optimal position
                    m_temps = []
                    sms_temps = []

                    #                    upperX=min(xMax+1-6,mX-6)
                    #                    upperY=min(yMax+1-6,mY-6)

                    upperX = min(xMax + 1 - 7, mX - 7)
                    upperY = min(yMax + 1 - 7, mY - 7)
                    #
                    #                    upperX=min(xMax+1-8,mX-8)
                    #                    upperY=min(yMax+1-8,mY-8)
                    #
                    #                    for x1 in range(0,upperX):#xMax+1): #(xMin+2,xMin+3):
                    #                        for y1 in range(0,upperY):#yMax+1):#(yMin+2,yMin+3):
                    #                            m1=m.copy()
                    #                            mask=np.zeros([mX,mY])
                    #
                    ##                            mask[x1][y1+1:y1+5]=[1,1,1,1]; mask[x1+5]=mask[x1]
                    ##                            mask[x1+1][y1:y1+6]=[1,1,1,1,1,1];
                    ##                            mask[x1+2]=mask[x1+1];mask[x1+3]=mask[x1+1];mask[x1+4]=mask[x1+1]
                    ##
                    #                            mask[x1][y1+2:y1+5]=[1,1,1]; mask[x1+6]=mask[x1]
                    #                            mask[x1+1][y1+1:y1+6]=[1,1,1,1,1];mask[x1+5]=mask[x1+1]
                    #                            mask[x1+2][y1:y1+7]=[1,1,1,1,1,1,1];mask[x1+3]=mask[x1+2]
                    #                            mask[x1+4]=mask[x1+2]
                    #
                    ##                            mask[x1][y1+2:y1+6]=[1,1,1,1]; mask[x1+7]=mask[x1]
                    ##                            mask[x1+1][y1+1:y1+7]=[1,1,1,1,1,1];mask[x1+6]=mask[x1+1]
                    ##                            mask[x1+2][y1:y1+8]=[1,1,1,1,1,1,1,1];mask[x1+3]=mask[x1+2]
                    ##                            mask[x1+4]=mask[x1+2];mask[x1+5]=mask[x1+2]
                    #
                    #                            roiCutout=np.where(mask==0)
                    #                            m1[roiCutout]=0
                    #                            totalSites_mask=sum(sum(abs(mask)))
                    #                            totalSites=sum(sum(abs(m1)))
                    #                            if totalSites==totalSites_mask: #only take position if window completely in big ROI
                    #                                # get stagg mag
                    #                                sm=0;
                    #                                for i in range(x1-1,min(x1+12,mX)):
                    #                                    for j in range(y1-1,min(y1+12,mY)):
                    #                                        sm=sm+(-1)**(i+j)*m1[i,j]
                    #                                m_temps.append(m1)
                    #                                sms_temps.append(abs(sm))
                    #
                    #                    indices=sorted(range(len(sms_temps)), key=lambda k: -abs(sms_temps[k]))
                    #                    #m=m_temps[indices[0]]
                    #                   # print len(indices)

                    # cut out small square around new smaller ROI
                    roi = np.where(m != 0)
                    # find corners of new small roi
                    xMin = min(np.unique(roi[0]))
                    xMax = max(np.unique(roi[0]))
                    yMin = min(np.unique(roi[1]))
                    yMax = max(np.unique(roi[1]))
                    if (xMax - xMin + 1) % 2 != 0:
                        xMax = xMax + 1
                    m = np.transpose(np.transpose(m[xMin:xMax + 1])[yMin:yMax + 1])
                    mX = m.shape[0]
                    mY = m.shape[1]

                    # get difference picture
                    # initiate Neel state
                    ref = np.ones([m.shape[0], m.shape[1]])
                    for i in range(m.shape[0]):
                        for j in range(m.shape[1]):
                            ref[i, j] = (-1) ** (i + j)
                    # get stagg mag
                    sm = 0;
                    for i in range(m.shape[0]):
                        for j in range(m.shape[1]):
                            sm = sm + (-1) ** (i + j) * m[i, j]
                    if sm < 0:
                        ref = ref * (-1)
                    mDiff = abs(m - ref)
                    # set corners to 'not red areas' (only 0s and 1s)
                    ones = np.where(mDiff == 1)
                    mDiff[ones] = 0
                    mDiff = 0.5 * mDiff

                    #
                    #                    cmap = mpl.colors.ListedColormap(['white','lightseagreen','red'])
                    #                    bounds=[-1.5,-0.5,0.5,1.5]
                    #                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    #                    img = plt.imshow(m,interpolation='nearest',cmap = cmap,norm=norm)
                    #                    # make a color bar
                    #                    #plt.colorbar(img,cmap=cmap,norm=norm,boundaries=bounds,ticks=[-1,1])
                    #                    plt.show()
                    #

                    # flatten the picture
                    diffReshaped = np.reshape(mDiff, m.shape[0] * m.shape[1])
                    mReshaped = np.reshape(m, m.shape[0] * m.shape[1])
                    sample_tmp.append(mReshaped)
                    # get label in form of 'one hot vector':
                    label_map = np.zeros(len(cases))
                    label_map[cnt] = 1  # set entry for current case to one

                    labels_tmp.append(label_map)
                    sms.append(abs(sm))

        # take a maximum number of shots per case to avoid bias in training!
        smple = random.sample(range(0, len(sample_tmp)), min(maxShots, len(sample_tmp)))
        sample_tmp = [sample_tmp[i] for i in smple]
        labels_tmp = [labels_tmp[i] for i in smple]
        sms = [sms[i] for i in smple]

        indices = sorted(range(len(sms)), key=lambda k: -abs(sms[k]))
        sample_tmp = [sample_tmp[i] for i in indices]
        labels_tmp = [labels_tmp[i] for i in indices]
        sms_tmp = [sms[i] for i in indices]

        # get mean value of stagg mag of snapshots taken into account 
        print('mean stagg mag taken: ', str(np.mean(sms_tmp[:int(ps * len(sample_tmp))])))
        print('mean stagg mag: ', str(np.mean(sms)))
        sms_fin.append(np.mean(sms_tmp[:int(ps * len(sample_tmp))]))

        hist, bin_edges = np.histogram([abs(sm) for sm in sms_tmp], bins=np.arange(1, 50, 2.0))  # 0.66,0.06))
        hist = [1.0 * h / len(sms_tmp) for h in hist]
        plt.plot(bin_edges[:-1], hist, '--^')
        plt.show()

        print(str(len(sample_tmp)), ' snapshots')

        for i in range(int(ps * len(sample_tmp))):
            sample.append(sample_tmp[i])
            labels.append(labels_tmp[i])

        cnt += 1

    # this is now done in cnn.py because we only load snapshots once for each doping value, then re-shuffle for different trials to get a different test-dataset in each run
    #    # shuffle snapshots: (because the first 100 are reserved for testing --> makes no sense if they are sorted!)
    #    smple=random.sample(range(0, len(sample)), len(sample))
    #    sample=[sample[i] for i in smple]
    #    labels=[labels[i] for i in smple]

    return sample, labels, sms_fin
