import MainAnalysis as ma
import ExpFile as exp
import MatplotlibPlotters as mp
import AnalysisHelpers as ah
import numpy as np

from fitters.Gaussian import gaussian_2d as ga2

exp.setPath('28','June','2021')
key, rawData, dataMinusBg, dataMinusAvg, avgPic,\
pictureFitParams, pictureFitErrors, plottedData,\
v_params, v_errs, h_params, h_errs, intRawData = \
ma.standardImages(54,loadType='mako2',cameraType='mako2',
                  key = np.arange(100),
                  fitPics=False)
initial_guess = (115,300,300,60,60,105)
ga2.f_notheta(np.meshgrid(np.linspace(0, rawData[0].shape[1]-1, rawData[0].shape[1]), 
                          np.linspace(0, rawData[0].shape[0]-1, rawData[0].shape[0])),
             *initial_guess)