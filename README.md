# nonlinear_prediction
Matlab files for non-linear prediction

This matlab code makes use of dynamic system theory and investigates its application for fault estimation by analysing non-stationarities which arise due to the changing dynamics under intermittent conditions. Intermittent fault detection presents a challenge for traditional fault diagnostic equipment as they do not manifest themselves all the time and disappear in an unpredictable manner. They can be a symptom of the degradation of some physical property of a component. The idea was to move away from the traditional conditional monitoring approaches and investigate the use of non-linear analysis by building a reference trajectory using the phase space reconstruction. The "n-d phase space conversion" section of the code will convert a univariate signal into its phase space representation. This can be used as an objective measure for any deviations when an intermittent phenomena. The "prediction" section of the code will attempt to reconstruct what the future signal should look like and compare the result with the actual response. Thus it calculates the residual. The final part of the code (which has been omitted in this publication) attempts to regulate the sensitivity of the residual results. The implications of the study was to identify new fault isolation bounds necessary to improve diagnostic success rates and potentially lead to early diagnosis of intermittent faults in electrical equipment. 