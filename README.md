# Phase bias work while at SatSense

To plot loop closure and plot Figure 2 from Yasser's paper (need to check that hte func to generate ifgs with different temp baselines is the correct way around):

	python loopClosure.py 
	
	w: Working directory (folder containing the IFG folder)
	i: Start index in the time series
	d: Loop delta (or length of loop). d=2 for a 12,6,6 phase closure
	m: Multilook factor
	s: Whether to save the output or not (if True, saves the loop closure plot and the array as a numpy binary file). 
	p: Whether to plot the output or not.

To recreate Figure 3 from Yasser’s paper: 

	python Fig3_Yasser.py -i 274 -m 0 -g 1 

	i: start ifg 
	m: length of m-day ifg (see paper for explanation). if m != 0, the code will produce a single image with n=60 and m=m, while if m==0 then it will reproduce the whole figure from m=6, 12, ..., 36.
	g: grid boolean. If True (1), segments the data by taking the mean of each segment for the plots. Helps to show deviations from 0 for the loop closure in the maps. 
	w: used to specify the working directory - where the ifgs are. Default: '/workspace/rapidsar_test_data/south_yorkshire/jacob2/IFG/singlemaster/*/*.*'
	a: multilook factor: nr,na
	
To recreate Figure 4 from Yasser's paper:

	python Fig4_Yasser.py -i 274 -m 0 -g 1
	
	Same arguments as the Fig3 script.

## Implementing the empirical correction:
Need to check that the loops are calculated the correct way around (this would effect the output correction and then how the correction is applied in the phase unwrapping. 

	python empCorrection_TS.py -d "20210101,20220101" -p 0 -s 1
	
	w: Working directory 
	f: Frame ID
	d: date range: YYYYmmdd,YYYYmmdd
	m: Multilook factor: nr,na
	p: Plot output: bool (Don't do if input is N>5)
	s: Save output arrays: bool
	
This script can be used to generate the a1 and a2 values. It is worth checking this over to make sure there aren't any more mistakes...
	
	python generate_a_variables.py
	Doesn't take cmd line arguments so will have to change paths from within the script. Saves the a1 and a2 arrays as a stacked array (2, i, j) called a_variables.npy. 

Additional scripts used to create plots
	
	python a_comparison_timeseries.py
	This script generates the loop closure time-series and corrected loop closure time series for aregion that can be specified in the script. 
	
	python a_comparison.py
	This script generates the loop closure for a set of dates and plots them. 
	
	python loopClosure_TS.py 
	This scirpt also calculates the loop closure for a time series and plots it. 
	
	
