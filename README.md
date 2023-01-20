# Phase bias work while at SatSense

To plot loop closure and plot Figure 2 form Yasser's paper:

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

	python empCorrection_TS.py -d "20210101,20220101" -p 0 -s 1
	
	w: Working directory 
	f: Frame ID
	d: date range: YYYYmmdd,YYYYmmdd
	m: Multilook factor: nr,na
	p: Plot output: bool (Don't do if input is N>5)
	s: Save output arrays: bool
	
	
