# Phase bias work while at SatSense

## loopClosure.py:
Script that calculates the loop closure for a stack of interferograms. It can handle different amounts of multilooking and different lengths of loops.  

	python loopClosure.py -i 0 -d 2 -m 3,12 -p 1 -s 0    

	i: starting index (of images) 

	d: delta of loop (eg. d=2, jumpy two images forward to create 12 day loop) 

	m: Multilooking size (str 3,12) 

	p: plot (boolean, 1=True, 0=False) 

	s: save plot (boolean, 1=True, 0=False) 

 

This can be ran as a loop to process a whole timeseries: 

	delta=2 

	for d in {286..342}; do python loopClosure.py -i $d -d $delta -m 12,48; done 

Another script does some post processing to produce figures given a directory of loop closure data: 

	python loopClosure_post.py 

 

To recreate Figure 3 from Yasser’s paper: 

	python Fig3_Yasser.py -i 274 -m 0 -g 1 

	i: start ifg 

	m: length of m-day ifg (see paper for explanation). if m != 0, the code will produce a single image with n=60 and m=m, while if m==0 then it will reproduce the whole figure from m=6, 12, ..., 36.

	g: grid boolean. If True (1), segments the data by taking the mean of each segment for the plots. Helps to show deviations from 0 for the loop closure in the maps. 

## Implementing the empirical correction:

	python empCorrection_TS.py -d "20210101,20220101" -p 0 -s 1
	
	d: date range
	p: Plot output (Don't do if input is N>5)
	s: Save output arrays.
