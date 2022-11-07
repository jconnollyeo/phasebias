# Phase bias work while at SatSense
I have created a script that calculates the loop closure for a stack of interferograms. It can handle different amounts of multilooking and different lengths of loops.  

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

m:  
