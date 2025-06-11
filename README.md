# kz-visual-correspondence-via-graph-cuts

Original paper: *Computing Visual Correspondence with
Occlusions via Graph Cuts* (https://www.cs.cornell.edu/rdz/Papers/KZ-ICCV01-tr.pdf)

Follow-up writeup: *Kolmogorov and Zabih’s Graph Cuts Stereo Matching
Algorithm* (https://www.ipol.im/pub/art/2014/97/article.pdf)

### To generate disparity maps for each of the three test images, run the following 3 commands (may take up to 30 minutes)
```
python3 main.py Aloe
python3 main.py Baby
python3 main.py Bowling
```

### To evaluate the generated disparity maps for all three test images, run the following command (will run in <1 minute)
```
python3 evaluate.py
```
