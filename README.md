# cracking-captcha-tensorflow
building captcha cracking model with low computing power but highly effective

# Guide to use
<p>simple.</p>
<p>just clone the repository to your own drive and run the trainning.py on your own terminal.</p>
<p>if you want to train the whole model by yourself from scratch, simply just comment out the line 52 of trainning.py and undo the comment for line 50 which is the init of the variables.</p>

# performance
I run it on my own macbook pro is about 45mins for 1000 iterations. it start converge at about 730 iterations and achieve 50% accuracy at step 1400 which is rougly takes about a bit more than 1hrs.


# dependency
<p>tensorflow</p>
<p>numpy</p>
<p>matplotlib</p>
<p>termcolor</p>
<p>tflearn</p>
