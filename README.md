# cracking-captcha-tensorflow
building captcha cracking model with low computing power but highly effective

# Guide to use
simple.

just clone the repository to your own drive and run the trainning.py on your own terminal.

if you want to train the whole model by yourself from scratch, simply just comment out the line 52 of trainning.py and undo the comment for line 50 which is the init of the variables.

if you wanna try how amazing it is for cracking captcha, you can download the pretrained weights which has been just trained for 7hrs but already can reach 93% accuracy.

Download Link: [Pretrained Weight](https://drive.google.com/open?id=1R2brFB8ZuIGaDZnJG612cPN4lRw22jpN)


# performance
I run it on my own macbook pro is about 45mins for 1000 iterations. it start converge at about 730 iterations and achieve 50% accuracy at step 1400 which is rougly takes about a bit more than 1hrs.To reaches 90% accuracy just need 6-7 hrs.


# dependency
python3.6

tensorflow

numpy

matplotlib

termcolor

tflearn

# TODO
The next stage of machine learning is probably the generative machine learning model.AS RCN (namely recursive cortical network) appear, it's performance for cracking captcha has defeat cnn or rnn totally. So lets study RCN!

But for machine learning beginner, cracking captcha is a good project after creating models for mnist. 

below is the link for RCN:

[RCN for mnist](https://github.com/vicariousinc/science_rcn)

[RCN paper](https://drive.google.com/open?id=1d9yZi0DYYtyY9BhunYZYBsBimBGD4S0y)
